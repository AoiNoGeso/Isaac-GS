"""ai4animationpy(Biped Locomotion)による歩行モーション生成"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "ai4animation"
DEMO_SUPPORT_DIR = Path(__file__).resolve().parent / "_ai4anim_demo_support"
MODEL_GLB = str(ASSETS_DIR / "Model.glb")
NETWORK_PT = str(ASSETS_DIR / "Network.pt")
POSTPROCESSOR_PT = str(ASSETS_DIR / "PostProcessor.pt")
GUIDANCES_DIR = ASSETS_DIR / "Guidances"

SEQUENCE_WINDOW = 0.5
SEQUENCE_LENGTH = 16
SEQUENCE_FPS = 30
PREDICTION_FPS = 10
CONTACT_POWER = 3.0
NETWORK_ITERATIONS = 3
SOLVER_ITERATIONS = 1
SOLVER_ACCURACY = 1e-3
TRAJECTORY_CORRECTION = 0.25
MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
TIMESCALE_SENSITIVITY = 5
SYNCHRONIZATION_SENSITIVITY = 5

# ai4animationpy Actor(FULL_BODY_NAMES)の骨階層
BONE_PARENTS = {
    "Hips": None,
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot",
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Spine3": "Spine2",
    "Neck": "Spine3", "Head": "Neck",
    "LeftShoulder": "Spine3", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine3", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
}


def _lazy_import_ai4anim():
    """ai4animationpy本体はSimulationApp起動後、初回使用時にのみimportする"""
    import sys

    if str(DEMO_SUPPORT_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_SUPPORT_DIR))
    import Definitions
    from ai4animation import (
        Actor,
        AI4Animation,
        FABRIK,
        FeedTensor,
        GuidanceModule,
        MotionModule,
        ReadTensor,
        RootModule,
        Rotation,
        Tensor,
        Time,
        TimeSeries,
        Transform,
        Vector3,
    )
    from LegIK import LegIK
    from Sequence import Sequence

    return dict(
        Definitions=Definitions, Actor=Actor, AI4Animation=AI4Animation, FABRIK=FABRIK,
        FeedTensor=FeedTensor, GuidanceModule=GuidanceModule, MotionModule=MotionModule,
        ReadTensor=ReadTensor, RootModule=RootModule, Rotation=Rotation, Tensor=Tensor,
        Time=Time, TimeSeries=TimeSeries, Transform=Transform, Vector3=Vector3,
        LegIK=LegIK, Sequence=Sequence,
    )


@dataclass
class SharedLocomotionModel:
    """Network.pt/PostProcessor.ptは全エージェントで共有する"""

    Model: object = field(default=None)
    PostProcessor: object = field(default=None)

    @classmethod
    def load(cls) -> "SharedLocomotionModel":
        model = torch.load(NETWORK_PT, weights_only=False)
        model.eval()
        post = torch.load(POSTPROCESSOR_PT, weights_only=False)
        post.eval()
        return cls(Model=model, PostProcessor=post)


class LocomotionAgent:
    """1体分のai4animationpy Actor + Locomotion状態ゲームパッド/マウス入力の代わりに
    外部から与える速度ベクトル(ワールド座標系)で駆動する"""

    def __init__(self, shared: SharedLocomotionModel, api: dict, world_offset_xy: tuple[float, float] = (0.0, 0.0)):
        self._api = api
        Actor, AI4Animation, Definitions = api["Actor"], api["AI4Animation"], api["Definitions"]
        TimeSeries, RootModule, GuidanceModule = api["TimeSeries"], api["RootModule"], api["GuidanceModule"]
        FABRIK, LegIK = api["FABRIK"], api["LegIK"]

        self.Model = shared.Model
        self.PostProcessor = shared.PostProcessor

        self.world_offset_xy = world_offset_xy

        self.Actor = AI4Animation.Scene.AddEntity(f"Actor_{id(self)}").AddComponent(
            Actor, MODEL_GLB, Definitions.FULL_BODY_NAMES, True
        )

        self.Synchronization = 0.0
        self.Timescale = 1.0
        self.ControlSeries = TimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootModule.Series(self.ControlSeries)
        self.RootControl = RootModule.Series(self.ControlSeries)

        self.GuidanceControl = GuidanceModule.Guidance(
            "Guidance", self.Actor.GetBoneNames(), self.Actor.GetPositions().copy()
        )
        self.GuidanceTemplates = _load_guidance_templates(api)
        self.SelectedGuidance = "Neutral" if "Neutral" in self.GuidanceTemplates else sorted(self.GuidanceTemplates)[0]
        self.GuidanceControl.Positions = self.GuidanceTemplates[self.SelectedGuidance].Positions.copy()

        self.Previous = None
        self.Sequence = None
        self.ContactBones = [
            Definitions.LeftAnkleName, Definitions.LeftBallName, Definitions.RightAnkleName, Definitions.RightBallName,
        ]
        self.ContactIndices = self.Actor.GetBoneIndices(self.ContactBones)
        self.LeftLegIK = LegIK(
            FABRIK(self.Actor.GetBone(Definitions.LeftHipName), self.Actor.GetBone(Definitions.LeftAnkleName)),
            FABRIK(self.Actor.GetBone(Definitions.LeftAnkleName), self.Actor.GetBone(Definitions.LeftBallName)),
        )
        self.RightLegIK = LegIK(
            FABRIK(self.Actor.GetBone(Definitions.RightHipName), self.Actor.GetBone(Definitions.RightAnkleName)),
            FABRIK(self.Actor.GetBone(Definitions.RightAnkleName), self.Actor.GetBone(Definitions.RightBallName)),
        )
        self.Timestamp = api["Time"].TotalTime
        self._pref_velocity_local_xz = (0.0, 0.0)

    def set_velocity_world_xy(self, vx: float, vy: float):
        """ワールド座標系(Z-up, x,y)での目標速度を、ai4animationpyのローカル座標系
        (Y-up, 地面=XZ)速度に変換して渡 すローカルZ->ワールド-Yの符号反転を吸収する"""
        self._pref_velocity_local_xz = (vx, -vy)

    def _hips_local_xz(self) -> tuple[float, float]:
        """Hipsボーンの現在位置(Actor.Entity基準ローカル)`Actor.GetRootPosition()`は
        歩行アルゴリズム内部のトラジェクトリ予測用の値でレンダリング結果とは別物のため、
        外部への位置報告には必ずこちら(Hipsボーン自身の位置)を使う"""
        pos = np.array(self.Actor.GetBone("Hips").GetTransform())[:3, 3]
        return float(pos[0]), float(pos[2])

    def ground_position_world_xy(self) -> tuple[float, float]:
        """ワールド座標系(world_offset_xy込み)での実際の(メッシュが動いた)位置"""
        local_x, local_z = self._hips_local_xz()
        return local_x + self.world_offset_xy[0], -local_z + self.world_offset_xy[1]

    def teleport_world_xy(self, new_world_xy: tuple[float, float]):
        """Actor内部の歩行状態には触れず、world_offset_xyだけを再計算し、次フレームの
        ground_position_world_xy()がnew_world_xyになるようにする(エピソードリセット用)"""
        local_x, local_z = self._hips_local_xz()
        self.world_offset_xy = (new_world_xy[0] - local_x, new_world_xy[1] - (-local_z))

    def _control(self):
        Vector3, Transform, Tensor = self._api["Vector3"], self._api["Transform"], self._api["Tensor"]
        Time = self._api["Time"]
        vx, vz = self._pref_velocity_local_xz
        velocity = Vector3.Create(vx, 0.0, vz)
        speed = float(np.linalg.norm([vx, vz]))
        direction = velocity if speed > 1e-4 else Vector3.Create(0.0, 0.0, 1.0)

        position = Vector3.Lerp(self.SimulationObject.GetPosition(0), self.Actor.GetRootPosition(), self.Synchronization)
        self.SimulationObject.Control(position, direction, velocity, Time.DeltaTime)

        if self.Sequence is not None:
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms, self.Sequence.Trajectory.Transforms, TRAJECTORY_CORRECTION
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = self.Actor.GetRootPosition().reshape(-1, 3)
                t = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(target - current, axis=0, keepDim=False) / Tensor.Sum(t, axis=0, keepDim=False)
            self.RootControl.Velocities = Vector3.Lerp(self.RootControl.Velocities, self.Sequence.Trajectory.Velocities, TRAJECTORY_CORRECTION)

    def _predict(self):
        api = self._api
        FeedTensor, Transform, Vector3 = api["FeedTensor"], api["Transform"], api["Vector3"]
        ReadTensor, Tensor, Rotation = api["ReadTensor"], api["Tensor"], api["Rotation"]
        RootModule, MotionModule, Sequence = api["RootModule"], api["MotionModule"], api["Sequence"]

        inputs = FeedTensor("X", self.Model.input_dim())
        root = self.Actor.Root
        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        futureRootTransforms = Transform.TransformationTo(self.RootControl.Transforms, root)
        futureRootVelocities = Vector3.DirectionTo(self.RootControl.Velocities, root)
        inputs.FeedVector3(Transform.GetPosition(futureRootTransforms), x=True, y=False, z=True)
        inputs.FeedVector3(Transform.GetAxisZ(futureRootTransforms), x=True, y=False, z=True)
        inputs.FeedVector3(futureRootVelocities, x=True, y=False, z=True)
        inputs.Feed(self.GuidanceControl.Positions)
        outputs = self.Model(inputs.GetTensor().reshape(1, -1), iterations=NETWORK_ITERATIONS)
        outputs = outputs.reshape(SEQUENCE_LENGTH, -1)
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))
        futureRootVectors = outputs.ReadVector3()
        futureRootDelta = Tensor.ZerosLike(futureRootVectors)
        for i in range(1, SEQUENCE_LENGTH):
            futureRootDelta[i] = futureRootDelta[i - 1] + futureRootVectors[i]
        futureRootTransforms = Transform.TransformationFrom(Transform.DeltaXZ(futureRootDelta), root)
        futureRootVelocities = Tensor.ZerosLike(futureRootVectors)
        futureRootVelocities[..., [0, 2]] = futureRootVectors[..., [0, 2]] * SEQUENCE_FPS
        futureRootVelocities = Vector3.DirectionFrom(futureRootVelocities, futureRootTransforms)
        futureMotionTransforms = Transform.TransformationFrom(
            Transform.TR(outputs.ReadVector3(self.Actor.GetBoneCount()), outputs.ReadRotation3D(self.Actor.GetBoneCount())),
            futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )
        futureMotionVelocities = Vector3.DirectionFrom(outputs.ReadVector3(self.Actor.GetBoneCount()), futureRootTransforms.reshape(SEQUENCE_LENGTH, 1, 4, 4))
        self.Previous = self.Sequence
        self.Sequence = Sequence()
        self.Previous = self.Sequence if self.Previous is None else self.Previous
        self.Sequence.Timestamps = Tensor.LinSpace(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.Sequence.Trajectory = RootModule.Series(self.ControlSeries, futureRootTransforms, futureRootVelocities)
        self.Sequence.Motion = MotionModule.Series(self.ControlSeries, self.Actor.GetBoneNames(), futureMotionTransforms, futureMotionVelocities)

        inputs = FeedTensor("X", self.PostProcessor.input_dim())
        currentTransforms = self.Actor.GetTransforms(self.ContactIndices)
        currentVelocities = self.Actor.GetVelocities(self.ContactIndices)
        targetTransforms = self.Sequence.Motion.GetTransforms(self.ContactBones)[1:, :, :]
        targetVelocities = self.Sequence.Motion.GetVelocities(self.ContactBones)[1:, :, :]
        delta_distances = Vector3.Distance(Transform.GetPosition(currentTransforms), Transform.GetPosition(targetTransforms))
        delta_angles = Rotation.Angle(Transform.GetRotation(currentTransforms), Transform.GetRotation(targetTransforms))
        delta_velocities = Vector3.Distance(currentVelocities, targetVelocities)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)
        inputs.Feed(delta_distances)
        inputs.Feed(delta_angles)
        inputs.Feed(delta_velocities)
        contacts = Tensor.ToNumPy(self.PostProcessor(inputs.GetTensor()).reshape(SEQUENCE_LENGTH, len(self.ContactBones)))
        self.Sequence.Contacts = Tensor.Pow(Tensor.Clamp(contacts, 0, 1), CONTACT_POWER)

    def _animate(self):
        api = self._api
        Transform, Vector3, Rotation, Tensor = api["Transform"], api["Vector3"], api["Rotation"], api["Tensor"]
        Definitions, Time = api["Definitions"], api["Time"]

        dt = Time.DeltaTime
        requiredSpeed = (
            Vector3.Distance(self.Actor.GetRootPosition(), self.SimulationObject.GetPosition(0)) + self.SimulationObject.GetLength()
        ) / SEQUENCE_WINDOW
        predictedSpeed = self.Sequence.GetLength() / SEQUENCE_WINDOW
        if requiredSpeed > 0.1 and predictedSpeed > 0.1:
            ts = requiredSpeed / predictedSpeed
            sync = 1.0
        else:
            ts = 1.0
            sync = 0.0
        self.Timescale = Tensor.InterpolateDt(self.Timescale, ts, dt, TIMESCALE_SENSITIVITY)
        self.Timescale = Tensor.Clamp(self.Timescale, MIN_TIMESCALE, MAX_TIMESCALE).item()
        self.Synchronization = Tensor.InterpolateDt(self.Synchronization, sync, dt, SYNCHRONIZATION_SENSITIVITY)
        sdt = dt * self.Timescale
        blend = (Time.TotalTime - self.Timestamp) * PREDICTION_FPS
        root = Transform.Interpolate(self.Previous.SampleRoot(sdt), self.Sequence.SampleRoot(sdt), blend)
        positions = Vector3.Lerp(self.Previous.SamplePositions(sdt), self.Sequence.SamplePositions(sdt), blend)
        rotations = Rotation.Interpolate(self.Previous.SampleRotations(sdt), self.Sequence.SampleRotations(sdt), blend)
        velocities = Vector3.Lerp(self.Previous.SampleVelocities(sdt), self.Sequence.SampleVelocities(sdt), blend)
        contacts = Tensor.Interpolate(self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend)
        self.Actor.Root = Transform.Interpolate(root, self.Actor.Root, self.Sequence.GetRootLock())
        self.Actor.SetTransforms(Transform.TR(Vector3.Lerp(self.Actor.GetPositions() + velocities * sdt, positions, 0.5), rotations))
        self.Actor.SetVelocities(velocities)
        self.Actor.RestoreBoneLengths()
        self.Actor.RestoreBoneAlignments()
        self.LeftLegIK.Solve(
            ankleContact=contacts[0], ballContact=contacts[1], maxIterations=SOLVER_ITERATIONS, maxAccuracy=SOLVER_ACCURACY,
            poleTarget=Vector3.PositionFrom(Vector3.Create(0.0, 0.0, 1.0), self.Actor.GetBone(Definitions.LeftKneeName).GetTransform()),
            poleWeight=1.0,
        )
        self.RightLegIK.Solve(
            ankleContact=contacts[2], ballContact=contacts[3], maxIterations=SOLVER_ITERATIONS, maxAccuracy=SOLVER_ACCURACY,
            poleTarget=Vector3.PositionFrom(Vector3.Create(0.0, 0.0, 1.0), self.Actor.GetBone(Definitions.RightKneeName).GetTransform()),
            poleWeight=1.0,
        )
        self.Actor.SyncToScene()
        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def update(self):
        """1物理ステップ分進める(呼び出し側でAI4Animation.Update()済みであること)"""
        self._control()
        Time = self._api["Time"]
        if self.Timestamp == 0.0 or Time.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS:
            self.Timestamp = Time.TotalTime
            self._predict()
        if self.Sequence is not None:
            self._animate()

    def bone_local_transforms(self) -> dict[str, np.ndarray]:
        """ボーン名 -> 親ボーン相対ローカル変換(4x4、列ベクトル規約)
        Hipsは親なしのためActor.Entity基準のグローバル値そのもの"""
        result = {}
        for name, parent_name in BONE_PARENTS.items():
            bone = self.Actor.GetBone(name)
            global_m = np.array(bone.GetTransform())
            if parent_name is None:
                result[name] = global_m
            else:
                parent_bone = self.Actor.GetBone(parent_name)
                parent_global = np.array(parent_bone.GetTransform())
                result[name] = np.linalg.inv(parent_global) @ global_m
        return result


_GUIDANCE_CACHE: dict = {}


def _load_guidance_templates(api: dict) -> dict:
    if "templates" in _GUIDANCE_CACHE:
        return _GUIDANCE_CACHE["templates"]
    GuidanceModule = api["GuidanceModule"]
    templates = {}
    for path in sorted(os.listdir(GUIDANCES_DIR)):
        with np.load(GUIDANCES_DIR / path, allow_pickle=True) as data:
            gid = Path(path).stem
            templates[gid] = GuidanceModule.Guidance(gid, data["Names"], data["Positions"])
    _GUIDANCE_CACHE["templates"] = templates
    return templates


class LocomotionPool:
    """複数の`LocomotionAgent`をまとめて管理する(モーション生成のみ担当、CrowdControllerとの
    結線・USD配置は`human_manager.py`が担う)"""

    def __init__(self, num_humans: int, world_offsets_xy: Optional[list[tuple[float, float]]] = None):
        self._api = _lazy_import_ai4anim()
        AI4Animation = self._api["AI4Animation"]

        class _EmptyProgram:
            def Start(self):
                pass

        AI4Animation(_EmptyProgram(), mode=AI4Animation.Mode.MANUAL)  # Scene初期化のみ行う

        shared = SharedLocomotionModel.load()
        offsets = world_offsets_xy or [(0.0, 0.0)] * num_humans
        self.agents: list[LocomotionAgent] = [
            LocomotionAgent(shared, self._api, world_offset_xy=offsets[i]) for i in range(num_humans)
        ]

    def set_velocities_world_xy(self, velocities: list[tuple[float, float]]):
        for agent, v in zip(self.agents, velocities):
            agent.set_velocity_world_xy(*v)

    def ground_positions_world_xy(self) -> list[tuple[float, float]]:
        return [a.ground_position_world_xy() for a in self.agents]

    def teleport_agent_world_xy(self, index: int, new_world_xy: tuple[float, float]):
        self.agents[index].teleport_world_xy(new_world_xy)

    def step(self, dt: float):
        AI4Animation = self._api["AI4Animation"]
        AI4Animation.Update(dt)
        for agent in self.agents:
            agent.update()

    def all_bone_local_transforms(self) -> list[dict[str, np.ndarray]]:
        return [a.bone_local_transforms() for a in self.agents]
