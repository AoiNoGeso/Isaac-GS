"""Point Navigation 実機デプロイ用ROS2ノード。
実行例: python3 deploy/deploy.py --model runs/point_nav/sac_final.pt"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from envs.config import JACKAL, EnvConfig
from envs.geometry import goal_vec, quat_to_yaw
from models.sac.policy import SACAgent

# 学習時と推論時で条件がずれないよう、既定値はすべてシミュレータ側の設定から引く
_ENV_DEFAULTS = EnvConfig.model_fields
_IMG_SIZE = _ENV_DEFAULTS["camera_resolution"].default[0]  # 入力画像の一辺 [px]
_GOAL_THRESHOLD = _ENV_DEFAULTS["goal_threshold"].default  # ゴール到達判定の距離 [m]
_V_MAX = JACKAL.v_linear_max  # 最大直進速度 [m/s]
_W_MAX = JACKAL.v_angular_max  # 最大角速度 [rad/s]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="モデルパス (.pt)")
    p.add_argument("--v-max", type=float, default=_V_MAX)
    p.add_argument("--w-max", type=float, default=_W_MAX)
    p.add_argument("--goal-threshold", type=float, default=_GOAL_THRESHOLD)
    p.add_argument("--hz", type=float, default=10.0, help="制御周期 [Hz]")
    p.add_argument(
        "--input-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ゴールベクトルを観測に含める（RGB+Goalモデル用）。--no-input-goal で無効化",
    )
    return p.parse_args()


# ROS2 ノード


class PointNavDeployNode(Node):
    def __init__(self, model: SACAgent, args, input_goal: bool = True):
        super().__init__("point_nav_deploy")
        self._model = model
        self._v_max = args.v_max
        self._w_max = args.w_max
        self._goal_threshold = args.goal_threshold
        self._input_goal = input_goal
        self._lock = threading.Lock()

        # map→base_footprint の自己位置推定をTFから取得する
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._rgb: np.ndarray | None = None  # (3, 84, 84) float32 [0,1]
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._goal: tuple[float, float] | None = None  # mapフレームでの(x, y)

        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._cb_image, 1
        )
        self.create_subscription(PoseStamped, "/goal_pose", self._cb_goal, 1)
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(1.0 / args.hz, self._cb_control)

        self.get_logger().info(f"モデルロード完了: {args.model}")
        self.get_logger().info(
            "'/goal_pose' トピックでゴールを指定してください (RViz2 '2D Goal Pose')"
        )

    # コールバック

    def _cb_image(self, msg: Image):
        """カメラ画像を受信し RGB に変換, リサイズする"""
        try:
            dtype = np.uint8
            raw = np.frombuffer(msg.data, dtype=dtype).reshape(
                msg.height, msg.width, -1
            )
            # encodingに応じてRGB順へ変換
            if msg.encoding in ("rgb8",):
                rgb = raw[..., :3]
            elif msg.encoding in ("bgr8",):
                rgb = raw[..., :3][..., ::-1]
            elif msg.encoding in ("rgba8",):
                rgb = raw[..., :3]
            elif msg.encoding in ("bgra8",):
                rgb = raw[..., 2::-1]
            else:
                rgb = raw[..., :3]
            if rgb.shape[0] != _IMG_SIZE or rgb.shape[1] != _IMG_SIZE:
                from PIL import Image as PILImage

                rgb = np.array(PILImage.fromarray(rgb).resize((_IMG_SIZE, _IMG_SIZE)))
            arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            with self._lock:
                self._rgb = arr
        except Exception as e:
            self.get_logger().warn(f"画像受信エラー: {e}")

    def _update_pose_from_tf(self):
        """TF の map→base_footprint からロボット位置・向きを更新する"""
        try:
            tf = self._tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            with self._lock:
                self._robot_x = t.x
                self._robot_y = t.y
                self._robot_yaw = quat_to_yaw(q.w, q.x, q.y, q.z)
        except Exception:
            pass  # TFが未取得の間は前回値を保持

    def _cb_goal(self, msg: PoseStamped):
        """RViz2 からゴール位置を受信する"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        with self._lock:
            self._goal = (x, y)
        self.get_logger().info(f"ゴール設定: ({x:.2f}, {y:.2f})")

    # 制御ループ

    def _cb_control(self):
        """制御周期ごとに policy 推論を行い cmd_vel を発行する"""
        self._update_pose_from_tf()
        with self._lock:
            rgb = self._rgb
            robot_x, robot_y, robot_yaw = self._robot_x, self._robot_y, self._robot_yaw
            goal = self._goal

        if goal is None or rgb is None:
            self._publish_stop()
            return

        goal_x, goal_y = goal

        dist = float(np.hypot(goal_x - robot_x, goal_y - robot_y))
        if dist < self._goal_threshold:
            self.get_logger().info(f"ゴール到達！ dist={dist:.2f}m")
            self._publish_stop()
            with self._lock:
                self._goal = None
            return

        gvec = goal_vec(robot_x, robot_y, robot_yaw, goal_x, goal_y)

        obs = {"rgb": rgb}
        if self._input_goal:
            obs["goal"] = gvec
        action = self._model.act(obs, deterministic=True)
        v_x_norm = float(np.clip(action[0], -1.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))

        cmd = Twist()
        cmd.linear.x = v_x_norm * self._v_max
        cmd.angular.z = w_norm * self._w_max
        self._pub_cmd.publish(cmd)

    def _publish_stop(self):
        self._pub_cmd.publish(Twist())


# エントリポイント


def main():
    args = _parse_args()

    import torch

    from models.sac.config import ModelConfig, TrainConfig
    from models.sac.network import make_encoder

    input_goal = args.input_goal
    model_cfg = ModelConfig(input_goal=input_goal)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SACAgent(
        encoder_factory=lambda: make_encoder(model_cfg, img_size=_IMG_SIZE),
        action_dim=2,
        cfg=TrainConfig(stage="unused"),  # 実機デプロイではstage(シミュレータ用ステージ名)は使わない
        device=device,
    )
    model.load(args.model)

    rclpy.init()
    node = PointNavDeployNode(model, args, input_goal=input_goal)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
