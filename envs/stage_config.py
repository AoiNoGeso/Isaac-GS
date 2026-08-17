from pydantic import BaseModel


class StagePreset(BaseModel):
    stage_path: str
    fixed_spawn_pos: tuple[float, float, float] | None
    fixed_goal_pos: tuple[float, float, float] | None
    fixed_spawn_yaw_deg: float | None
    overhead_camera_translation: tuple[float, float, float] | None = None
    overhead_camera_orientation: tuple[float, float, float, float] | None = None  # (w, x, y, z)順


STAGE_PRESETS: dict[str, StagePreset] = {
    "room1": StagePreset(
        stage_path="assets/stages/room1/stage.usda",
        fixed_spawn_pos=(0.9, -0.19, -2.6),
        fixed_goal_pos=(-3.0, 1.6, -2.6),
        fixed_spawn_yaw_deg=137,
    ),
    "corridor1": StagePreset(
        stage_path="assets/stages/corridor1/stage.usda",
        fixed_spawn_pos=(0.4, 1.4, -1.0),
        fixed_goal_pos=(-0.1, -1.3, -0.8),
        fixed_spawn_yaw_deg=-90,
    ),
    "corridor2": StagePreset(
        stage_path="assets/stages/corridor2/stage.usda",
        fixed_spawn_pos=(0.0, -2.2, -1.5),
        fixed_goal_pos=(0.21, 6.9, -1.0),
        fixed_spawn_yaw_deg=90,
        overhead_camera_translation=(-0.25, -14.3, -0.46),
        overhead_camera_orientation=(0.7071, 0.0, 0.0, 0.7071),
    ),
}


def get_preset(name: str) -> StagePreset:
    """名前からStagePresetを取得する"""
    try:
        return STAGE_PRESETS[name]
    except KeyError:
        raise KeyError(
            f"未知のステージ名です: {name!r}. 利用可能なステージ: {stage_names()}"
        ) from None


def stage_names() -> list[str]:
    """利用可能な全ステージ名の一覧"""
    return list(STAGE_PRESETS.keys())
