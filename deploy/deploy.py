"""
Point Navigation デプロイスクリプト（ROS2 policy ノード）

sim_ros2_bridge.py（または実機ドライバ）と組み合わせて使用する。
RViz2 で '2D Goal Pose' を指定することでゴールを設定できる。

購読トピック:
  /camera/color/image_raw   sensor_msgs/Image
  /odom                     nav_msgs/Odometry
  /goal_pose                geometry_msgs/PoseStamped

発行トピック:
  /cmd_vel                  geometry_msgs/Twist

実行方法:
  # 別ターミナルで sim_ros2_bridge.py を起動してから:
  python3 deploy/deploy.py --model runs/point_nav/sac_final
  python3 deploy/deploy.py --model runs/point_nav/checkpoints/sac_10000_steps
"""

import argparse
import math
import threading

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from stable_baselines3 import SAC

# -------------------------------------------------------------------
# 定数（シミュレータの isaac_env.py と合わせること）
# -------------------------------------------------------------------

_IMG_SIZE = 84
_DIST_NORM_SCALE = 10.0    # d_norm = dist / 10.0
_V_MAX = 0.3               # [m/s]  実機に合わせて調整
_W_MAX = 1.0               # [rad/s] 実機に合わせて調整
_GOAL_THRESHOLD = 0.4      # [m]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="SB3 モデルパス (.zip 拡張子不要)")
    p.add_argument("--v-max", type=float, default=_V_MAX)
    p.add_argument("--w-max", type=float, default=_W_MAX)
    p.add_argument("--goal-threshold", type=float, default=_GOAL_THRESHOLD)
    p.add_argument("--hz", type=float, default=10.0, help="制御周期 [Hz]")
    return p.parse_args()


# -------------------------------------------------------------------
# ゴールベクトル計算（isaac_env.py の _compute_goal_vec と同一ロジック）
# -------------------------------------------------------------------

def _compute_goal_vec(
    robot_x: float, robot_y: float, robot_yaw: float,
    goal_x: float, goal_y: float,
) -> np.ndarray:
    """
    ロボット位置・向き・ゴール位置から policy への入力ベクトルを計算する。

    シミュレータ（Z-up, ロボット前方=-Y）と同じ計算式を使用:
      angle_rel = (arctan2(dx, -dy) - yaw + π) % (2π) - π

    Returns:
        np.ndarray shape (2,): [d_norm, angle_norm]
          d_norm    = clip(dist / 10.0, 0, 1)
          angle_norm = angle_rel / π  ∈ [-1, 1]
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.sqrt(dx ** 2 + dy ** 2)
    d_norm = float(np.clip(dist / _DIST_NORM_SCALE, 0.0, 1.0))
    angle_rel = (math.atan2(dx, -dy) - robot_yaw + math.pi) % (2 * math.pi) - math.pi
    return np.array([d_norm, angle_rel / math.pi], dtype=np.float32)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# -------------------------------------------------------------------
# ROS2 ノード
# -------------------------------------------------------------------

class PointNavDeployNode(Node):
    def __init__(self, model: SAC, args):
        super().__init__("point_nav_deploy")
        self._model = model
        self._v_max = args.v_max
        self._w_max = args.w_max
        self._goal_threshold = args.goal_threshold
        self._lock = threading.Lock()

        # 状態変数
        self._rgb: np.ndarray | None = None           # (3, 84, 84) float32 [0,1]
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._goal: tuple[float, float] | None = None # (x, y) in odom frame

        # サブスクライバ
        self.create_subscription(Image,       "/camera/color/image_raw", self._cb_image, 1)
        self.create_subscription(Odometry,    "/odom",                   self._cb_odom,  1)
        self.create_subscription(PoseStamped, "/goal_pose",              self._cb_goal,  1)

        # パブリッシャ
        self._pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)

        # 制御タイマー
        self.create_timer(1.0 / args.hz, self._cb_control)

        self.get_logger().info(f"モデルロード完了: {args.model}")
        self.get_logger().info("'/goal_pose' トピックでゴールを指定してください (RViz2 '2D Goal Pose')")

    # ── コールバック ────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        """カメラ画像を受信して前処理する"""
        try:
            # sensor_msgs/Image → numpy (H, W, C)
            dtype = np.uint8
            raw = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
            # encoding に応じて RGB に変換
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
            # 84×84 にリサイズ（NumPy のみ、バイリニア近似）
            if rgb.shape[0] != _IMG_SIZE or rgb.shape[1] != _IMG_SIZE:
                from PIL import Image as PILImage
                rgb = np.array(PILImage.fromarray(rgb).resize((_IMG_SIZE, _IMG_SIZE)))
            arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3,84,84)
            with self._lock:
                self._rgb = arr
        except Exception as e:
            self.get_logger().warn(f"画像受信エラー: {e}")

    def _cb_odom(self, msg: Odometry):
        """オドメトリからロボット位置・向きを取得する"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self._robot_x = p.x
            self._robot_y = p.y
            self._robot_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _cb_goal(self, msg: PoseStamped):
        """RViz2 からゴール位置を受信する"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        with self._lock:
            self._goal = (x, y)
        self.get_logger().info(f"ゴール設定: ({x:.2f}, {y:.2f})")

    # ── 制御ループ ─────────────────────────────────────────────────

    def _cb_control(self):
        """制御周期ごとに policy 推論を行い cmd_vel を発行する"""
        with self._lock:
            rgb = self._rgb
            robot_x, robot_y, robot_yaw = self._robot_x, self._robot_y, self._robot_yaw
            goal = self._goal

        # ゴール未設定 or 画像未受信 → 停止
        if goal is None or rgb is None:
            self._publish_stop()
            return

        goal_x, goal_y = goal

        # ゴール到達判定
        dist = math.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
        if dist < self._goal_threshold:
            self.get_logger().info(f"ゴール到達！ dist={dist:.2f}m")
            self._publish_stop()
            with self._lock:
                self._goal = None
            return

        # ゴールベクトル計算
        goal_vec = _compute_goal_vec(robot_x, robot_y, robot_yaw, goal_x, goal_y)

        # policy 推論
        obs = {"rgb": rgb[np.newaxis], "goal": goal_vec[np.newaxis]}
        action, _ = self._model.predict(obs, deterministic=True)
        v_x_norm = float(np.clip(action[0][0], -1.0, 1.0))
        w_norm   = float(np.clip(action[0][1], -1.0, 1.0))

        # スケール変換 → cmd_vel 発行
        cmd = Twist()
        cmd.linear.x  = v_x_norm * self._v_max
        cmd.angular.z = w_norm   * self._w_max
        self._pub_cmd.publish(cmd)

    def _publish_stop(self):
        self._pub_cmd.publish(Twist())


# -------------------------------------------------------------------
# エントリポイント
# -------------------------------------------------------------------

def main():
    args = _parse_args()
    model = SAC.load(args.model)

    rclpy.init()
    node = PointNavDeployNode(model, args)
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
