import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import numpy as np
import roboticstoolbox as rtb
import time

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets

# ===== 좌표계 회전 정의: 로봇 왼팔 베이스 → 월드 =====
sqrt2 = np.sqrt(2) / 2

# p_W = R_L * p_L   (p_L: 왼팔 베이스 좌표, p_W: 월드 좌표)
R_L = np.array([
    [0.0,      1.0,      0.0    ],   # x_W
    [sqrt2,    0.0,      sqrt2  ],   # y_W
    [sqrt2,    0.0,     -sqrt2  ],   # z_W
])

def rotate_left_to_world(p_L):
    """p_L: (3,) numpy → 월드 프레임으로 회전만 적용."""
    p_L = np.asarray(p_L).reshape(3,)
    return R_L @ p_L


class LivePlot3D(Node):
    def __init__(self):
        super().__init__('live_plot_3d')

        # URDF 로봇 모델
        urdf_path = "/home/vision/dualarm_ws/src/doosan-robot2/dsr_description2/urdf/m0609.white.urdf"
        self.robot = rtb.ERobot.URDF(urdf_path)
        self.last_fb_time = time.time()

        # ROS 구독
        self.sub_cmd = self.create_subscription(
            PoseStamped, '/TCP_target_pose_L', self.callback_command, 10
        )
        self.joint_name = [f"left_joint_{i}" for i in range(1,7)] + \
                          [f"right_joint_{i}" for i in range(1,7)]
        self.joint_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # 데이터 저장 (월드 프레임 기준)
        self.cmd_points = []   # [ [x,y,z], ... ]
        self.fb_points = []    # [t, x, y, z]

        # 속도/가속도 시작 트리거
        self.start_time = None
        self.speed_threshold = 0.02  # [m/s]

        # Qt App
        self.app = QtWidgets.QApplication([])

        # 3D Traj 창
        self.w = gl.GLViewWidget()
        self.w.show()
        self.w.setWindowTitle('Live 3D Pose Trajectory (World Frame)')
        self.w.setCameraPosition(distance=0.3)

        g = gl.GLGridItem()
        g.scale(0.1, 0.1, 0.1)
        self.w.addItem(g)

        self.cmd_line = gl.GLLinePlotItem(pos=np.array([[0,0,0]]),
                                          color=(1,0,0,0.5), width=1,
                                          antialias=True, mode='line_strip')
        self.cmd_scatter = gl.GLScatterPlotItem()
        self.w.addItem(self.cmd_line)
        self.w.addItem(self.cmd_scatter)

        self.fb_line = gl.GLLinePlotItem(pos=np.array([[0,0,0]]),
                                         color=(0.5,0,1,0.5), width=1,
                                         antialias=True, mode='line_strip')
        self.fb_scatter = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]),
                                               color=(0,0,1,1), size=4)
        self.w.addItem(self.fb_line)
        self.w.addItem(self.fb_scatter)

        # 2D Plot 창 (속력/가속도)
        self.plot_win = pg.GraphicsLayoutWidget(show=True, title="End-Effector Speed/Acceleration (World Frame)")

        # 속력 그래프
        self.ax_vel = self.plot_win.addPlot(title="Speed |v|")
        self.curve_vel = self.ax_vel.plot(pen='r')
        self.ax_vel.setLabel('left', "m/s")
        self.ax_vel.showGrid(x=True, y=True)
        self.ax_vel.setXRange(0, 10)
        self.ax_vel.setYRange(0, 1)

        # 가속도 그래프
        self.plot_win.nextRow()
        self.ax_acc = self.plot_win.addPlot(title="Acceleration |a|")
        self.curve_acc = self.ax_acc.plot(pen='b')
        self.ax_acc.setLabel('left', "m/s²")
        self.ax_acc.setLabel('bottom', "Time [s]")
        self.ax_acc.showGrid(x=True, y=True)
        self.ax_acc.setXRange(0, 10)
        self.ax_acc.setYRange(0, 15)

    # -------- ROS 콜백 --------
    def callback_command(self, msg: PoseStamped):
        pos = msg.pose.position
        p_L = np.array([pos.x, pos.y, pos.z], dtype=float)  # 왼팔 베이스 기준
        p_W = rotate_left_to_world(p_L)                     # 월드 기준

        self.cmd_points.append(p_W.tolist())
        pts = np.array(self.cmd_points)
        self.cmd_line.setData(pos=pts, mode='line_strip')

        # 점 크기 조절: 첫 번째 + 이후 6번째마다 크게
        sizes = np.array([4]*len(pts), dtype=float)  # 기본 크기
        if len(pts) > 0:
            sizes[0] = 10  # 첫 번째 점 크게
        for i in range(6, len(pts), 6):
            sizes[i] = 10

        self.cmd_scatter.setData(pos=pts, color=(1,0,0,1), size=sizes)
        self.update_view()

    def joint_callback(self, msg: JointState):
        joint_mapping = {n: p for n, p in zip(msg.name, msg.position)}
        joint_position = [joint_mapping.get(j) for j in self.joint_name]
        if None in joint_position:
            return

        qL = joint_position[:6]
        T = self.robot.fkine(qL)
        pos_L = np.array(T.t).flatten()   # (3,)
        pos_W = rotate_left_to_world(pos_L)

        now = time.time()
        if now - self.last_fb_time < 0.04:  # 25Hz 업데이트 제한
            return
        self.last_fb_time = now

        # fb_points: [t, x, y, z] (월드 좌표)
        self.fb_points.append([now, pos_W[0], pos_W[1], pos_W[2]])
        pts = np.array([p[1:] for p in self.fb_points])
        self.fb_line.setData(pos=pts, mode='line_strip')
        self.fb_scatter.setData(pos=pts)

        # 속력/가속도 업데이트 (월드 프레임 기준)
        derivatives = self.compute_derivatives()
        if derivatives is not None:
            t, speed, accel = derivatives

            # 시작 트리거: speed > threshold 되는 첫 시점
            if self.start_time is None:
                above_thresh = np.where(speed > self.speed_threshold)[0]
                if len(above_thresh) > 0:
                    idx_start = above_thresh[0]
                    self.start_time = t[idx_start]

            if self.start_time is not None:
                t_rel = t - self.start_time

                # 최근 0~10초 윈도우만 표시
                mask = (t_rel >= 0) & (t_rel <= 10)
                self.curve_vel.setData(t_rel[mask], speed[mask])

                if len(accel) > 0:
                    t_rel_acc = t_rel[1:]
                    mask_acc = (t_rel_acc >= 0) & (t_rel_acc <= 10)
                    self.curve_acc.setData(t_rel[1:][mask_acc], accel[mask_acc])

                # X축 고정
                self.ax_vel.setXRange(0, 10)
                self.ax_acc.setXRange(0, 10)

        self.update_view()

    # -------- 수학 처리 --------
    def compute_derivatives(self):
        if len(self.fb_points) < 3:
            return None
        fb = np.array(self.fb_points)  # [t, x, y, z]
        t = fb[:, 0]
        pos = fb[:, 1:4]

        dt = np.diff(t)
        if np.any(dt <= 0):
            return None

        vel = np.diff(pos, axis=0) / dt[:, None]     # (N-1, 3)
        speed = np.linalg.norm(vel, axis=1)          # (N-1,)
        acc = np.diff(vel, axis=0) / dt[1:, None]    # (N-2, 3)
        accel = np.linalg.norm(acc, axis=1)          # (N-2,)

        return t[1:], speed, accel

    def update_view(self):
        all_pts = []
        if self.cmd_points:
            all_pts.extend(self.cmd_points)
        if self.fb_points:
            all_pts.extend([p[1:] for p in self.fb_points])
        if not all_pts:
            return
        pts = np.array(all_pts)
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)
        center = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2]
        max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
        self.w.opts['center'] = pg.Vector(center[0], center[1], center[2])
        self.w.opts['distance'] = max_range * 2 if max_range > 0 else 1

    # -------- 루프 --------
    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            self.app.processEvents()


def main():
    rclpy.init()
    node = LivePlot3D()
    try:
        node.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
