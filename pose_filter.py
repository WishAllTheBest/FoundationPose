import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class SE3KalmanFilter:
    """
    面向李群流形 SE(3) 设计的扩展卡尔曼滤波器（简化实现）
    本滤波器作用于跟踪后期的 6DoF 姿态输出，
    剔除由于深度噪点引起的空间极微量高频平移抖动与旋转刺闪（Jittering）。
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-2, rot_alpha=0.7):
        self.state_t = np.zeros(3)
        self.state_q = np.array([0, 0, 0, 1.0]) # qx, qy, qz, qw
        self.state_v = np.zeros(3) # 平移速度
        
        # 平移的协方差阵与噪声参数
        self.P_t = np.eye(3) * 1.0
        self.P_v = np.eye(3) * 1.0
        self.Q_t = np.eye(3) * process_noise
        self.Q_v = np.eye(3) * process_noise * 0.1
        self.R_t = np.eye(3) * measurement_noise
        
        # 旋转流形更新的学习率（近似马尔可夫协方差权重），越小越平滑
        self.rot_alpha = rot_alpha
        self.is_initialized = False

    def update(self, pose_matrix, dt=1.0):
        # 确保传入的矩阵是可读写的，防止只读视图报错
        pose_matrix = np.array(pose_matrix, copy=True)
        t_meas = pose_matrix[:3, 3]
        r_meas = R.from_matrix(pose_matrix[:3, :3])
        q_meas = r_meas.as_quat() # x, y, z, w
        
        if not self.is_initialized:
            self.state_t = t_meas
            self.state_q = q_meas
            self.is_initialized = True
            return pose_matrix.copy()
            
        # ---------- 1. 平移部分 EKF (常速运动模型) ----------
        # 状态预测
        t_pred = self.state_t + self.state_v * dt
        P_t_pred = self.P_t + self.P_v * (dt**2) + self.Q_t
        
        # 计算卡尔曼增益 K
        S_t = P_t_pred + self.R_t
        K_t = P_t_pred @ np.linalg.inv(S_t)
        
        # 状态更新
        self.state_t = t_pred + K_t @ (t_meas - t_pred)
        self.P_t = (np.eye(3) - K_t) @ P_t_pred
        
        # 速度更新
        self.state_v = (self.state_t - t_pred) / dt
        self.P_v = self.P_v + self.Q_v # 简化的速度协方差漂移
        
        # ---------- 2. 旋转部分 SO(3) 流形状态更新 ----------
        # 约束四元数在同一半球区，避免大角度跳转
        if np.dot(self.state_q, q_meas) < 0:
            q_meas = -q_meas
            
        # 依据球型线性插值(SLERP)模拟 SO(3) 上的李代数切空间更新
        try:
            times = [0, 1]
            rotations = R.from_quat([self.state_q, q_meas])
            slerp = Slerp(times, rotations)
            # rot_alpha 作为观测可信度的马尔可夫近似衰减率
            q_new = slerp([self.rot_alpha])[0].as_quat()
        except:
            # 奇异点保底降级为正则化线性插值(NLERP)
            q_new = (1 - self.rot_alpha) * self.state_q + self.rot_alpha * q_meas
            q_new /= np.linalg.norm(q_new)
            
        self.state_q = q_new
        
        # ---------- 3. 回填平滑后的组合 SE(3) 矩阵 ----------
        filtered_pose = np.eye(4)
        filtered_pose[:3, :3] = R.from_quat(self.state_q).as_matrix()
        filtered_pose[:3, 3] = self.state_t
        
        return filtered_pose
