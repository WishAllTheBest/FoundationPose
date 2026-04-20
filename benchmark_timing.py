import os
import time
import csv
import numpy as np
import torch
import trimesh
from estimater import *
from datareader import *
from pose_filter import SE3KalmanFilter
from icp_refiner import FacialICPRefiner

def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    mesh_file = f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj'
    test_scene_dir = f'{code_dir}/FoundationPose_manual/kinect_data_with_calibrated_K'
    csv_file = f'{code_dir}/timing_results.csv'

    set_logging_format()
    set_seed(0)

    print("正在加载网络模型与构建环境 (请稍候)...")
    mesh = trimesh.load(mesh_file)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug=0, glctx=glctx)

    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
    icp_refiner = FacialICPRefiner(mesh=mesh, distance_threshold=0.02)
    pose_filter = SE3KalmanFilter(process_noise=1e-4, measurement_noise=1e-2, rot_alpha=0.7)

    # 预热阶段：使用第0帧执行 register 初始化全局位姿，以及模型预热
    print("正在执行第一帧注册估算与CUDA预热...")
    color0 = reader.get_color(0)
    depth0 = reader.get_depth(0)
    mask0 = reader.get_mask(0).astype(bool)
    est.register(K=reader.K, rgb=color0, depth=depth0, ob_mask=mask0, iteration=5)

    print(f"开始对第1张图像执行连续 10 次的估算时间统计...")
    # 选取第1帧图像作为测试数据
    color = reader.get_color(1)
    depth = reader.get_depth(1)

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Pose_Estimation(ms)", "ICP_Refinement(ms)", "EKF_Tracking(ms)", "Total(ms)"])

        for run in range(1, 11):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # --- 第一阶段：深度学习位姿估计 ---
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # --- 第二阶段：骨性特征加权局部修正 (ICP) ---
            pose_icp = icp_refiner.refine(pose, depth, reader.K)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # --- 第三阶段：多流形 EKF 姿态抗噪跟踪 ---
            pose_ekf = pose_filter.update(pose_icp, dt=1.0)
            t3 = time.perf_counter()

            t_est = (t1 - t0) * 1000
            t_icp = (t2 - t1) * 1000
            t_ekf = (t3 - t2) * 1000
            t_total = (t3 - t0) * 1000

            writer.writerow([run, f"{t_est:.2f}", f"{t_icp:.2f}", f"{t_ekf:.2f}", f"{t_total:.2f}"])
            print(f"-> 测试 {run:02d}/10 | 网络估计: {t_est:6.2f}ms | ICP修正: {t_icp:6.2f}ms | EKF滤波: {t_ekf:6.2f}ms | 流程总计: {t_total:6.2f}ms")

    print(f"\n全部测试完毕，统计结果已完整保存至: {csv_file}")

if __name__ == "__main__":
    main()
