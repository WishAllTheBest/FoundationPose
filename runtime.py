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
import argparse

def main():
    parser = argparse.ArgumentParser(description="分步骤验证 FoundationPose 追踪的时间消耗")
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose_manual/kinect_data_with_calibrated_K')
    parser.add_argument('--csv_file', type=str, default=f'{code_dir}/timing_benchmark_results.csv')
    parser.add_argument('--test_runs', type=int, default=10, help="要循环测试的次数")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    print("==================================================")
    print("1. 正在加载网络模型与构建环境 (请稍候)...")
    print("==================================================")
    mesh = trimesh.load(args.mesh_file)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    # 禁用 debug 以排除 I/O 和画图的耗时干扰
    est = FoundationPose(
        model_pts=mesh.vertices, 
        model_normals=mesh.vertex_normals, 
        mesh=mesh, 
        scorer=scorer, 
        refiner=refiner, 
        debug=0, 
        glctx=glctx
    )

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    icp_refiner = FacialICPRefiner(mesh=mesh, distance_threshold=0.02)
    pose_filter = SE3KalmanFilter(process_noise=1e-4, measurement_noise=1e-2, rot_alpha=0.7)

    # 预热阶段：使用第0帧执行 register 初始化全局位姿，以及模型预热
    print("\n==================================================")
    print("2. 正在执行第一帧全局初始注册与 PyTorch CUDA 预热...")
    print("==================================================")
    color0 = reader.get_color(0)
    depth0 = reader.get_depth(0)
    mask0 = reader.get_mask(0).astype(bool)
    init_pose = est.register(K=reader.K, rgb=color0, depth=depth0, ob_mask=mask0, iteration=5)

    print(f"\n==================================================")
    print(f"3. 开始对第 1 张图像执行连续 {args.test_runs} 次的单帧跟踪时间独立统计...")
    print("==================================================")
    # 选取第1帧图像作为测试连续跟踪的数据靶标
    color = reader.get_color(1)
    depth = reader.get_depth(1)

    with open(args.csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Run_Index", "Pose_Estimation(ms)", "ICP_Refinement(ms)", "EKF_Tracking(ms)", "Total_Time(ms)"])

        for run in range(1, args.test_runs + 1):
            # 将初始化位姿作为输入强行重置跟踪管线（仅供时间测试）
            est.pose = init_pose.copy()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # --- 第一阶段：深度学习位姿连续跟踪估计 (Tracking) ---
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # --- 第二阶段：骨性特征加权局部修正 (ICP Refinement) ---
            pose_icp = icp_refiner.refine(pose, depth, reader.K)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # --- 第三阶段：多流形 EKF 姿态抗噪跟踪 (EKF Filter) ---
            pose_ekf = pose_filter.update(pose_icp, dt=1.0)
            t3 = time.perf_counter()

            # 计算秒耗时
            t_est = (t1 - t0) 
            t_icp = (t2 - t1) 
            t_ekf = (t3 - t2) 
            t_total = (t3 - t0) 

            # 写入 CSV 并打印
            writer.writerow([run, f"{t_est:.4f}", f"{t_icp:.4f}", f"{t_ekf:.4f}", f"{t_total:.4f}"])
            print(f"-> 测试回次 {run:02d}/{args.test_runs} | "
                  f"网络估计: {t_est:8.4f}ms | "
                  f"ICP修正: {t_icp:8.4f}ms | "
                  f"EKF滤波: {t_ekf:8.4f}ms | "
                  f"单帧总计: {t_total:8.4f}ms")

    print(f"\n全部测试完毕，统计指标已完整保存至: {args.csv_file}")

if __name__ == "__main__":
    main()