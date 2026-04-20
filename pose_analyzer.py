import numpy as np
import trimesh
import cv2
import os
import glob
import argparse
import imageio
from Utils import *
from tools.evaluate_registration import RegistrationEvaluator

def main():
    parser = argparse.ArgumentParser(description="位姿估计结果可视化与量化评估")
    code_dir = os.path.dirname(os.path.realpath(__file__))
    file_dir = 'kinect_data_with_calibrated_K'
    # 路径配置
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj', help='Mesh .obj 文件路径')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/FoundationPose_manual/'+file_dir, help='原始数据目录 (含rgb, depth, cam_K.txt)')
    parser.add_argument('--est_pose_dir', type=str, default=f'{code_dir}/debug_'+file_dir+'_mesh/ob_in_cam', help='估计出的位姿目录 (ob_in_cam)')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/FoundationPose_manual/'+file_dir+'/analysis_results', help='结果保存目录')
    parser.add_argument('--threshold', type=float, default=0.007, help='Fitness 评估阈值 (米)')
    args = parser.parse_args()

    # 1. 初始化环境
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)
    
    mesh = trimesh.load(args.mesh_file)
    K = np.loadtxt(os.path.join(args.test_scene_dir, 'cam_K.txt')).reshape(3,3)
    evaluator = RegistrationEvaluator(mesh, K)
    
    # 获取 Mesh 包围盒用于绘制 BBox
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    # 2. 检索文件
    rgb_files = sorted(glob.glob(os.path.join(args.test_scene_dir, "rgb", "*.png")))
    est_pose_files = sorted(glob.glob(os.path.join(args.est_pose_dir, "*.txt")))
    depth_files = sorted(glob.glob(os.path.join(args.test_scene_dir, "depth", "*.png")))
    mask_files = sorted(glob.glob(os.path.join(args.test_scene_dir, "masks_generated", "*.png"))) # 如果有生成好的mask更好
    
    print(f"找到 {len(est_pose_files)} 帧位姿估计结果")

    all_fitness = []
    all_rmse = []

    # 3. 循环处理每一帧
    for i, pose_file in enumerate(est_pose_files):
        id_str = os.path.basename(pose_file).replace('.txt', '')
        
        # 加载位姿 (Est Pose是将Mesh原始顶点变换到相机系的矩阵)
        pose = np.loadtxt(pose_file).reshape(4,4)
        
        # 加载图像
        rgb = cv2.imread(os.path.join(args.test_scene_dir, "rgb", f"{id_str}.png"))
        depth = cv2.imread(os.path.join(args.test_scene_dir, "depth", f"{id_str}.png"), -1) / 1000.0
        
        mask = None
        mask_path = os.path.join(args.test_scene_dir, "masks_generated", f"{id_str}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)

        # A. 计算量化指标 (无监督)
        fitness, rmse = evaluator.depth_fitness_score(pose, depth, mask_img=mask, threshold=args.threshold)
        all_fitness.append(fitness)
        all_rmse.append(rmse)

        # B. 绘制可视化图
        # 绘制 BBox (需要中心化位姿)
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = rgb.copy()
        vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
        
        # 投影 Mesh 边缘
        m_viz = mesh.copy()
        m_viz.apply_transform(pose)
        verts = m_viz.vertices
        proj = (K @ verts.T).T
        proj2 = (proj[:, :2] / (proj[:, 2:3] + 1e-8)).astype(int)
        
        # 绘制网格线
        for face in m_viz.faces:
            pts = proj2[face]
            cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (0,170,255), 1, lineType=cv2.LINE_AA)
            cv2.line(vis, tuple(pts[1]), tuple(pts[2]), (0,170,255), 1, lineType=cv2.LINE_AA)
            cv2.line(vis, tuple(pts[2]), tuple(pts[0]), (0,170,255), 1, lineType=cv2.LINE_AA)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=4, transparency=0, is_input_rgb=True)

        # 在图上标注指标
        cv2.putText(vis, f"Fitness: {fitness:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis, f"RMSE: {rmse:.2f}mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 保存图片
        output_path = os.path.join(args.output_dir, "vis", f"{id_str}_analysis.png")
        cv2.imwrite(output_path, vis)
        
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1} 帧...")

    # 4. 输出最终统计
    print("\n" + "="*50)
    print("      QUANTITATIVE EVALUATION SUMMARY")
    print("="*50)
    print(f"Mean Fitness (@{args.threshold*1000}mm): {np.mean(all_fitness):.4f}")
    print(f"Median Fitness:           {np.median(all_fitness):.4f}")
    print(f"Mean RMSE:                {np.mean(all_rmse):.4f} mm")
    print(f"Median RMSE:              {np.median(all_rmse):.4f} mm")
    print("="*50)
    print(f"可视化结果已保存至: {os.path.join(args.output_dir, 'vis')}")

if __name__ == "__main__":
    main()