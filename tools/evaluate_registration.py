'''
Docstring for evaluate_registration
liu yaning
2026.02.02
用于评估mesh配准精度的脚本，计算ADD、ADD-S、Chamfer距离、2D投影误差等指标。
'''
import numpy as np
import trimesh
import os
import glob
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import json

class RegistrationEvaluator:
    """用于评估mesh配准结果的类"""
    
    def __init__(self, mesh, K):
        """
        初始化评估器
        Args:
            mesh: trimesh.Trimesh 对象
            K: 相机内参 (3,3)
        """
        self.mesh = mesh
        self.K = K
        self.model_points = mesh.vertices
    
    def rotation_error(self, R_est, R_gt):
        """
        计算旋转误差（角度）
        Args:
            R_est: 估计的旋转矩阵 (3,3)
            R_gt: GT旋转矩阵 (3,3)
        Returns:
            旋转误差（度）
        """
        # 计算两个旋转矩阵的相对旋转
        R_rel = R_est @ R_gt.T
        # 使用迹和行列式计算旋转角
        trace = np.trace(R_rel)
        trace = np.clip(trace, -1, 3)
        rotation_angle_rad = np.arccos((trace - 1) / 2)
        rotation_angle_deg = np.degrees(rotation_angle_rad)
        return rotation_angle_deg
    
    def translation_error(self, t_est, t_gt):
        """
        计算平移误差（毫米）
        Args:
            t_est: 估计的平移向量 (3,)
            t_gt: GT平移向量 (3,)
        Returns:
            平移误差（毫米，假设单位为米则乘1000）
        """
        trans_error = np.linalg.norm(t_est - t_gt)
        return trans_error * 1000  # 转换为毫米
    
    def add_metric(self, pose_est, pose_gt):
        """
        计算ADD (Average Distance of Model Points)
        Args:
            pose_est: 估计的位姿矩阵 (4,4)
            pose_gt: GT位姿矩阵 (4,4)
        Returns:
            ADD值（毫米）
        """
        # 获取旋转和平移
        R_est = pose_est[:3, :3]
        t_est = pose_est[:3, 3]
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        
        # 变换模型点
        points_est = (R_est @ self.model_points.T).T + t_est
        points_gt = (R_gt @ self.model_points.T).T + t_gt
        
        # 计算每个点的距离
        distances = np.linalg.norm(points_est - points_gt, axis=1)
        add_value = np.mean(distances)
        
        return add_value * 1000  # 转换为毫米
    
    def add_s_metric(self, pose_est, pose_gt):
        """
        计算ADD-S (Symmetry-aware Average Distance)
        使用最近点对应来处理对称物体
        Args:
            pose_est: 估计的位姿矩阵 (4,4)
            pose_gt: GT位姿矩阵 (4,4)
        Returns:
            ADD-S值（毫米）
        """
        R_est = pose_est[:3, :3]
        t_est = pose_est[:3, 3]
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        
        # 变换模型点
        points_est = (R_est @ self.model_points.T).T + t_est
        points_gt = (R_gt @ self.model_points.T).T + t_gt
        
        # 计算每个估计点到GT点云的最近距离
        distances = cdist(points_est, points_gt)
        min_distances = np.min(distances, axis=1)
        add_s_value = np.mean(min_distances)
        
        return add_s_value * 1000  # 转换为毫米
    
    def chamfer_distance(self, pose_est, pose_gt):
        """
        计算Chamfer距离（双向最近点距离）
        Args:
            pose_est: 估计的位姿矩阵 (4,4)
            pose_gt: GT位姿矩阵 (4,4)
        Returns:
            Chamfer距离（毫米）
        """
        R_est = pose_est[:3, :3]
        t_est = pose_est[:3, 3]
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        
        # 变换模型点
        points_est = (R_est @ self.model_points.T).T + t_est
        points_gt = (R_gt @ self.model_points.T).T + t_gt
        
        # 估计到GT的最小距离
        dist_est_to_gt = np.min(cdist(points_est, points_gt), axis=1)
        # GT到估计的最小距离
        dist_gt_to_est = np.min(cdist(points_gt, points_est), axis=1)
        
        # Chamfer距离是两个方向的平均
        chamfer = (np.mean(dist_est_to_gt) + np.mean(dist_gt_to_est)) / 2
        
        return chamfer * 1000  # 转换为毫米

    def depth_fitness_score(self, pose_est, depth_img, mask_img=None, threshold=0.005):
        """
        [无监督指标] 计算模型与实际深度图的拟合程度。不需要GT。
        Args:
            pose_est: 估计的位姿 (4,4)
            depth_img: 实际拍摄的深度图 (H,W)，单位米
            mask_img: 目标的掩码/分割图（可选，用于排除背景点）
            threshold: 判定对齐成功的阈值（默认2cm）
        Returns:
            fitness: 对齐点比例 (0-1)
            rmse: 均方根误差 (mm)
        """
        h, w = depth_img.shape
        # 将深度图转为点云
        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        valid = depth_img > 0.1
        if mask_img is not None:
            valid = valid & (mask_img > 0)
        
        z = depth_img[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        scene_points = np.stack([x, y, z], axis=1)
        
        if len(scene_points) == 0:
            return 0.0, 0.0

        # 变换模型点到估计位姿
        model_transformed = (pose_est[:3, :3] @ self.model_points.T).T + pose_est[:3, 3]

        # m_debug = trimesh.Trimesh(vertices=model_transformed, faces=self.mesh.faces)
        # m_debug.export('./tools/debug_eval_model.obj')
        # scene_pcd = trimesh.points.PointCloud(scene_points)
        # scene_pcd.export('./tools/debug_eval_scene.ply')
        # 计算场景点到模型表面的最小距离
        from scipy.spatial import cKDTree
        tree = cKDTree(model_transformed)
        dists, _ = tree.query(scene_points)
        
        # Fitness: 距离小于阈值的点占比
        inliers = dists < threshold
        fitness = np.mean(inliers)
        # RMSE: 仅计算内点 (inliers) 的均方根误差
        if np.sum(inliers) > 0:
            rmse = np.sqrt(np.mean(dists[inliers]**2)) * 1000 # 换算成mm
        else:
            rmse = 0.0
        
        return fitness, rmse

    def evaluate_all(self, pose_est, pose_gt, rgb_img, depth_img, mask_img):
        """
        计算所有指标。如果不提供pose_gt，则只计算无监督指标。
        """
        results = {}
        
        # 有监督指标 (需要GT)
        if pose_gt is not None:
            R_est = pose_est[:3, :3]
            t_est = pose_est[:3, 3]
            R_gt = pose_gt[:3, :3]
            t_gt = pose_gt[:3, 3]
            results['rotation_error_deg'] = self.rotation_error(R_est, R_gt)
            results['translation_error_mm'] = self.translation_error(t_est, t_gt)
            results['ADD_mm'] = self.add_metric(pose_est, pose_gt)
            results['ADD-S_mm'] = self.add_s_metric(pose_est, pose_gt)
            results['chamfer_distance_mm'] = self.chamfer_distance(pose_est, pose_gt)
            if rgb_img is not None:
                proj_errors = self.projection_2d_error(pose_est, pose_gt, rgb_img)
                results['projection_2d_error_px'] = proj_errors

        # 无监督指标 (需要深度图)
        if depth_img is not None:
            fitness, rmse = self.depth_fitness_score(pose_est, depth_img, mask_img)
            results['depth_fitness'] = fitness # 对齐率
            results['depth_rmse_mm'] = rmse    # 几何误差
        
        return results


def load_pose(pose_file):
    """加载位姿文件"""
    return np.loadtxt(pose_file).reshape(4, 4)


def evaluate_registration(mesh_file, gt_pose_dir, est_pose_dir, K, rgb_dir=None, depth_dir=None, mask_dir=None):
    """
    批量评估所有位姿
    """
    # 加载mesh
    mesh = trimesh.load(mesh_file)
    print(f"Mesh loaded: {mesh_file}")
    
    # 初始化评估器
    evaluator = RegistrationEvaluator(mesh, K)
    
    # 获取所有估计位姿文件（以估计位姿为基准）
    est_pose_files = sorted(glob.glob(os.path.join(est_pose_dir, "*.txt")))
    print(f"Found {len(est_pose_files)} estimated pose files to evaluate")
    
    all_results = []
    
    for est_pose_file in est_pose_files:
        filename = os.path.basename(est_pose_file)
        gt_pose_file = os.path.join(gt_pose_dir, filename) if gt_pose_dir else None
        
        # 加载估计位姿
        try:
            pose_est = load_pose(est_pose_file)
        except Exception as e:
            print(f"Error loading est pose {filename}: {e}")
            continue

        # 加载GT位姿（如果有）
        pose_gt = None
        if gt_pose_file and os.path.exists(gt_pose_file):
            pose_gt = load_pose(gt_pose_file)
        
        # 加载RGB图像（用于2D投影误差）
        rgb_img = None
        if rgb_dir:
            rgb_file = os.path.join(rgb_dir, filename.replace('.txt', '.png'))
            if os.path.exists(rgb_file):
                import cv2
                rgb_img = cv2.imread(rgb_file)

        # 加载深度图和掩码（用于无监督评估）
        depth_img = None
        if depth_dir:
            depth_file = os.path.join(depth_dir, filename.replace('.txt', '.png'))
            if not os.path.exists(depth_file):
                # 尝试其他可能的后缀
                depth_file = depth_file.replace('.png', '.exr')
            
            if os.path.exists(depth_file):
                import cv2
                if depth_file.endswith('.exr'):
                    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                else:
                    depth_img = cv2.imread(depth_file, -1) / 1000.0 # 假设单位mm转m
        
        mask_img = None
        if mask_dir:
            mask_file = os.path.join(mask_dir, filename.replace('.txt', '.png'))
            if os.path.exists(mask_file):
                import cv2
                mask_img = cv2.imread(mask_file, 0)
                if mask_img is None:
                    print(f"Warning: Mask not found for {filename}, calculating full scene distance!")
        
        # 评估
        try:
            results = evaluator.evaluate_all(pose_est, pose_gt, rgb_img, depth_img, mask_img)
            results['filename'] = filename
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {filename}: {e}")
            continue
    
    # 输出结果统计
    if all_results:
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nTotal frames evaluated: {len(all_results)}")

        # 1. 统计有监督指标
        if 'rotation_error_deg' in all_results[0]:
            for key in ['rotation_error_deg', 'translation_error_mm', 'ADD_mm', 'ADD-S_mm', 'chamfer_distance_mm']:
                vals = [r[key] for r in all_results]
                unit = "°" if "deg" in key else " mm"
                print(f"\n{key.replace('_', ' ').capitalize()}:")
                print(f"  Mean:   {np.mean(vals):.4f}{unit}")
                print(f"  Median: {np.median(vals):.4f}{unit}")
                print(f"  Std:    {np.std(vals):.4f}{unit}")

        # 2. 统计无监督指标
        if 'depth_fitness' in all_results[0]:
            fitness_vals = [r['depth_fitness'] for r in all_results]
            rmse_vals = [r['depth_rmse_mm'] for r in all_results]
            print(f"\nDepth Alignment Fitness (Higher is better, 0-1):")
            print(f"  Mean:   {np.mean(fitness_vals):.4f}")
            print(f"  Median: {np.median(fitness_vals):.4f}")
            
            print(f"\nDepth RMSE (Lower is better, mm):")
            print(f"  Mean:   {np.mean(rmse_vals):.4f} mm")
            print(f"  Median: {np.median(rmse_vals):.4f} mm")
        
        # 保存详细结果为JSON
        output_json = os.path.join(est_pose_dir, "..", "evaluation_results.json")
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_json}")
        print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="评估mesh配准精度")
    parser.add_argument("--mesh_file", type=str, default=f'{code_dir}/../FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj', help="Mesh文件路径")
    parser.add_argument("--est_pose_dir", type=str, default=f'{code_dir}/../debug_kinectCaptured_mesh/ob_in_cam', help="估计位姿目录")
    parser.add_argument("--gt_pose_dir", type=str, default=None, help="GT位姿目录(可选)")
    parser.add_argument("--K_file", type=str, default=f'{code_dir}/../FoundationPose_manual/kinectCapturedHead/cam_K.txt', help="相机内参文件")
    parser.add_argument("--rgb_dir", type=str, default=f'{code_dir}/../FoundationPose_manual/kinectCapturedHead/rgb', help="RGB图像目录（可选）")
    parser.add_argument("--depth_dir", type=str, default=f'{code_dir}/../FoundationPose_manual/kinectCapturedHead/depth', help="深度图目录（可选，用于无监督评估）")
    parser.add_argument("--mask_dir", type=str, default=f'{code_dir}/../masks_generated', help="掩码目录（可选）")
    
    args = parser.parse_args()
    
    # 加载相机内参
    K = np.loadtxt(args.K_file).reshape(3, 3)
    
    # 执行评估
    evaluate_registration(
        mesh_file=args.mesh_file,
        gt_pose_dir=args.gt_pose_dir,
        est_pose_dir=args.est_pose_dir,
        K=K,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir
    )
    
    # 执行评估
    # evaluate_registration(
    #     mesh_file=args.mesh_file,
    #     gt_pose_dir=args.gt_pose_dir,
    #     est_pose_dir=args.est_pose_dir,
    #     K=K,
    #     rgb_dir=args.rgb_dir
    # )
