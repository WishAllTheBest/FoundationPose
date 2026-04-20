"""
从估计的位姿生成每一帧的 mask（用于评估）
通过将模型投影到图像平面，生成精确的前景掩码
"""
import numpy as np
import trimesh
import cv2
import os
import glob
from Utils import *
import argparse

def render_mask_from_pose(mesh, pose, K, img_shape):
    """
    根据位姿将mesh渲染成mask
    Args:
        mesh: trimesh对象
        pose: 4x4位姿矩阵
        K: 3x3内参
        img_shape: (H, W)
    Returns:
        mask: 二值图 (H, W)
    """
    H, W = img_shape
    
    # 变换mesh到相机坐标系
    vertices = mesh.vertices
    vertices_hom = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    vertices_cam = (pose @ vertices_hom.T).T[:, :3]
    
    # 投影到图像平面
    pts_2d = (K @ vertices_cam.T).T
    pts_2d = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)
    pts_2d = pts_2d.astype(np.int32)
    
    # 创建空白mask
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 绘制所有三角形面片
    faces = mesh.faces
    for face in faces:
        # 获取三个顶点
        v_idx = face
        pts = pts_2d[v_idx]
        
        # 检查深度（剔除相机后方的面）
        if (vertices_cam[v_idx, 2] < 0.01).any():
            continue
        
        # 检查是否在图像范围内
        if ((pts[:, 0] < 0) | (pts[:, 0] >= W) | 
            (pts[:, 1] < 0) | (pts[:, 1] >= H)).all():
            continue
        
        # 填充三角形
        cv2.fillPoly(mask, [pts], 255)
    
    return mask

def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    file_dir = "kinect_data_with_calibrated_K"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj', help='Path to mesh .obj file')
    parser.add_argument('--pose_dir', type=str, default=f'{code_dir}/debug_'+file_dir+'_mesh/ob_in_cam', help='Directory containing estimated poses')
    parser.add_argument('--rgb_dir', type=str, default=f'{code_dir}/FoundationPose_manual/'+file_dir+'/rgb', help='RGB images directory (to get image size)')
    parser.add_argument('--cam_K', type=str, default=f'{code_dir}/FoundationPose_manual/'+file_dir+'/cam_K.txt', help='Path to cam_K.txt')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/FoundationPose_manual/'+file_dir+'/masks_generated', help='Output directory for masks')
    args = parser.parse_args()
    
    # 加载mesh
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh_file)
    # mesh.apply_scale(1.1)
    # 加载内参
    K = np.loadtxt(args.cam_K).reshape(3, 3)
    
    # 获取图像尺寸（从第一张RGB图）
    rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, "*.png")))
    if len(rgb_files) == 0:
        print("Error: No RGB images found!")
        return
    
    sample_img = cv2.imread(rgb_files[0])
    H, W = sample_img.shape[:2]
    print(f"Image size: {W}x{H}")
    
    # 获取所有位姿文件
    pose_files = sorted(glob.glob(os.path.join(args.pose_dir, "*.txt")))
    print(f"Found {len(pose_files)} pose files")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 为每个位姿生成mask
    for i, pose_file in enumerate(pose_files):
        filename = os.path.basename(pose_file)
        pose = np.loadtxt(pose_file).reshape(4, 4)
        
        # 渲染mask
        mask = render_mask_from_pose(mesh, pose, K, (H, W))
        
        # 保存
        output_path = os.path.join(args.output_dir, filename.replace('.txt', '.png'))
        cv2.imwrite(output_path, mask)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(pose_files)} frames")
    
    print(f"\nAll masks saved to: {args.output_dir}")
#    print("Now you can re-run evaluate_registration.py with --mask_dir pointing to this folder.")

if __name__ == "__main__":
    main()
