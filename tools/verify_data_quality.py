import numpy as np
import trimesh
import cv2
import os
import argparse
from Utils import *

def diagnostic_report(mesh_file, test_scene_dir):
    print("\n" + "="*50)
    print("      FoundationPose Data Diagnostic Report")
    print("="*50)

    # 1. 检查 Mesh 尺度 (Unit Check)
    print("\n[1] Checking Mesh Units...")
    mesh = trimesh.load(mesh_file)
    extents = mesh.extents
    print(f"Mesh file: {os.path.basename(mesh_file)}")
    print(f"Mesh extents (Width, Height, Depth): {extents}")
    
    # 逻辑判断：如果 extents 很大，通常是 mm
    is_mm = np.any(extents > 5.0) # 假设物体通常小于5米，如果超过5通常是mm为单位的100mm+
    if is_mm:
        print(">>> WARNING: Mesh dimensions seem to be in MILLIMETERS.")
        print(">>> FoundationPose expects METERS. You should divide vertices by 1000.")
    else:
        print(">>> OK: Mesh dimensions seem to be in METERS.")

    # 2. 检查内参 (Intrinsic Check)
    print("\n[2] Checking Intrinsics...")
    cam_K_file = os.path.join(test_scene_dir, "cam_K.txt")
    if os.path.exists(cam_K_file):
        K = np.loadtxt(cam_K_file).reshape(3,3)
        print(f"Intrinsic Matrix K:\n{K}")
        print(f"Principal Point (cx, cy): ({K[0,2]}, {K[1,2]})")
    else:
        print(">>> ERROR: cam_K.txt not found in test_scene_dir!")
        return

    # 3. 检查图像和深度图 (Image & Depth Check)
    print("\n[3] Checking Image & Depth Alignment...")
    rgb_files = sorted(glob.glob(os.path.join(test_scene_dir, "rgb/*.png")))
    depth_files = sorted(glob.glob(os.path.join(test_scene_dir, "depth/*.png")))
    
    if len(rgb_files) == 0 or len(depth_files) == 0:
        print(">>> ERROR: No RGB or Depth images found!")
        return
    
    rgb = cv2.imread(rgb_files[0])
    depth = cv2.imread(depth_files[0], -1) # 16-bit
    
    print(f"RGB Shape: {rgb.shape[:2]} (H, W)")
    print(f"Depth Shape: {depth.shape[:2]} (H, W)")
    
    if rgb.shape[:2] != depth.shape[:2]:
        print(">>> WARNING: RGB and Depth resolutions DO NOT MATCH!")
        print(f"    RGB: {rgb.shape[1]}x{rgb.shape[0]}, Depth: {depth.shape[1]}x{depth.shape[0]}")
    else:
        print(">>> OK: RGB and Depth resolutions match.")

    # 4. 检查深度值范围 (Depth Value Check)
    print("\n[4] Checking Depth Values...")
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        d_min, d_max, d_mean = valid_depth.min(), valid_depth.max(), valid_depth.mean()
        print(f"Depth Min: {d_min}, Max: {d_max}, Mean: {d_mean}")
        
        # 判断：如果 Mean 很大，比如 1000，那是 mm；如果很小 0.5，那是 m
        if d_mean > 50:
            print(">>> INFO: Depth map seems to be in MILLIMETERS (standard for Azure Kinect PNG).")
            depth_m = depth / 1000.0
        else:
            print(">>> INFO: Depth map seems to be in METERS.")
            depth_m = depth.astype(float)
    else:
        print(">>> ERROR: Depth map is empty (all zeros)!")
        return

    # 5. 可视化对齐情况 (Visual Verification)
    print("\n[5] Generating Visual Verification (check 'diagnostic_output' folder)...")
    os.makedirs("diagnostic_output", exist_ok=True)
    
    # 将深度图投影为点云，看是否符合人类直觉
    xyz_map = depth2xyzmap(depth_m, K)
    valid_mask = (depth_m > 0.1) & (depth_m < 2.0)
    pcd = toOpen3dCloud(xyz_map[valid_mask], rgb[valid_mask][..., ::-1])
    import open3d as o3d
    o3d.io.write_point_cloud("diagnostic_output/scene_check.ply", pcd)
    print(">>> Created 'diagnostic_output/scene_check.ply'. Please open in MeshLab to see if it looks like a 3D scene.")

    # 看看首帧 Mask 是否对得上
    mask_files = sorted(glob.glob(os.path.join(test_scene_dir, "masks/*.png")))
    if len(mask_files) > 0:
        mask = cv2.imread(mask_files[0], 0)
        overlay = rgb.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        cv2.imwrite("diagnostic_output/mask_overlay.png", overlay)
        print(">>> Created 'diagnostic_output/mask_overlay.png'. Check if the green mask aligns with the object.")

    print("\n" + "="*50)
    print("结论: 如果 [1] 提示是 mm，请缩小模型 1000 倍。")
    print("      如果 [3] 提示分辨率不匹配，FoundationPose 结果会完全偏移。")
    print("      请修复问题后重新运行评估。")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=f'{code_dir}/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj', help='Path to the .obj mesh file')
    parser.add_argument('--data', type=str, default=f'{code_dir}/FoundationPose_manual/kinectCapturedHead', help='Path to the data directory (containing rgb/, depth/, etc.)')
    args = parser.parse_args()
    
    diagnostic_report(args.mesh, args.data)
