"""
交互式内参校准工具
通过滑动条调整 fx, fy, cx, cy，实时观察 mesh 投影是否与实物对齐
"""
import cv2
import numpy as np
import trimesh
import argparse
import os
from Utils import *

# 全局变量
mesh = None
pose = None
rgb = None
K_original = None

# 滑动条当前值（千分比，避免浮点数滑动条精度问题）
fx_scale = 1000  # 1.000 倍
fy_scale = 1000
cx_offset = 0    # 像素偏移
cy_offset = 0

def update_projection(_=None):
    """滑动条回调函数"""
    global fx_scale, fy_scale, cx_offset, cy_offset
    
    # 根据滑动条值构建新的 K
    K_new = K_original.copy()
    K_new[0, 0] = K_original[0, 0] * (fx_scale / 1000.0)  # fx
    K_new[1, 1] = K_original[1, 1] * (fy_scale / 1000.0)  # fy
    K_new[0, 2] = K_original[0, 2] + cx_offset             # cx
    K_new[1, 2] = K_original[1, 2] + cy_offset             # cy
    
    # 投影 mesh
    vis = rgb.copy()
    m_viz = mesh.copy()
    m_viz.apply_transform(pose)
    verts = m_viz.vertices
    
    proj = (K_new @ verts.T).T
    proj2 = (proj[:, :2] / (proj[:, 2:3] + 1e-8)).astype(int)
    
    # 绘制网格
    for face in m_viz.faces:
        pts = proj2[face]
        cv2.line(vis, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(vis, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(vis, tuple(pts[2]), tuple(pts[0]), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    
    # 显示当前参数
    cv2.putText(vis, f"fx_scale: {fx_scale/1000.0:.3f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"fy_scale: {fy_scale/1000.0:.3f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"cx_offset: {cx_offset}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"cy_offset: {cy_offset}", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, "Press 's' to save calibrated K", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("Intrinsic Calibration", vis)

def main():
    global mesh, pose, rgb, K_original, fx_scale, fy_scale, cx_offset, cy_offset
    
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    file_dir = "/home/lyn/code/FoundationPose/FoundationPose_manual/kinect_data_with_calibrated_K"
    parser.add_argument('--mesh_file', type=str, default='/home/lyn/code/FoundationPose/FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj')
    parser.add_argument('--pose_file', type=str, default='/home/lyn/code/FoundationPose/debug_kinect_data_with_calibrated_K_mesh/ob_in_cam/000007.txt')
    parser.add_argument('--rgb_file', type=str, default=file_dir+'/rgb/000007.png')
    parser.add_argument('--K_file', type=str, default=file_dir+'/cam_K.txt')
    parser.add_argument('--output_K', type=str, default=file_dir+'/cam_K_calibrated.txt')
    args = parser.parse_args()
    
    # 加载数据
    mesh = trimesh.load(args.mesh_file)
    pose = np.loadtxt(args.pose_file).reshape(4, 4)
    rgb = cv2.imread(args.rgb_file)
    K_original = np.loadtxt(args.K_file).reshape(3, 3)
    
    print("\n=== 交互式内参标定工具 ===")
    print(f"原始内参:\n{K_original}")
    print("\n操作说明:")
    print("  - 调整滑动条使绿色网格完美覆盖实物")
    print("  - fx/fy_scale: 控制焦距（影响投影大小）")
    print("  - cx/cy_offset: 控制中心点偏移")
    print("  - 按 's' 键保存校准后的内参")
    print("  - 按 'q' 键退出\n")
    
    # 创建窗口和滑动条
    cv2.namedWindow("Intrinsic Calibration")
    
    # fx_scale: 0.8 ~ 1.2 倍（800 ~ 1200）
    cv2.createTrackbar("fx_scale (x1000)", "Intrinsic Calibration", 1000, 1500, 
                       lambda x: globals().update(fx_scale=x) or update_projection())
    cv2.createTrackbar("fy_scale (x1000)", "Intrinsic Calibration", 1000, 1500, 
                       lambda x: globals().update(fy_scale=x) or update_projection())
    
    # cx/cy 偏移: -50 ~ +50 像素
    cv2.createTrackbar("cx_offset", "Intrinsic Calibration", 50, 100, 
                       lambda x: globals().update(cx_offset=x-50) or update_projection())
    cv2.createTrackbar("cy_offset", "Intrinsic Calibration", 50, 100, 
                       lambda x: globals().update(cy_offset=x-50) or update_projection())
    
    update_projection()
    
    while True:
        key = cv2.waitKey(10)
        if key == ord('s'):
            K_calibrated = K_original.copy()
            K_calibrated[0, 0] *= (fx_scale / 1000.0)
            K_calibrated[1, 1] *= (fy_scale / 1000.0)
            K_calibrated[0, 2] += cx_offset
            K_calibrated[1, 2] += cy_offset
            
            np.savetxt(args.output_K, K_calibrated)
            print(f"\n已保存校准后的内参到: {args.output_K}")
            print(f"新内参:\n{K_calibrated}")
            break
        elif key == ord('q'):
            print("未保存，退出")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
