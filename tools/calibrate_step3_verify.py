"""
步骤3：验证标定结果
使用标定得到的内参重新投影棋盘格，检查误差
"""
import cv2
import numpy as np
import glob
import os
import argparse

def verify_calibration(images_dir, K_file, dist_file, pattern_size=(9, 6)):
    """验证标定精度"""
    K = np.loadtxt(K_file)
    dist = np.loadtxt(dist_file).reshape(-1, 1) if os.path.exists(dist_file) else None
    
    images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    print(f"使用内参:\n{K}\n")
    
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= 0.025
    
    total_error = 0
    count = 0
    
    for fname in images[:5]:  # 验证前5张
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            continue
        
        # 求解PnP
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        
        # 重投影
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        
        # 计算误差
        error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        count += 1
        
        # 可视化
        for i in range(len(corners)):
            cv2.circle(img, tuple(corners[i, 0].astype(int)), 5, (0, 0, 255), -1)  # 检测点（红色）
            cv2.circle(img, tuple(imgpoints2[i, 0].astype(int)), 3, (0, 255, 0), -1)  # 投影点（绿色）
        
        cv2.putText(img, f"Reprojection Error: {error:.3f} px", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Verification (Red: Detected, Green: Projected)', cv2.resize(img, (960, 540)))
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if count > 0:
        print(f"\n平均重投影误差: {total_error / count:.4f} 像素")
        print("如果误差 < 1 像素，说明标定非常成功！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='calibration_images')
    parser.add_argument('--K_file', type=str, default='cam_K_calibrated.txt')
    parser.add_argument('--dist_file', type=str, default='cam_K_calibrated_dist.txt')
    args = parser.parse_args()
    
    verify_calibration(args.images_dir, args.K_file, args.dist_file)

if __name__ == "__main__":
    main()
