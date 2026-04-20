"""
步骤2：根据棋盘格图像计算真实的相机内参
"""
import cv2
import numpy as np
import glob
import os

def calibrate_camera(images_dir, pattern_size=(9, 6), square_size=0.025):
    """
    使用棋盘格图像标定相机
    Args:
        images_dir: 棋盘格图像目录
        pattern_size: 棋盘格内角点数量 (宽, 高)，默认9x6
        square_size: 棋盘格每个格子的实际尺寸（米），默认2.5cm
    Returns:
        K: 3x3 内参矩阵
        dist: 畸变系数
        rvecs, tvecs: 外参
    """
    # 准备棋盘格的3D坐标 (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 缩放到实际尺寸
    
    # 存储所有图像的角点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    
    images = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    print(f"找到 {len(images)} 张图片")
    
    img_shape = None
    successful = 0
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            
            # 可视化（可选）
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
            cv2.imshow('Detected Corners', cv2.resize(img, (960, 540)))
            cv2.waitKey(200)
            
            successful += 1
            print(f"✓ {os.path.basename(fname)}")
        else:
            print(f"✗ {os.path.basename(fname)} - 未检测到棋盘格")
    
    cv2.destroyAllWindows()
    
    if successful < 10:
        print(f"\n警告：成功标定的图片太少（{successful}张），建议至少15张")
        return None, None, None, None
    
    print(f"\n开始标定（使用 {successful} 张图片）...")
    
    # 执行标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if ret:
        print("\n标定成功！")
        print(f"重投影误差 (RMS): {ret:.4f} 像素")
        print(f"\n内参矩阵 K:\n{K}")
        print(f"\n畸变系数:\n{dist.ravel()}")
        
        return K, dist, rvecs, tvecs
    else:
        print("标定失败！")
        return None, None, None, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='calibration_images')
    parser.add_argument('--pattern_width', type=int, default=9, help='棋盘格内角点宽度')
    parser.add_argument('--pattern_height', type=int, default=6, help='棋盘格内角点高度')
    parser.add_argument('--square_size', type=float, default=0.025, help='格子实际尺寸(米)')
    parser.add_argument('--output', type=str, default='cam_K_calibrated.txt')
    args = parser.parse_args()
    
    K, dist, rvecs, tvecs = calibrate_camera(
        args.images_dir,
        pattern_size=(args.pattern_width, args.pattern_height),
        square_size=args.square_size
    )
    
    if K is not None:
        np.savetxt(args.output, K)
        np.savetxt(args.output.replace('.txt', '_dist.txt'), dist)
        print(f"\n内参已保存到: {args.output}")
        print(f"畸变系数已保存到: {args.output.replace('.txt', '_dist.txt')}")
        print("\n下一步：用新内参重新采集数据或替换原有的 cam_K.txt")

if __name__ == "__main__":
    main()
