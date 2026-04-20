"""
步骤1：采集棋盘格图像用于相机标定
使用说明：
1. 打印一张棋盘格标定板（9x6或10x7，黑白格子）
2. 运行此脚本，按's'保存当前帧
3. 从不同角度、距离拍摄15-20张图片
4. 确保棋盘格占据画面的大部分区域
"""
import cv2
import numpy as np
import os
import sys

try:
    import pyk4a
    from pyk4a import Config, PyK4A, CalibrationType
except ImportError:
    print("Error: pyk4a not found.")
    sys.exit(1)

def main():
    output_dir = "calibration_images"
    os.makedirs(output_dir, exist_ok=True)
    
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    
    print("="*60)
    print("相机标定 - 图像采集")
    print("="*60)
    print("\n准备工作：")
    print("  1. 准备一张棋盘格标定板（推荐 9x6 内角点）")
    print("  2. 确保棋盘格平整、光线均匀")
    print("\n操作说明：")
    print("  - 按 's' 保存当前帧")
    print("  - 从不同角度、距离拍摄15-20张")
    print("  - 尽量让棋盘格充满画面")
    print("  - 按 'q' 完成采集\n")
    
    idx = 0
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None:
                img = capture.color[..., :3].copy()
                
                # 在预览中绘制提示
                cv2.putText(img, f"Captured: {idx} images", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, "Press 's' to capture, 'q' to finish", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Calibration Capture", img)
                key = cv2.waitKey(1)
                
                if key == ord('s'):
                    filename = os.path.join(output_dir, f"calib_{idx:03d}.png")
                    cv2.imwrite(filename, capture.color[..., :3])
                    print(f"已保存: {filename}")
                    idx += 1
                elif key == ord('q'):
                    break
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
    
    print(f"\n采集完成！共保存 {idx} 张图片到 {output_dir}/")
    print("下一步：运行 calibrate_step2_compute.py 计算内参")

if __name__ == "__main__":
    main()
