"""
使用标定后的内参采集高精度数据
此脚本会使用您通过棋盘格标定得到的真实内参
"""
import cv2
import numpy as np
import os
import sys
import time

try:
    import pyk4a
    from pyk4a import Config, PyK4A
except ImportError:
    print("Error: pyk4a not found.")
    sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrated_K', type=str, default='cam_K_calibrated.txt',
                       help='标定得到的内参文件路径')
    parser.add_argument('--output_dir', type=str, default='kinect_data_with_calibrated_K',
                       help='输出目录')
    args = parser.parse_args()
    
    # 检查是否存在标定文件
    if not os.path.exists(args.calibrated_K):
        print(f"错误：未找到标定文件 {args.calibrated_K}")
        print("请先运行标定流程：")
        print("  1. python tools/calibrate_step1_capture.py")
        print("  2. python tools/calibrate_step2_compute.py")
        sys.exit(1)
    
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    
    # 加载标定好的内参
    K_calibrated = np.loadtxt(args.calibrated_K)
    
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()
    
    # 保存标定后的内参
    np.savetxt(os.path.join(output_dir, "cam_K.txt"), K_calibrated)
    
    print("="*60)
    print("使用标定内参采集数据")
    print("="*60)
    print(f"内参矩阵:\n{K_calibrated}\n")
    print("操作：按 's' 保存（含连拍去噪），按 'q' 退出\n")
    
    idx = 0
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None:
                img = capture.color[..., :3].copy()
                cv2.putText(img, f"Frames: {idx}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, "Using CALIBRATED intrinsics", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Capture with Calibrated K", img)
                
                key = cv2.waitKey(1)
                if key == ord('s'):
                    print(f"采集第 {idx} 帧（连拍去噪）...")
                    
                    # 连拍深度取中值
                    depth_list = []
                    for _ in range(5):
                        c = k4a.get_capture()
                        if c.transformed_depth is not None:
                            depth_list.append(c.transformed_depth)
                        time.sleep(0.01)
                    
                    depth_median = np.median(np.stack(depth_list), axis=0).astype(np.uint16)
                    rgb = capture.color[..., :3]
                    
                    # 保存（不做去畸变，因为标定已经包含了实际畸变）
                    cv2.imwrite(os.path.join(output_dir, "rgb", f"{idx:06d}.png"), rgb)
                    cv2.imwrite(os.path.join(output_dir, "depth", f"{idx:06d}.png"), depth_median)
                    
                    print(f"已保存第 {idx} 帧")
                    idx += 1
                elif key == ord('q'):
                    break
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
    
    print(f"\n采集完成！数据保存在 {output_dir}/")
    print("现在您可以直接用这些数据运行 FoundationPose")

if __name__ == "__main__":
    main()
