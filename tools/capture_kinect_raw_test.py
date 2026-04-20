"""
临时测试脚本：采集原始数据（不去畸变）
用于对比去畸变前后的效果差异
"""
import cv2
import numpy as np
import os
import sys
import time

try:
    import pyk4a
    from pyk4a import Config, PyK4A, CalibrationType
except ImportError:
    print("Error: pyk4a not found.")
    sys.exit(1)

def main():
    output_base_dir = "kinect_raw_no_undistort_test"
    os.makedirs(os.path.join(output_base_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "depth"), exist_ok=True)

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()

    # 重要：直接使用原始内参（不经过getOptimalNewCameraMatrix）
    raw_K = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    np.savetxt(os.path.join(output_base_dir, "cam_K.txt"), raw_K)
    
    print(f"[模式] 原始数据采集（无去畸变）")
    print(f"内参K:\n{raw_K}")
    print("\n按 's' 保存，按 'q' 退出")

    idx = 0
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None:
                cv2.imshow("Live (Raw)", capture.color[..., :3])
                key = cv2.waitKey(1)
                
                if key == ord('s'):
                    print(f"采集第 {idx} 帧（连拍去噪中）...")
                    
                    # 连拍深度取中值
                    depth_list = []
                    for _ in range(5):
                        c = k4a.get_capture()
                        if c.transformed_depth is not None:
                            depth_list.append(c.transformed_depth)
                        time.sleep(0.01)
                    
                    if len(depth_list) > 0:
                        depth_median = np.median(np.stack(depth_list), axis=0).astype(np.uint16)
                    else:
                        continue

                    rgb_raw = capture.color[..., :3]

                    # 直接保存原始数据（不做任何去畸变）
                    cv2.imwrite(os.path.join(output_base_dir, "rgb", f"{idx:06d}.png"), rgb_raw)
                    cv2.imwrite(os.path.join(output_base_dir, "depth", f"{idx:06d}.png"), depth_median)
                    
                    print(f"已保存第 {idx} 帧（无去畸变）")
                    idx += 1
                elif key == ord('q'):
                    break
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
