import cv2
import numpy as np
import os
import pyk4a
from pyk4a import Config, PyK4A

def main():
    output_base_dir = "kinect_high_precision_data"
    os.makedirs(os.path.join(output_base_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "depth"), exist_ok=True)

    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    k4a.start()

    # --- 核心：获取完美匹配去畸变后图像的新内参 ---
    # 获取原始标定
    calibration = k4a.calibration
    # 获取 Color 相机的原始内参和畸变系数
    raw_K = calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
    dist_coeffs = calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
    
    # 计算新内参（cv2.getOptimalNewCameraMatrix 可以保证去畸变后图像边缘不丢失）
    w, h = 1280, 720
    new_K, roi = cv2.getOptimalNewCameraMatrix(raw_K, dist_coeffs, (w, h), 0, (w, h))
    
    # 保存这个【新内参】，它是 FoundationPose 渲染 mesh 时必须使用的
    np.savetxt(os.path.join(output_base_dir, "cam_K.txt"), new_K)
    
    print("\n[系统就绪]")
    print(f"生成的 cam_K.txt 已根据去畸变逻辑进行了更新。")
    print("按 's' 保存数据 (自动执行: 连拍融合 + RGB/Depth 协同去畸变)")

    idx = 0
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None:
                cv2.imshow("Live", capture.color[..., :3])
                key = cv2.waitKey(1)
                
                if key == ord('s'):
                    # 连拍取中值去噪
                    depth_list = []
                    for _ in range(5):
                        cap = k4a.get_capture()
                        depth_list.append(cap.transformed_depth)
                    
                    depth_median = np.median(np.stack(depth_list), axis=0).astype(np.uint16)
                    rgb_raw = capture.color[..., :3]

                    # --- 协同去畸变：RGB 和 Depth 必须共用同一张 Map ---
                    map1, map2 = cv2.initUndistortRectifyMap(raw_K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
                    
                    # RGB 去畸变 (Lanczos4 插值保证纹理清晰)
                    img_undistorted = cv2.remap(rgb_raw, map1, map2, cv2.INTER_LANCZOS4)
                    
                    # 深度图去畸变 (必须用 INTER_NEAREST 保证深度值不被错误插值)
                    depth_undistorted = cv2.remap(depth_median, map1, map2, cv2.INTER_NEAREST)

                    # 保存结果
                    cv2.imwrite(os.path.join(output_base_dir, "rgb", f"{idx:06d}.png"), img_undistorted)
                    cv2.imwrite(os.path.join(output_base_dir, "depth", f"{idx:06d}.png"), depth_undistorted)
                    
                    print(f"已保存第 {idx} 帧，数据已完美对齐并去畸变。")
                    idx += 1
                elif key == ord('q'):
                    break
    finally:
        k4a.stop()