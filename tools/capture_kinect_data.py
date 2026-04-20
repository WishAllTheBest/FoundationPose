import cv2
import numpy as np
import os
import sys

# 尝试导入 pyk4a，如果没有安装给出提示
try:
    import pyk4a
    from pyk4a import Config, PyK4A
except ImportError:
    print("错误: 未找到 'pyk4a' 库。")
    print("请先安装 Azure Kinect SDK 和 pyk4a 库。")
    print("例如: pip install pyk4a")
    print("注意: 在 Linux 上你需要先安装 Azure Kinect Sensor SDK (libk4a)。")
    sys.exit(1)

def main():
    # ---------------------------------------------------------
    # 配置部分
    # ---------------------------------------------------------
    output_base_dir = "kinect_data_captured" # 数据保存根目录
    
    # 按照 FoundationPose 的 datareader 要求建立目录结构
    # rgb/ 和 depth/ 文件夹
    rgb_dir = os.path.join(output_base_dir, "rgb")
    depth_dir = os.path.join(output_base_dir, "depth")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    print(f"数据将保存到: {output_base_dir}")

    # ---------------------------------------------------------
    # Kinect 配置
    # ---------------------------------------------------------
    # Resolution: RES_720P (1280x720) 符合你想要的常用分辨率
    # Depth Mode: NFOV_UNBINNED (高精度模式)
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
    )
    
    k4a = PyK4A(config)
    k4a.start()

    # ---------------------------------------------------------
    # 保存相机内参 (FoundationPose 需要 cam_K.txt)
    # ---------------------------------------------------------
    # 获取 Color 相机的内参 (因为我们将要把 Depth 对齐到 Color)
    # 这里的 intrinsics 格式通常是 [fx, 0, cx, 0, fy, cy, 0, 0, 1] 或者是 3x3 矩阵
    # pyk4a 的 calibration.get_camera_matrix 返回 3x3 numpy array
    intrinsic_matrix = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
    
    print("Color Camera Intrinsic Matrix:")
    print(intrinsic_matrix)
    
    np.savetxt(os.path.join(output_base_dir, "cam_K.txt"), intrinsic_matrix)
    print(f"内参已保存至: {os.path.join(output_base_dir, 'cam_K.txt')}")

    # ---------------------------------------------------------
    # 采集循环
    # ---------------------------------------------------------
    print("\n开始采集...")
    print("按 's' 保存当前帧")
    print("按 'q' 退出")
    
    idx = 0
    # 检查当前目录下是否已有文件，避免覆盖
    while os.path.exists(os.path.join(rgb_dir, f"{idx:06d}.png")):
        idx += 1
    
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None and capture.depth is not None:
                # 获取 RGB 图像
                # 注意: capture.color 通常是 BGRA 格式
                color_image = capture.color
                if color_image.shape[2] == 4:
                    color_image = color_image[..., :3] # 去除 Alpha 通道，保留 BGR
                
                # 获取对齐后的深度图 (Transformed Depth)
                # 这个属性会自动将 Depth 投影到 Color 相机的视角和分辨率
                # 结果是 uint16 类型，单位毫米(mm)，与 Color 图像像素一一对应
                transformed_depth = capture.transformed_depth
                
                # 可视化
                cv2.imshow("RGB", color_image)
                
                # 深度图可视化 (仅用于显示)
                depth_vis = cv2.normalize(transformed_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Aligned Depth", depth_vis)
                
                key = cv2.waitKey(1)
                
                if key == ord('s'):
                    # 保存 RGB
                    rgb_filename = os.path.join(rgb_dir, f"{idx:06d}.png")
                    cv2.imwrite(rgb_filename, color_image)
                    
                    # 保存 Depth
                    # 直接保存为 16-bit PNG (单位 mm)
                    depth_filename = os.path.join(depth_dir, f"{idx:06d}.png")
                    cv2.imwrite(depth_filename, transformed_depth)
                    
                    print(f"已保存帧: {idx} -> {rgb_filename}")
                    idx += 1
                    
                elif key == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("程序中断")
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("采集结束")

if __name__ == "__main__":
    main()
