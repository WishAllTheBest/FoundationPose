import cv2
import numpy as np
import os
import sys
import time

# 尝试导入 pyk4a
try:
    import pyk4a
    from pyk4a import Config, PyK4A
except ImportError:
    print("错误: 未找到 'pyk4a' 库。请运行: pip install pyk4a")
    sys.exit(1)

def preprocess_depth(depth_frames, use_bilateral=True):
    """
    深度图预处理优化方案：
    1. 时间维度：多帧中值滤波 (Temporal Median Filter)，消除 iToF 随机跳变噪声。
    2. 空间维度：双边滤波 (Bilateral Filter)，在保持边缘锐度的同时，平滑平面噪点。
    """
    # --- 1. 时间中值滤波 ---
    # 相比于平均值，中值滤波能完美剔除“飞点”和这种极端的离群噪点
    depth_stack = np.stack(depth_frames, axis=0)
    depth_median = np.median(depth_stack, axis=0).astype(np.uint16)

    # --- 2. 空间双边滤波 ---
    if use_bilateral:
        # 双边滤波要求输入为 float32
        depth_float = depth_median.astype(np.float32)
        # 参数调整建议：
        # d=5: 邻域直径
        # sigmaColor=20: 颜色空间标准差。对于单位为mm的深度图，20代表平滑20mm以内的起伏
        # sigmaSpace=5: 坐标空间标准差
        filtered_depth = cv2.bilateralFilter(depth_float, 5, 20, 5)
        depth_final = filtered_depth.astype(np.uint16)
    else:
        depth_final = depth_median

    return depth_final

def main():
    # ---------------------------------------------------------
    # 配置
    # ---------------------------------------------------------
    output_base_dir = "kinect_data_captured_v3"
    BURST_SIZE = 5  # 关键改进：按下s键时连拍5帧进行时间维度的融合
    
    rgb_dir = os.path.join(output_base_dir, "rgb")
    depth_dir = os.path.join(output_base_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    config = Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, # 确保使用最精准的窄视场模式
        camera_fps=pyk4a.FPS.FPS_30,
        synchronized_images_only=True,
    )
    
    k4a = PyK4A(config)
    k4a.start()
    
    # 自动保存匹配 720P RGB 空间的内参 (FoundationPose 必需)
    K = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
    np.savetxt(os.path.join(output_base_dir, "cam_K.txt"), K)
    
    print(f"数据保存路径: {output_base_dir}")
    print(f"模式: NFOV_UNBINNED, 连拍融合数: {BURST_SIZE} (Burst Mode)")

    idx = 0
    print("\n[操作提示]")
    print("按 's' 连拍融合并保存 (自动双边滤波 + 时间中值)")
    print("按 'q' 退出")
    
    try:
        while True:
            capture = k4a.get_capture()
            if capture.color is not None:
                # 实时预览
                color_vis = capture.color[..., :3].copy()
                cv2.imshow("Kinect Live (RGB)", color_vis)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"\n正在采集并融合第 {idx} 帧，请保持相机不动...")
                    rgb_burst = []
                    depth_burst = []
                    
                    # --- 核心数据预处理逻辑：连拍 ---
                    for i in range(BURST_SIZE):
                        cap = k4a.get_capture()
                        if cap.color is not None and cap.transformed_depth is not None:
                            rgb_burst.append(cap.color[..., :3].copy())
                            depth_burst.append(cap.transformed_depth.copy())
                        time.sleep(0.01) # 短暂延迟，获取不同时刻的噪声分布
                    
                    if len(depth_burst) >= 3:
                        # 1. 深度图预处理 (中值 + 双边)
                        processed_depth = preprocess_depth(depth_burst)
                        # 2. RGB 选取中间一帧（减少连拍过程及由于轻微晃动产生的位移）
                        final_rgb = rgb_burst[len(rgb_burst)//2]
                        
                        # 保存文件
                        rgb_path = os.path.join(rgb_dir, f"{idx:06d}.png")
                        depth_path = os.path.join(depth_dir, f"{idx:06d}.png")
                        cv2.imwrite(rgb_path, final_rgb)
                        cv2.imwrite(depth_path, processed_depth)
                        
                        print(f"保存完成: {idx:06d} (已应用融合去噪)")
                        
                        # 可视化反馈
                        depth_vis = cv2.normalize(processed_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow("Processed Depth Feedback", depth_vis)
                        cv2.waitKey(300) 
                        
                        idx += 1
                    else:
                        print("错误：连拍帧数不足，保存失败。")

    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("采集任务已关闭。")

if __name__ == "__main__":
    main()

