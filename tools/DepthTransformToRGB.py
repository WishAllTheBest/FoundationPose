import numpy as np
import cv2

def align_depth_to_rgb(
    depth,
    K_d, K_rgb,
    R_d2rgb, t_d2rgb,
    rgb_shape
):
    """
    depth: (H_d, W_d) depth image (meters)
    rgb_shape: (H_rgb, W_rgb)
    return: depth aligned to RGB pixel space (H_rgb, W_rgb)
    """

    H_d, W_d = depth.shape
    H_rgb, W_rgb = rgb_shape

    fx_d, fy_d = K_d[0,0], K_d[1,1]
    cx_d, cy_d = K_d[0,2], K_d[1,2]
    fx_rgb, fy_rgb = K_rgb[0,0], K_rgb[1,1]
    cx_rgb, cy_rgb = K_rgb[0,2], K_rgb[1,2]

    aligned_depth = np.zeros((H_rgb, W_rgb), dtype=np.float32)

    # 遍历 depth 像素
    for v in range(H_d):
        for u in range(W_d):
            Z = depth[v, u]
            if Z <= 0:
                continue

            # Depth pixel → Depth camera 3D
            X = (u - cx_d) * Z / fx_d
            Y = (v - cy_d) * Z / fy_d
            P_d = np.array([X, Y, Z]).reshape(3,1)

            # Depth cam → RGB cam
            P_rgb = R_d2rgb @ P_d + t_d2rgb
            Xr, Yr, Zr = P_rgb.flatten()
            if Zr <= 0:
                continue

            # RGB cam 3D → RGB pixel
            ur = int(fx_rgb * Xr / Zr + cx_rgb)
            vr = int(fy_rgb * Yr / Zr + cy_rgb)

            if 0 <= ur < W_rgb and 0 <= vr < H_rgb:
                if aligned_depth[vr, ur] == 0 or Zr < aligned_depth[vr, ur]:
                    aligned_depth[vr, ur] = Zr

    return aligned_depth
if __name__ == "__main__":
    # 读入数据
    rgb = cv2.imread("rgb.png")
    depth_raw = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
    K_d = [[691.423, 0, 641.681],
             [0, 691.632, 348.506],
             [0.0, 0.0, 1.0]]
    K_rgb = [[691.423, 0, 641.681],
             [0, 691.632, 348.506],
             [0.0, 0.0, 1.0]]

    R_d2rgb = [[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]

    t_d2rgb = [0.0, 0.0, 0.0]
    # 如果 depth 是 mm，转成 meter
    # depth = depth_raw.astype(np.float32) / 1000.0
    depth = depth_raw
    aligned_depth = align_depth_to_rgb(
        depth,
        K_d=np.array(K_d),
        K_rgb=np.array(K_rgb),
        R_d2rgb=np.array(R_d2rgb),
        t_d2rgb=np.array(t_d2rgb).reshape(3,1),
        rgb_shape=rgb.shape[:2]
    )

    cv2.imwrite("depth_aligned.png", (aligned_depth * 1000).astype(np.uint16))
