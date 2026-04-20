'''
liu yaning
2026.01.29
将原始RGB图像和生成的位姿估计结果图按中心700x700裁剪后横向拼接。
左侧：原始RGB，右侧：位姿估计结果。
并生成每7帧一组的竖向长图，以及包含所有帧的完整长图。
'''
import cv2
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

def center_crop(img, size=700):
    """
    裁剪图像中心的 size*size 区域。
    """
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    r = size // 2
    
    # 计算裁剪区域
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    
    return img[y1:y2, x1:x2]

def combine_images(rgb_dir, vis_dir, output_dir):
    """
    将原始RGB图像和生成的位姿估计结果图按中心 700x700 裁剪后横向拼接。
    左侧：原始RGB，右侧：位姿估计结果。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取所有可视化结果文件
    vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
    
    if len(vis_files) == 0:
        print(f"在 {vis_dir} 中未找到可视化图像。")
        return

    print(f"找到 {len(vis_files)} 张图像待处理...")

    all_combined = []

    for vis_path in tqdm(vis_files):
        filename = os.path.basename(vis_path)
        rgb_path = os.path.join(rgb_dir, filename)

        if not os.path.exists(rgb_path):
            print(f"警告: 在 {rgb_dir} 中找不到 {filename} 的原始RGB图。跳过。")
            continue

        # 读取图像
        img_rgb = cv2.imread(rgb_path)
        img_vis = cv2.imread(vis_path)

        if img_rgb is None or img_vis is None:
            print(f"警告: 无法读取 {filename}。跳过。")
            continue

        # 确保两张图片高度一致（主要是缩放位姿结果图以匹配RGB，如果原始尺寸不一致）
        h1, w1 = img_rgb.shape[:2]
        h2, w2 = img_vis.shape[:2]

        if h1 != h2:
            new_w2 = int(w2 * (h1 / h2))
            img_vis = cv2.resize(img_vis, (new_w2, h1))

        # 执行 700x700 中心裁剪
        img_rgb_cropped = center_crop(img_rgb, 700)
        img_vis_cropped = center_crop(img_vis, 700)

        # 横向拼接一张 (700 x 1400)
        combined = np.hstack((img_rgb_cropped, img_vis_cropped))
        all_combined.append(combined)

        # 保存单帧结果
        output_single_dir = os.path.join(output_dir, "single_frames")
        if not os.path.exists(output_single_dir): os.makedirs(output_single_dir)
        output_path = os.path.join(output_single_dir, filename)
        cv2.imwrite(output_path, combined)

    if all_combined:
        # 1. 生成所有图像合在一起的长图
        print("正在生成包含所有帧的完整长图...")
        full_vertical_strip = np.vstack(all_combined)
        full_strip_path = os.path.join(output_dir, "all_frames_full_strip.png")
        cv2.imwrite(full_strip_path, full_vertical_strip)
        print(f"完整完整长图已保存至: {full_strip_path}")

        # 2. 生成每7帧一组的竖向长图
        print("正在按每7帧一组生成分组长图...")
        group_size = 7
        for i in range(0, len(all_combined), group_size):
            group = all_combined[i : i + group_size]
            group_vertical_strip = np.vstack(group)
            
            group_idx = (i // group_size) + 1
            group_strip_path = os.path.join(output_dir, f"frames_group_{group_idx:02d}.png")
            cv2.imwrite(group_strip_path, group_vertical_strip)
            print(f"第 {group_idx:02d} 组长图已保存至: {group_strip_path}")

    print(f"处理完成。单帧对比图、分组长图及完整长图均已保存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将RGB原图和位姿估计结果图拼接显示。")
    parser.add_argument("--rgb_dir", type=str, default="./FoundationPose_manual/geminiHead/rgb", help="原始RGB图像目录")
    parser.add_argument("--vis_dir", type=str, default="./debug_mesh/track_vis", help="位姿估计结果图目录")
    parser.add_argument("--output_dir", type=str, default="debug_mesh/combined", help="拼接结果保存目录")

    args = parser.parse_args()

    # 处理路径
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    def get_abs_path(p):
        if os.path.isabs(p): return p
        return os.path.join(code_dir, p)

    rgb_dir = get_abs_path(args.rgb_dir)
    vis_dir = get_abs_path(args.vis_dir)
    output_dir = get_abs_path(args.output_dir)

    combine_images(rgb_dir, vis_dir, output_dir)
