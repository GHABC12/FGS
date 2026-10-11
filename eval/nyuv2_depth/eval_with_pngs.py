import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def compute_abs_rel(gt, pred):
    # 只计算 Abs Rel: |gt - pred| / gt 的平均值
    # 避免除以 0
    valid_mask = gt > 0
    abs_rel = np.mean(np.abs(gt[valid_mask] - pred[valid_mask]) / gt[valid_mask])
    return abs_rel


def build_gt_index(gt_path):
    print(f"🔍 正在扫描 GT 索引 (路径: {gt_path})...")
    gt_map = {}
    count = 0
    for root, dirs, files in os.walk(gt_path):
        for f in files:
            if "sync_depth_" in f and f.endswith(".png"):
                try:
                    name_no_ext = os.path.splitext(f)[0]
                    file_id = name_no_ext.split('_')[-1]
                    if file_id.isdigit():
                        gt_map[file_id] = os.path.join(root, f)
                        count += 1
                except:
                    continue
    print(f"✅ 扫描完成！共索引 {count} 个 GT 文件。")
    return gt_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--min_depth', type=float, default=1e-3)
    parser.add_argument('--max_depth', type=float, default=10.0)
    args = parser.parse_args()

    # 1. 获取预测文件
    # [解决方案 2] 优先寻找 .npy 高精度文件，找不到再找 .png
    pred_files = glob.glob(os.path.join(args.pred_path, "*.npy"))
    if len(pred_files) == 0:
        print("⚠️ 未找到 .npy 文件，尝试寻找 .png (精度可能较低)...")
        pred_files = glob.glob(os.path.join(args.pred_path, "*.png"))

    # 过滤掉可视化文件
    pred_files = [f for f in pred_files if "_vis" not in f]

    if len(pred_files) == 0:
        print("❌ Error: 没找到预测文件 (.npy 或 .png)")
        return

    # 2. 扫描 GT
    gt_index = build_gt_index(args.gt_path)
    if len(gt_index) == 0:
        print("❌ Error: GT 索引为空！")
        return

    abs_rel_list = []
    print(f"🚀 开始计算 Abs Rel ({len(pred_files)} 张)...")

    # [解决方案 1] 定义 Eigen Crop 区域 (针对 640x480 的标准 NYUv2 图像)
    # 格式: [top, bottom, left, right]
    eigen_crop = [45, 471, 41, 601]

    for pred_file in tqdm(pred_files):
        # 读取预测
        if pred_file.endswith('.npy'):
            # [解决方案 2] 读取高精度浮点数据
            pred = np.load(pred_file)
            # 如果是 (H, W, 3) 取第一通道
            if pred.ndim == 3:
                pred = pred[:, :, 0]
            # 确保是 float32
            pred = pred.astype(np.float32)
        else:
            # 兼容旧的 PNG 读取
            pred = cv2.imread(pred_file, -1)
            if pred is None: continue
            pred = pred.astype(np.float32) + 1e-6

        # 匹配 GT
        filename = os.path.basename(pred_file)
        try:
            # 兼容 .npy 和 .png 后缀
            pred_id = os.path.splitext(filename)[0].split('_')[-1]
        except:
            continue

        if pred_id in gt_index:
            gt_file = gt_index[pred_id]
        else:
            continue

        # 读取 GT (转为米)
        gt = cv2.imread(gt_file, -1)
        if gt is None: continue
        gt = gt.astype(np.float32) / 1000.0

        # [关键修正] 尺寸对齐
        # 标准评估是在 GT 分辨率 (640x480) 下进行的
        # 无论预测图多大 (e.g. 448x448)，都必须 resize 到 GT 的尺寸
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 掩码过滤
        mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

        # [解决方案 1] 应用 Eigen Crop
        # 创建裁剪 mask
        crop_mask = np.zeros_like(gt, dtype=bool)
        crop_mask[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]] = True

        # 合并 mask
        mask = np.logical_and(mask, crop_mask)

        pred_valid = pred[mask]
        gt_valid = gt[mask]

        if len(gt_valid) == 0: continue

        # 中值对齐 (Median Scaling) - 单目深度估计标准步骤
        scale = np.median(gt_valid) / np.median(pred_valid)
        pred_valid *= scale

        # 截断范围
        pred_valid[pred_valid < args.min_depth] = args.min_depth
        pred_valid[pred_valid > args.max_depth] = args.max_depth

        # 计算并保存
        val = compute_abs_rel(gt_valid, pred_valid)
        abs_rel_list.append(val)

    if len(abs_rel_list) == 0:
        print("❌ 错误: 无有效数据。")
        return

    mean_abs_rel = np.mean(abs_rel_list)

    print("\n" + "=" * 40)
    print(f"📊 最终结果: {os.path.basename(args.pred_path)}")
    print("=" * 40)
    print(f"Abs Rel  |  {mean_abs_rel:.4f}")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()