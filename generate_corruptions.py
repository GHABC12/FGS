import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from imagecorruptions import get_corruption_names, corrupt


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate ImageNet-C style corrupted datasets.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='原始干净图片所在的文件夹路径 (例如: datasets/ade20k/images/validation)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='损坏图片存放的根目录 (例如: datasets/ade20k-c)')
    parser.add_argument('--sub_path', type=str, default='',
                        help='级别文件夹之后的子路径，用于对齐项目的 JSON 路径要求')
    parser.add_argument('--subset', type=str, default='all',
                        choices=['all', 'common', 'noise', 'blur', 'weather', 'digital'],
                        help='要应用的损坏类型集合，默认应用所有 15 种')
    parser.add_argument('--max_images', type=int, default=0,
                        help='最多处理多少张图片。<=0 表示处理全部图片')
    return parser.parse_args()


def main():
    args = parse_args()

    # 获取需要处理的损坏类型列表 (15种) 和 严重级别 (1-5)
    corruptions = get_corruption_names(args.subset)
    severities = [5]

    # 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png')
    img_names = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_exts)])
    if args.max_images and args.max_images > 0:
        img_names = img_names[:args.max_images]

    if not img_names:
        print(f"❌ 在 {args.input_dir} 中没有找到图片，请检查路径！")
        return

    print(f"🔍 将处理 {len(img_names)} 张图片.")
    print(
        f"⚙️ 准备应用 {len(corruptions)} 种损坏类型, 每种 {len(severities)} 个级别 (共 {len(corruptions) * len(severities)} 个变体).")
    print(f"📁 目标根目录: {args.output_root}\n")

    # 外层循环：遍历每一张图片 (这样做可以减少重复读取原图的I/O开销)
    for img_name in tqdm(img_names, desc="处理图片进度"):
        img_path = os.path.join(args.input_dir, img_name)

        try:
            # 读取图片并转换为 NumPy 数组 (必须是 uint8 且 RGB 格式)
            img_pil = Image.open(img_path).convert('RGB')
            img_np = np.array(img_pil)
        except Exception as e:
            print(f"\n⚠️ 读取图片 {img_path} 失败跳过: {e}")
            continue

        # 遍历每一种损坏类型
        for c in corruptions:
            # 遍历 1-5 级严重程度
            for s in severities:
                try:
                    # 调用 imagecorruptions 核心函数生成损坏后的数组
                    corrupted_np = corrupt(img_np, corruption_name=c, severity=s)
                    corrupted_pil = Image.fromarray(corrupted_np)

                    # 动态拼接目标文件夹路径
                    # 例如: datasets/ade20k-c/gaussian_noise/1/ade/ADEChallengeData2016/images/validation
                    save_dir = os.path.join(args.output_root, c, str(s), args.sub_path)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, img_name)

                    # 保持原有格式保存
                    corrupted_pil.save(save_path)
                except Exception as e:
                    print(f"\n⚠️ 处理图片 {img_name} 失败 (类型: {c}, 级别: {s}): {e}")

    print("\n✅ 所有图片转换完成！")


if __name__ == '__main__':
    main()