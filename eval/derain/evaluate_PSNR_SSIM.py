"""
Notes:
- Convert images into YCbCr and only take the Y channel for computing SSIM and PSNR
- Don't convert images into float, just use uint8 directly
"""
import argparse
import os
import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import numpy as np


def compute_psnr(img1, img2):
    imdff = np.double(img1) - np.double(img2)
    print(imdff.sum())
    print(sum(img1.flatten()), sum(img2.flatten()))
    imdff = imdff.flatten()
    rmse = np.sqrt(np.mean(imdff ** 2))
    print(rmse)
    psnr = 20 * np.log10(255 / rmse)

    return psnr


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def compute_ssim(img1, img2):
    filter_size = 11
    sigma = 1.5
    K = [0.01, 0.03]
    L = 255

    # Generating a 2D Gaussian filter
    window = matlab_style_gauss2D((filter_size, filter_size), sigma)

    # Assuming img1 and img2 are the input images

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    # Normalize window
    window /= np.sum(window)

    # Convert img1 and img2 to double
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Compute local means using the window
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')

    # Compute local variances and covariances
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # Calculate SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    mssim = np.mean(ssim_map)
    return mssim


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(float) / 255.
    Y = 16 + (65.481 * im_rgb[:, :, 0] + 128.553 * im_rgb[:, :, 1] + 24.966 * im_rgb[:, :, 2])
    Y += 0.5
    Y = Y.astype(np.uint8)
    return Y


def get_args_parser():
    parser = argparse.ArgumentParser('Deraining', add_help=False)
    parser.add_argument('--pred_path', type=str,
                        help='path to the prediction results in png/jpg', required=True)
    parser.add_argument('--gt_path', type=str, default='datasets/derain/test',
                        help='root path to the groundtruth data', required=True)
    parser.add_argument('--sample_list', type=str, default=None,
                        help='Optional: only evaluate images listed in this file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    datasets = ['Test100', 'Rain100H', 'Rain100L', 'Test2800', 'Test1200']

    # 解析采样列表为每个子数据集的文件名集合
    sample_filter = {}  # {子数据集名: 文件名集合}
    if args.sample_list:
        with open(args.sample_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 从路径中提取子数据集名：.../Rain100H/input/31.png
                parts = line.replace('\\', '/').split('/')
                for ds_name in ['Rain100H', 'Rain100L', 'Test100', 'Test2800', 'Test1200']:
                    if ds_name in parts:
                        basename = os.path.basename(line)
                        if ds_name not in sample_filter:
                            sample_filter[ds_name] = set()
                        sample_filter[ds_name].add(basename)
                        break

    psnr_alldatasets = 0
    ssim_alldatasets = 0
    evaluated_sets = 0
    for idx_set in range(len(datasets)):

        file_path = os.path.join(args.pred_path, f'results/{datasets[idx_set]}')
        gt_path = os.path.join(args.gt_path, f'{datasets[idx_set]}/target')

        if not os.path.isdir(file_path) or not os.path.isdir(gt_path):
            print(f'Skipping {datasets[idx_set]}: missing pred or gt folder ({file_path} / {gt_path})')
            continue

        path_list = [f for f in os.listdir(file_path) if f.endswith('.jpg') or f.endswith('.png')]
        gt_list = [f for f in os.listdir(gt_path) if f.endswith('.jpg') or f.endswith('.png')]

        # 按采样列表过滤（如果提供）
        if sample_filter and datasets[idx_set] in sample_filter:
            allowed = sample_filter[datasets[idx_set]]
            path_list = [f for f in path_list if f in allowed]
            gt_list = [f for f in gt_list if f in allowed]

        img_num = len(path_list)
        if img_num == 0 or len(gt_list) == 0:
            print(f'Skipping {datasets[idx_set]}: no images found (pred={img_num}, gt={len(gt_list)})')
            continue

        total_psnr = 0
        total_ssim = 0
        path_list = sorted(path_list)
        gt_list = sorted(gt_list)
        print(len(path_list), len(gt_list))

        for j in tqdm.tqdm(range(img_num)):
            image_name = path_list[j]
            gt_name = gt_list[j]

            gt_img = Image.open(os.path.join(gt_path, gt_name))
            gt_img = np.array(gt_img)
            gt_img = rgb2ycbcr(gt_img)

            input_img = Image.open(os.path.join(file_path, image_name))
            input_img = np.array(input_img)
            input_img = rgb2ycbcr(input_img)

            ssim_val = compute_ssim(input_img, gt_img)
            psnr_val = psnr_loss(input_img, gt_img)

            total_ssim += ssim_val
            total_psnr += psnr_val

        qm_psnr = total_psnr / img_num
        qm_ssim = total_ssim / img_num

        print(f'For {datasets[idx_set]} dataset PSNR: {qm_psnr} SSIM: {qm_ssim}')

        psnr_alldatasets += qm_psnr
        ssim_alldatasets += qm_ssim
        evaluated_sets += 1

    if evaluated_sets > 0:
        print(f'For all datasets PSNR: {psnr_alldatasets / evaluated_sets} SSIM: {ssim_alldatasets / evaluated_sets}')
    else:
        print('For all datasets PSNR: nan SSIM: nan')
