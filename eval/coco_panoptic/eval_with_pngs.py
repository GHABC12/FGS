import argparse
import os
import subprocess
import tempfile


def parse_panoptic_metrics(stdout):
    metrics = {}
    wanted = {"PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) >= 2 and parts[0] in wanted:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        else:
            parts = stripped.split()
            if len(parts) >= 2 and parts[0] in wanted:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path_sem', type=str, required=True, help='语义分割推断路径')
    parser.add_argument('--pred_path_inst', type=str, required=True, help='实例分割推断路径')
    parser.add_argument('--gt_path', type=str, required=True, help='COCO GT标注路径')
    parser.add_argument('--dist_thr', type=float, default=19.0)
    parser.add_argument('--prompt', type=str, default="000000391460")
    parser.add_argument('--ckpt_file', type=str, default="painter_vit_large.pth")
    parser.add_argument('--input_size', type=int, default=448)
    args = parser.parse_args()

    print("🔍 正在整合并计算 Panoptic Quality (PQ)...")
    print("[Protocol] 当前评估器按预测 image id 裁剪 GT；subset 结果不能标成 full-val COCO PQ。")

    ckpt_tag = os.path.basename(args.ckpt_file)
    with tempfile.TemporaryDirectory(prefix="coco_pano_eval_") as tmp_work_dir:
        sem_dst = os.path.join(
            tmp_work_dir,
            f"pano_semseg_inference_{ckpt_tag}_{args.prompt}_size{args.input_size}",
        )
        inst_dst = os.path.join(
            tmp_work_dir,
            f"pano_inst_inference_{ckpt_tag}_{args.prompt}_size{args.input_size}",
        )
        os.symlink(os.path.abspath(args.pred_path_sem), sem_dst)
        os.symlink(os.path.abspath(args.pred_path_inst), inst_dst)

        # 1. 执行 COCO 实例评估计算 (Matrix NMS 合并)
        cmd_inst = (
            "python eval/coco_panoptic/COCOInstSegEvaluatorCustom.py "
            f"--work_dir \"{tmp_work_dir}\" "
            f"--ckpt_file \"{ckpt_tag}\" "
            f"--dist_thr {args.dist_thr} --prompt {args.prompt} --input_size {args.input_size}"
        )
        inst_result = subprocess.run(cmd_inst, shell=True, capture_output=True, text=True)
        if inst_result.returncode != 0:
            print("[Error] COCOInstSegEvaluatorCustom failed:")
            print(inst_result.stdout)
            print(inst_result.stderr)
            raise SystemExit(inst_result.returncode)

        # 2. 执行 COCO 全景分割评估并捕获输出
        cmd_pano = (
            "python eval/coco_panoptic/COCOPanoEvaluatorCustom.py "
            f"--work_dir \"{tmp_work_dir}\" "
            f"--ckpt_file \"{ckpt_tag}\" "
            f"--dist_thr {args.dist_thr} --prompt {args.prompt} --input_size {args.input_size}"
        )
        result = subprocess.run(cmd_pano, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("[Error] COCOPanoEvaluatorCustom failed:")
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(result.returncode)

    metrics = parse_panoptic_metrics(result.stdout)
    pq_val = metrics.get("PQ", 0.0)

    if pq_val == 0.0:
        print("[Warning] 未从输出流中捕获到有效的 PQ 分数！")
        print("====== 原始输出排错 =====")
        print(result.stdout)
        print("========================")

    print("\n" + "=" * 40)
    print("📊 最终结果 (合并 Inst & SemSeg)")
    print("=" * 40)
    for key in ["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]:
        if key in metrics:
            print(f"{key}  |  {metrics[key]:.4f}")
    if "PQ" not in metrics:
        print(f"PQ  |  {pq_val:.4f}")
    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
