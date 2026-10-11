# COCO-C Gaussian Noise Zero-Shot 排查与 dev100 验证

日期：2026-07-16

## 结论

1. 当前调参 CSV 的 PQ 是固定 100 图子集指标，不是 VICT 论文中的完整 5000 图 COCO-C PQ。相同 100 图上，未启用 VPT/FC、0 step 的 Painter baseline 也只有 14.1696，因此不能把 dev100 的 14 分与论文 full-val 的 Painter 24.9 / VICT 25.3 直接比较。
2. 原 zero-shot auto tuner 只为 SemSeg 生成 anchor/winner，并把同一个 winner 用于 SemSeg 和 Inst；COCO 是双分支任务，此缓存与任务错配是 COCO 特有问题。
3. 原 auto tuner 的 anchor/winner 固定 prompt_length=50，但实际推理使用 prompt_length=30，并且缓存键不含 task/prompt length，导致 winner 与推理配置不一致。
4. prompt length 对齐后，当前投票器在本候选上为两个任务都选择 group2，dev100 PQ 降至 9.0225；固定 group1 后恢复到 14.4427。现有频谱+梯度 winner 分数不能可靠代理最终 COCO PQ。
5. Painter checkpoint 主体完整匹配；只缺新增的 9 个 VPT prompt embeddings。COCO 数据、JSON、类别映射和 5000 张 clean/corrupted 图均完整。
6. 旧 painter_inference.py 对实例 RGB 编码使用 bilinear，历史 CSV 受污染；当前 painter_inference_new.py 已使用 Inst nearest，此项不是新实验低分原因。
7. 论文 VICT 是 encoder、60 steps、lr=1e-6、weight_decay=0；当前方法是 VPT prompt、10 steps、lr=2e-5、FC prompt calibration。两者不是同一方法，不能直接用论文 25.3 验收当前 dev100。

## 同一 dev100 实验

| 实验 | Sem winner | Inst winner | WD | PQ | PQ_th | PQ_st |
|---|---:|---:|---:|---:|---:|---:|
| Painter baseline，no FC/no VPT/0 step | none | none | n/a | 14.1696 | 9.3720 | 21.1309 |
| 修正 task/prompt-length 后自动投票 | group2 | group2 | 0.01 | 9.0225 | 5.8054 | 13.6905 |
| 固定 group1，零梯度保护，WD=0 | group1 | group1 | 0 | 14.4427 | 11.5207 | 18.7970 |

固定 group1 相对 baseline：+0.2731 PQ。自动 group2 相对 baseline：-5.1471 PQ。

## 新建文件

- coco_gaussian_zero_shot_baseline_probe.sh：固定 dev100 Painter sanity check。
- coco_gaussian_zero_shot_corrected_tune.sh：task-specific anchor/winner、prompt length 一致的候选脚本。
- painter_inference_coco_zero_shot_fixed.py：基于现有推理代码的新文件；增加零梯度除法保护和可配置 weight_decay=0，不修改原文件。
- coco_gaussian_zero_shot_corrected_tune_v2.sh：使用上述新推理文件的修正版候选脚本。
- coco_gaussian_zero_shot_group1_wd0_probe.sh：固定 group1、WD=0 的 dev100 对照。

## 建议的后续 100 图搜索

不要继续把 lambda_gate/num_bands 的投票 winner 当成连续超参结果。先缓存并直接评估 Sem/Inst 的 3×3 winner 组合，再在最佳组合上搜索：

- lr_prompt: 1e-6, 3e-6, 1e-5, 2e-5
- steps: 10, 30, 60
- rho: 0, 0.025, 0.05
- lambda_f: 0, 0.01, 0.05

所有结果必须标记为 dev100 subset PQ；按用户最新要求，本轮不运行完整 5000 图。
