# COCO Gaussian Noise Zero-shot 异常排查报告

日期：2026-07-17

## 结论

COCO PQ 异常低的主因是 instance prompt 资产落盘位置与评测读取位置不一致，并且项目根目录下后来补入的两张 prompt 文件内容错误。

评测代码读取：

```text
datasets/coco/pano_ca_inst/train_org/
```

正确生成资产实际位于：

```text
data/mmdet_custom/datasets/coco/pano_ca_inst/train_org/
```

## 资产证据

Prompt ID 固定为 `000000391460`。

| 资产 | 尺寸 | 内容统计 | 判定 |
|---|---:|---|---|
| 根目录 instance image | 491×640 | 12 色，32.46% 黑色 | 与 semantic target 的 SHA256 完全相同，错误 |
| 根目录 instance target | 491×640 | 1 色，100% 全黑 | 错误 |
| mmdet 生成 instance image | 1024×1024 | 433646 色，0% 黑色 | 正确 RGB prompt |
| mmdet 生成 instance target | 1024×1024 | 14 色，76.85% 黑色 | 正确实例位置编码 |

正确生成目录包含 9906 个文件；项目根评测目录仅有上述 2 个文件。生成配置中的 `target_path='datasets/coco/pano_ca_inst'` 是相对路径，任务从 `data/mmdet_custom` 运行后，正确数据因此落在其子目录中。

## 单变量验证

固定同一批 100 张 gaussian_noise severity=5 图像，只更换 instance prompt 路径：

| 配置 | PQ | SQ | RQ | PQ_th | PQ_st |
|---|---:|---:|---:|---:|---:|
| 原生 Painter + 错误 prompt | 14.0454 | 30.3510 | 17.5642 | 9.1617 | 21.1316 |
| 原生 Painter + 正确 prompt | 25.0056 | 49.4323 | 30.0951 | 27.6854 | 21.1173 |
| FC-VICT group1/wd0 + 正确 prompt | **26.2202** | **52.1805** | **32.2369** | **29.7006** | **21.2384** |

`PQ_st` 基本不变，而 `PQ_th` 从 9.16 提升到 27.69，形成了 instance prompt 根因的直接因果证据。

## 已排除项

- Painter checkpoint SHA256 与官方文件一致：`1bdd60ce013e82253e2a031726f36d07587947438b0e561d4338d6f0c1113f44`。
- 本地 `models_painter` 与上游 VICT shared Painter 对同一输入逐元素输出一致。
- COCO 80 个 thing、53 个 stuff 类别映射一致，133 类全部存在于 held-out 2000 子集。
- 最终 panoptic evaluator 的合并、颜色解码、GT 裁剪和 PQ 解析与上游一致。
- 错误运行中语义分支仍有正常颜色覆盖；异常集中在实例输出塌缩和低 RQ。

## 最佳候选参数

```text
winner_group=group1
steps=10
lr_prompt=2e-5
lr_backbone=0
weight_decay=0
prompt_length=30
vpt_target_layer=0 1 2 10 11 12 21 22 23
num_bands=2
lambda_gate=0.375
lambda_f=0.05
rho=0.05
seed=20260426
```

## Held-out 2000 验证

固定排除 dev100 的 2000 张样本，样本清单 SHA256：

```text
f01e2878be8d1472a514fa727a84b90c32385f4ce56d511a04519aa7f80ddd06
```

复用相同最佳候选已有的 2000 张 semantic 输出，仅使用正确 instance prompt 重跑受影响的实例分支。

运行目录：

```text
eval/coco_panoptic/gaussian_zero_shot_group1_correct_inst_prompt_val2000_codex_20260717
```

最终结果：

| PQ | SQ | RQ | PQ_th | PQ_st |
|---:|---:|---:|---:|---:|
| **24.7155** | 76.4364 | 29.9782 | 26.7585 | 21.6317 |

结果达到预设的 `PQ >= 24` 正常区间。与错误 instance prompt 的同一 held-out 2000 配置（PQ 14.1156）相比，纠正后提升 `+10.5999 PQ`。

### 完整性审计

- input、semantic PNG、instance PNG、样本清单各 2000 个唯一 ID，四个集合完全一致。
- 样本清单无重复，与固定 dev100 的交集为 0。
- 2000 个 instance PNG 全部通过 Pillow 完整性验证，无损坏文件。
- 样本清单 SHA256 与启动记录一致。
- `run.log` 中没有 Traceback、CUDA OOM、RuntimeError、Killed 或磁盘空间错误。
- 主运行进程正常退出并生成 `result.txt`。
- `git diff --name-only -- eval/coco_panoptic` 为空；现有 tracked 代码未被修改。
