# IVF-DecompFuse

红外—可见光图像融合框架：双流 Inception 编码器为 Baseline，依次接入 AID（自适应迭代分解）、DFD（可变形对齐与频域解耦）和 FreqCL（频带对比学习）。

## 数据目录

```text
data/MSRS/
  train/ir/*.png
  train/vi/*.png
  test/ir/*.png
  test/vi/*.png
```

红外与可见光图像按不带扩展名的文件名配对。训练默认从训练集固定划分 10% 作为验证集，不用测试集选择最佳检查点。

## GPU 环境（RTX 50 系列）

```bash
python -m pip install --upgrade torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 训练与消融

```bash
python train.py --config configs/default.yaml --ablation full
python train.py --config configs/improved.yaml --ablation full
python train.py --config configs/research.yaml --ablation full
python train.py --config configs/research/global_context_ft.yaml --ablation full
python train.py --config configs/research/global_cddfuse_ft.yaml --ablation full
python train.py --config configs/research/lifting_channelwise_fix.yaml --ablation full
python train.py --config configs/ablation/region_lifting_miqabf_photo.yaml --ablation full
python train.py --config configs/ablation/region_lifting_miqabf_photo_ft.yaml --ablation full
python -m scripts.blend_checkpoints --base checkpoints/ablation/region_lifting_miqabf/best.pth --finetuned checkpoints/ablation/region_lifting_miqabf_photo_ft/best.pth --output checkpoints/ablation/region_lifting_miqabf_blend/best.pth
bash scripts/train_msrs.sh
```

`baseline`、`aid`、`dfd`、`freqcl` 分别表示 Baseline、+AID、+DFD、+FreqCL；`full` 使用配置中的完整设置。`configs/improved.yaml` 额外启用最大强度像素目标、双模态浅层跳跃连接、DFD 偏移正则，并按验证集 SSIM 选择最佳模型。`configs/research.yaml` 在此基础上启用光照—显著性自适应强度目标、逐区域动态停止 AID 和可逆可学习提升小波，使用独立的 `checkpoints/research` 与 `logs/research`。`region_lifting_miqabf_photo.yaml` 使用区域 AID、提升小波和 MI/Qabf 联合选模，并在训练阶段增加双模态独立亮度、对比度与 gamma 增强；验证和测试不做光度增强。`region_lifting_miqabf_photo_ft.yaml` 从已有检查点进行低学习率弱增强微调，`scripts/blend_checkpoints.py` 按验证集 MI/Qabf 选择原始与微调权重的融合比例。`region_lifting_miqabf_photo_ft_distill.yaml` 额外通过 `train.distill_from` 加载冻结教师，并以 `lambda_distill` 约束学生输出；教师不参与推理。`global_context_ft.yaml` 在深层编码器中加入池化全局上下文残差块，新增门控从零初始化，加载旧检查点时可用 `init_strict: false` 保留原输出。`global_cddfuse_ft.yaml` 进一步通过 `teacher_type: cddfuse` 使用官方 CDDFuse 作为训练期教师，教师不参与推理。检查点保存对应配置，可通过 `--resume` 指定 `last.pth` 恢复优化器并续训，或通过 `--init_from best.pth` 仅初始化权重并使用新优化器微调；配置中的 `train.init_from` 也可记录初始化检查点。

## 测试、推理与鲁棒性

```bash
python test.py --ckpt checkpoints/full/best.pth --save_dir results/full --efficiency
python test.py --ckpt checkpoints/improved/best.pth --save_dir results/improved --efficiency
python test.py --ckpt checkpoints/research/best.pth --save_dir results/research --efficiency
python infer.py --ir data/MSRS/test/ir/00200D.png --vi data/MSRS/test/vi/00200D.png --ckpt checkpoints/full/best.pth --output fused.png
python robustness.py --ckpt checkpoints/full/best.pth --runs 3 --output results/robustness.json
```

测试输出 PSNR、SSIM、VIF、EN、SD、AG、SF、MI、Qabf，其中 SD、AG、SF 按 `[0,255]` 像素尺度输出，MI 为红外/可见光分别与融合图的互信息之和，Qabf 使用 Sobel 边缘强度和方向保持度；另可统计参数量、卷积 FLOPs、同设备推理时延和 FPS。融合图统一保存为无损 PNG，`metrics.json` 保存均值，`metrics.csv` 保存逐图指标。鲁棒性脚本支持平移、旋转、局部非刚性形变、弱光和噪声，输出多次运行的均值、标准差及 95% 置信区间。

旧版检查点不含网络配置，加载时显式指定原结构：`--ablation baseline`。默认配置和改进配置会在训练时排除与测试集同名的样本，不删除原始文件。M3FD 下游检测与 MSRS 语义分割需要相应数据、标注和固定任务模型，不包含在融合网络训练中。

## 价值引导的深度控制

当前控制器使用源图统计量和岭回归预测四个迭代深度的六指标效用，并在实际执行前选择深度。以下命令以 MSRS seed 42 为例；控制器训练数据已排除所有与测试集同名的图像：

```bash
python -m scripts.analyze_iteration_quality --ckpt checkpoints/research/haar_channelwise_control/best.pth --data_root data/MSRS --split train --exclude_test_overlap --output results/diagnostics/iteration_quality/haar_msrs_train --gpu 0 --modes minmax
python -m scripts.analyze_iteration_quality --ckpt checkpoints/research/haar_channelwise_control/best.pth --data_root data/MSRS --output results/diagnostics/iteration_quality/haar_msrs --gpu 0 --modes minmax
python -m scripts.evaluate_value_depth_control --train_metrics results/diagnostics/iteration_quality/haar_msrs_train.csv --test_metrics results/diagnostics/iteration_quality/haar_msrs.csv --data_root data/MSRS --target balanced --output results/diagnostics/value_depth_control/msrs_balanced.json
python -m scripts.run_value_controlled --ckpt checkpoints/research/haar_channelwise_control/best.pth --controller results/diagnostics/value_depth_control/msrs_balanced.npz --data_root data/MSRS --output_norm minmax --save_dir results/research/haar_value_control_minmax/msrs --gpu 0
```

评价脚本同时保存 `.json` 指标和 `.npz` 控制器参数。`run_value_controlled.py` 只执行选定的深度，输出深度直方图、平均深度和融合指标。效率统计包含重复模块的每次卷积调用，但不等同于完整 ATen 运算量；`results/efficiency/value_control_runtime.json` 另行记录包含控制器开销的端到端延迟。

## 快速自检

```bash
python smoke_test.py
# 或
python -m pytest smoke_test.py
```
