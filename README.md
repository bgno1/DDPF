# DDPF：基于双域特征原型融合的小样本图像化恶意软件检测

本仓库为论文《基于双域特征原型融合的小样本图像化恶意软件检测》支撑代码。

## 数据路径

默认按以下路径组织数据集：

```text
../../dataset/maldeb/all
../../dataset/malimg/train
../../dataset/malimg/validation
```

Maldeb 的 `Benign` / `Malicious` 目录在自监督训练中只作为无标签图像池读取，不使用类别标签。Malimg validation 中的 `Rbotigen` 会在代码层映射为 train 中的 `Rbot!gen`，不修改原始数据。

## 环境

建议在已有 PyTorch CUDA 环境中安装以下依赖：

```bash
pip install -r requirements.txt
```

## 1. 构造小样本任务计划

```bash
python scripts/build_support_plan.py \
  --config configs/default.yaml \
  --shots 1,5,10,20 \
  --repeats 20 \
  --output results/support_plan_r20.json
```

## 2. 训练 Maldeb-SimCLR 编码器

```bash
python scripts/pretrain_malsim.py \
  --config configs/default.yaml \
  --epochs 20 \
  --batch-size 256 \
  --output checkpoints/malsim_resnet18_ep20.pth \
  --log-csv results/malsim_pretrain_log.csv
```

## 3. 运行小样本评估

```bash
python scripts/run_fewshot.py \
  --config configs/default.yaml \
  --support-plan results/support_plan_r20.json \
  --methods ImgNet-Proto,MalSim-Proto,DDPF-Oracle,DDPF-Adp \
  --malsim-checkpoint checkpoints/malsim_resnet18_ep20.pth \
  --output results/fewshot_results.csv
```

可选方法：

```text
ImgNet-1NN, MalSim-1NN,
ImgNet-Proto, MalSim-Proto,
ImgNet-LP, MalSim-LP,
ImgNet-FT, MalSim-FT,
DDPF-Oracle, DDPF-Adp
```

## 4. 汇总结果

```bash
python scripts/summarize_results.py \
  --input results/fewshot_results.csv \
  --output results/summary.csv
```


## 说明

`DDPF-Oracle` 只用于诊断固定融合权重的潜力；论文最终方法是 `DDPF-Adp`，其 `alpha` 选择严格依赖 support set，不访问 query label。
