# GlassEgo Compute

这是 GlassEgo 的离线预处理与 Flow Matching 训练子集。它不包含数据录制、设备标定、在线/机器人推理、仿真或 UI。

## 数据布局

```text
data/<task>/<unit>/
├── video.mp4
├── camera.csv
├── imu.csv
├── calibration.yaml
├── object_prompts.yaml
└── preprocess/
```

每个 task 至少需要两个有效 unit。完整字段、时间戳和 Aria MPS 世界坐标约定见 [DATA_STORAGE_PROTOCOL.md](DATA_STORAGE_PROTOCOL.md)。数据、权重和运行产物始终位于 Git 之外。

## 环境

目标平台是 Linux x86_64、Python 3.11、PyTorch 2.5.1、CUDA 12.1。环境由根目录的 `requirements.txt` 和 `setup.sh` 管理：

```bash
conda create -n glassego-compute python=3.11 pip -y
conda activate glassego-compute
bash setup.sh
```

Basalt 是唯一的外部 C++ 运行时。安装 `basalt_vio` 后确保它位于 `PATH`；已有有效 VIO 缓存时不需要重新执行 Basalt。

## 预处理

```bash
python preprocess/pipeline.py \
  --data-root /mnt/datasets/glassego \
  --units <task>/<unit>...
```

也可以使用现有的 `--units <task>/<unit>...` 和 `--config-root` 参数。结果写入每个 unit 的 `preprocess/` 目录。

## 训练

```bash
python -m training.FlowMatchingTrainer \
  --data_root /mnt/datasets/glassego \
  --runs_root /mnt/runs \
  --task <task> \
  --job baseline
```

训练消费 `preprocess/all_data/`，检查点、日志和评估结果写入 `runs/<task>/[<exp>/]<job>/`。

## 分支规则

`main` 是源码真源，`compute` 是从 `main` 投影得到的离线计算子集。不要把数据、模型、运行产物或独立的 `ui/` 仓库加入本仓库。
