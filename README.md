# GlassEgo

GlassEgo 是面向第一视角眼镜数据的处理与训练项目，包含 Rokid Glass3 标定、VIO、手部与物体跟踪、训练数据生成，以及 Flow Matching 策略训练。

## 数据结构

```text
data/<task>/<unit>/
├── video.mp4
├── camera.csv
├── imu.csv
├── calibration.yaml
├── object_prompts.yaml
└── preprocess/
```

每个 task 至少需要两个有效 unit 才能训练。完整数据约定见 [DATA_STORAGE_PROTOCOL.md](DATA_STORAGE_PROTOCOL.md)。

## 安装

```bash
conda create -n GlassEgo python=3.11 -y
conda activate GlassEgo

# 使用已有预处理数据进行训练
bash setup.sh

# 运行完整手部预处理
SKIP_HAND=0 bash setup.sh
```

## 标定

```bash
# 生成相机标定板
python datacollection/rokidglass3/calibration/calibration_pipeline.py board

# 相机内参
python datacollection/rokidglass3/calibration/calibration_pipeline.py camera \
  --unit data/calibration/camera_calibration

# IMU 噪声
python datacollection/rokidglass3/calibration/calibration_pipeline.py imu \
  --unit data/calibration/imu_calibration

# 相机与 IMU 外参
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic-board
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic \
  --unit data/calibration/cam_imu_calibration

# 验证标定结果
python datacollection/rokidglass3/calibration/calibration_pipeline.py validate
```

详细说明见 [标定文档](datacollection/rokidglass3/calibration/README.md)。

## 预处理

配置位于 `preprocess/config/`。命令会扫描全部 `data/<task>/<unit>`：

```bash
python preprocess/pipeline.py
```

产物写入各 unit 的 `preprocess/` 目录。

## 训练

```bash
# 使用默认配置
python -m training.FlowMatchingTrainer --task "<task>" --job baseline

# 使用 training/config/<task>/<job>.yaml
python -m training.FlowMatchingTrainer --task "<task>" --job baseline --use_cfg
```

`--job` 用于区分训练运行；结果写入 `runs/<task>/<job>/`。更多参数见 [训练文档](training/README.md)。
