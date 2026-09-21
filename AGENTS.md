# GlassEgo Compute 工作指南

本分支只承载离线预处理和训练。数据流为：

```text
data/<task>/<unit>/ -> preprocess/ -> preprocess/all_data/ -> training/
```

## 目录边界

- `preprocess/`：扫描数据单元、Basalt VIO、手部/物体模型前向、阶段划分和训练数据生成。
- `training/`：读取稳定的 `preprocess/all_data/`，执行 Flow Matching 训练和训练期评估。
- `utils/`：跨模块的小型数学、媒体和 artifact 工具；不要放入采集、模型生命周期或训练循环。
- `tests/`：正式单元测试和最小训练测试。
- 根目录 `requirements.txt`、`setup.sh`：唯一的 Python/Conda 环境入口。

数据、模型权重、Basalt 编译产物、checkpoint、日志和 `__pycache__/` 不进入 Git。独立的 `ui/` 目录不属于本仓库，禁止修改或强制加入索引。

## 不属于本分支

数据录制、标定工具、在线/机器人推理、仿真和 UI 均已从该分支移除。预处理内部为生成训练数据而进行的模型前向仍然属于本分支。

## 坐标系和数据契约

持久化世界坐标统一使用 `aria_mps_x_right_y_up_z_backward`。原始 Camera/IMU/Kalibr 坐标不得在采集端改轴；Basalt 原生世界系只允许存在于 VIO 内存或临时结果中。修改 JSON 字段、坐标含义或时间戳单位时，必须同步检查生产者、消费者和已有数据兼容性。

## 开发规则

通用修复先进入 `main`，再重新生成 `compute`。保持模块职责单一，避免在预处理入口实现算法或在训练代码中承担数据采集职责。运行测试前优先使用外部数据根目录和临时运行目录，避免覆盖已有数据。
