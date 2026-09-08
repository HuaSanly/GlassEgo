# GlassEgo 最小数据收集协议

本协议用于保存能够支持后续时间同步、相机与 IMU 标定及 VIO/SLAM 处理的最小原始数据。

## 1. 数据单元结构

每次连续录制对应一个独立数据单元：

```text
data/<task>/<unit>/
├── video.mp4
├── camera.csv
├── imu.csv
└── calibration.yaml
```

## 2. 视频

`video.mp4` 应满足：

- 使用 H.264 编码。
- 保持标定时使用的分辨率和图像方向。
- 不执行裁剪、旋转、缩放或视频防抖，除非重新进行相机标定。
- 一个数据单元只包含一次连续的相机运行。
- MP4 内部 PTS 只用于视频播放，VIO 使用 `camera.csv` 中的时间戳。

## 3. 相机帧时间戳

`camera.csv` 每行对应一个实际写入 `video.mp4` 的视频帧：

```csv
frame_idx,frame_id,rokid_timestamp_ns,device_monotonic_ns
0,1253,381264500000,91273455612000
1,1254,381297833000,91273488949000
```

字段含义：

- `frame_idx`：帧在 MP4 中的顺序，从 0 连续递增。
- `frame_id`：眼镜端生成的源帧编号，用于检测 WebRTC 丢帧；允许存在间隔。
- `rokid_timestamp_ns`：Rokid SDK 提供的原始相机时间戳。只转换为纳秒，不改变其时钟域。
- `device_monotonic_ns`：眼镜端收到该帧回调时读取的 `elapsedRealtimeNanos()`。

`device_monotonic_ns` 只用于建立相机时钟与 Android 单调时钟之间的映射，不直接作为相机曝光时间。

预处理阶段按每次连续录制估计：

```text
t_imu = a * t_rokid + b
```

其中 `a` 表示时钟漂移，`b` 表示时钟偏移。之后可以利用图像运动和陀螺仪数据进一步优化时间对齐。

## 4. IMU 数据

`imu.csv` 保存未经插值、滤波或坐标变换的原始加速度计和陀螺仪事件：

```csv
sensor_type,sequence,timestamp_ns,x,y,z
gyroscope,5816,91273452183456,0.012,-0.031,0.004
accelerometer,5822,91273452417391,0.148,9.721,-0.403
```

字段与单位：

- `sensor_type`：只允许 `gyroscope` 或 `accelerometer`。
- `sequence`：每种传感器独立递增的事件编号，用于检测传输丢包。
- `timestamp_ns`：直接保存 Android `SensorEvent.timestamp`。
- 陀螺仪 `x,y,z`：单位为 `rad/s`。
- 加速度计 `x,y,z`：单位为 `m/s²`，保留原始重力分量。

采集端不合并加速度计和陀螺仪事件。预处理阶段按时间戳排序，并根据 VIO 输入要求进行插值或重采样。

## 5. 标定数据

`calibration.yaml` 保存相机内参、畸变参数、相机与 IMU 外参以及 IMU 噪声参数：

```yaml
camera:
  model: pinhole
  resolution: [1920, 1080]
  intrinsics: [fx, fy, cx, cy]
  distortion_model: radtan
  distortion_coeffs: [k1, k2, p1, p2]

T_cam_imu:
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0.0, 0.0, 0.0, 1.0]

imu:
  gyro_noise_density: null
  gyro_random_walk: null
  accel_noise_density: null
  accel_random_walk: null
```

约定：

- `T_cam_imu` 将 IMU 坐标系中的点变换到相机坐标系，遵循 Kalibr 的定义。
- 相机内参和畸变参数必须对应实际保存的视频分辨率和方向。
- 未完成 IMU 噪声标定时，相关参数可以暂时为 `null`，正式运行 VIO 前再补齐。
- 采集端保存 IMU 原始坐标，不提前转换到相机坐标系。

## 6. 预处理世界坐标系

原始采集数据不定义全局世界坐标。Basalt VIO 完成后，所有持久化世界空间结果统一转换为右手 Aria MPS 坐标系：

- 原点：首帧 RGB 相机光心。
- `+X`：相对初始头部朝向的右方。
- `+Y`：反重力方向。
- `+Z`：初始头部朝向的反方向，即后方。
- 初始头部朝向：首帧 RGB 相机 OpenCV `+Z` 光轴在重力水平面上的投影。

RGB 相机局部坐标保持 OpenCV `X-right/Y-down/Z-forward`。`c2w` 遵循：

```text
p_world = c2w @ p_camera
world_frame = aria_mps_x_right_y_up_z_backward
```

Basalt 原生世界系只允许作为临时计算结果。`preprocess/temp_data/basalt_trajectory.csv` 中的 IMU 位姿、`poses.json` 中的相机位姿以及所有下游 `*_world` 字段必须使用同一个 Aria MPS 世界系并带有坐标元数据。旧 schema 或旧世界系结果不得与新结果混用，应通过重新预处理生成。

物体三角化结果 `preprocess/temp_data/object_3d_results.json` 使用 schema 3，并在顶层声明相同的
`world_frame`、`world_origin` 和 `initial_heading`。
`pose_method` 可以是全局字符串，也可以是按物体覆盖的映射；允许值为 `pca1`、`pca2`
或 `vlm`。`vlm` 姿态只负责从首个参考图像裁剪估计物体旋转，物体平移仍来自
Aria MPS 相机位姿下的多视图三角化。

物体后处理遵循 HumanEgo 的静态锚点约定：按名称排序后的首个 `obj*`（通常为 `obj1`）是
不可被手部锁定的静态 anchor；其余对象在手部从松开切换为抓取且与对象中心距离小于
`0.20 m` 时，保存手到物体的刚性变换并随手部世界位姿传播。松手后对象保留最后位姿。
全局中间结果写入 `preprocess/temp_data/`，质量检查图写入 `preprocess/vis/objects/`。

阶段划分结果使用 schema 4，并定义 `OPERATION`、`NON_OPERATION` 和 `FINISHED` 三种阶段。
按时间排序后，前 120 帧固定为 `NON_OPERATION`，最后 10 帧固定为 `FINISHED`，中间帧
由阶段算法判定为 `OPERATION` 或 `NON_OPERATION`。对象训练帧为 `OPERATION` 与 `FINISHED`
的并集；普通 `NON_OPERATION` 帧不进入对象传播、LaMa、VisualKpts 或 DatasetGen。

LaMa 和 VisualKpts 完成后，DatasetGen 只为上述对象训练帧写入：

```text
preprocess/
├── all_data/<frame>/
│   ├── rgb.png
│   ├── rgb_WoArm.png
│   ├── rgb_WArmObjKpts.png
│   ├── rgb_WoArm_WArmObjKpts.png
│   ├── mask_arm.png
│   ├── mask_arm_and_obj.png
│   └── training_data.json
├── temp_data/<frame>/       # object-centric 上下文帧
└── vis/                     # 按模块存放 PNG/JSON/LOG，MP4 直接位于此目录
```

`training_data.json` 的 `metadata.is_finished` 仅由阶段标签决定：`FINISHED` 帧写为 `1.0`，
`OPERATION` 帧写为 `0.0`，不得再根据最后一个操作帧推导完成尾段。

## 7. 最小有效性检查

一个数据单元至少应通过以下检查：

1. VIO 启动前顺序解码视频；少于 180 个可解码帧的数据单元跳过。该下限由 120 帧固定
   非操作阶段、至少 50 帧算法判定区间和 10 帧固定完成阶段组成。
2. `video.mp4` 的可解码帧数等于 `camera.csv` 的数据行数。
3. `frame_idx` 连续，相机时间戳单调递增。
4. 每种 IMU 的 `sequence` 和 `timestamp_ns` 单调递增。
5. 标定分辨率与视频一致，IMU 时间范围覆盖整个视频。
6. 持久化 VIO 和下游世界坐标产物声明 `aria_mps_x_right_y_up_z_backward`。

RTP 时间戳、MP4 PTS、主机接收时间和 Unix 时间可以用于采集过程中的调试与帧配对，但完成 `camera.csv` 后不属于长期数据协议。

## 8. 参考格式

- [Kalibr bag format](https://github.com/ethz-asl/kalibr/wiki/bag-format)
- [Kalibr YAML formats](https://github.com/ethz-asl/kalibr/wiki/yaml-formats)
- [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
- [Basalt](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/README.md)
- [PennCOSYVIO file format](https://daniilidis-group.github.io/penncosyvio/file_format/)
