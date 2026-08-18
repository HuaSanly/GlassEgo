# GlassEgo 最小数据收集协议

本协议用于保存能够支持后续时间同步、相机与 IMU 标定及 VIO/SLAM 处理的最小原始数据。

## 1. 数据单元结构

每次连续录制对应一个独立数据单元：

```text
data/<unit>/
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

## 6. 最小有效性检查

一个数据单元至少应通过以下检查：

1. `video.mp4` 的可解码帧数等于 `camera.csv` 的数据行数。
2. `frame_idx` 连续，相机时间戳单调递增。
3. 每种 IMU 的 `sequence` 和 `timestamp_ns` 单调递增。
4. 标定分辨率与视频一致，IMU 时间范围覆盖整个视频。

RTP 时间戳、MP4 PTS、主机接收时间和 Unix 时间可以用于采集过程中的调试与帧配对，但完成 `camera.csv` 后不属于长期数据协议。

## 7. 参考格式

- [Kalibr bag format](https://github.com/ethz-asl/kalibr/wiki/bag-format)
- [Kalibr YAML formats](https://github.com/ethz-asl/kalibr/wiki/yaml-formats)
- [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
- [Basalt](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/README.md)
- [PennCOSYVIO file format](https://daniilidis-group.github.io/penncosyvio/file_format/)
