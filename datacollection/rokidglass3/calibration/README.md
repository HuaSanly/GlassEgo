# Rokid Glass3 标定工具使用说明

所有命令均在仓库根目录执行。

## 1. 生成标定板

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py board
```

将生成的 `board/charuco_board.png` 按 A4 横向、100% 比例打印，不要启用自动缩放或适合页面。

## 2. 标定相机

准备一个相机标定单元：

```text
data/camera_calibration/
├── video.mp4
└── camera.csv
```

固定标定板并移动眼镜，从不同距离和角度拍摄，使标定板覆盖画面的不同区域，避免运动模糊。然后运行：

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py camera \
  --unit data/camera_calibration
```

结果写入：

- `calibration.yaml` 的 `camera` 段
- `report/camera_report.json`
- `report/camera_reprojection.png`

## 3. 标定 IMU 噪声

准备一个 IMU 标定单元：

```text
data/imu_calibration/
└── imu.csv
```

将眼镜完全静止放置。数据至少需要 60 秒，正式标定建议采集约 3 小时。然后运行：

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py imu \
  --unit data/imu_calibration
```

结果写入：

- `calibration.yaml` 的 `imu` 段
- `report/imu_report.json`
- `report/imu_allan.png`

## 4. 标定相机与 IMU 外参

先确认独立安装的 Kalibr 可以启动：

```bash
kalibr_calibrate_imu_camera --help
kalibr_create_target_pdf --help
```

生成 AprilGrid：

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic-board
```

将 `board/aprilgrid_board.pdf` 按 100% 比例打印，禁止使用“适合页面”或自动缩放。打印后测量每个黑色标签的边长，必须为 30 mm。

准备一个联合标定单元：

```text
data/cam_imu_calibration/
├── video.mp4
├── camera.csv
└── imu.csv
```

固定 AprilGrid，连续采集 60–90 秒。平滑移动整副眼镜，使标定板覆盖画面各区域，并充分覆盖三轴旋转和三向平移；避免运动模糊和剧烈冲击。然后运行：

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic \
  --unit data/cam_imu_calibration
```

成功后检查 `report/extrinsic_kalibr_report.pdf` 和 `report/extrinsic_results.txt` 中的重投影误差、IMU 误差与时间偏移。解析结果写入 `report/extrinsic_report.json`，完整日志写入 `report/extrinsic.log`，并更新 `calibration.yaml` 的 `T_cam_imu` 和 `timeshift_cam_imu`。

变换与时间偏移约定为：

```text
p_camera = T_cam_imu @ p_imu
t_imu = t_cam + timeshift_cam_imu
```

Kalibr 运行失败或输出校验失败时不会修改已有的 `calibration.yaml`。

## 5. 验证标定文件

```bash
python datacollection/rokidglass3/calibration/calibration_pipeline.py validate
```

输出 `"status": "valid"` 表示标定文件的字段、数值和外参矩阵结构合法。
