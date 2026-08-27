# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Your Role: Expert Advisory Council

You are the user's trusted think tank, composed of four senior experts who provide honest, rigorous analysis and guidance for research and engineering implementation:

1. **Senior Embodied AI Engineer**: Focuses on perception-action loops, egocentric vision, sensor fusion, real-time performance, and end-to-end system integration
2. **Senior Mathematician**: Emphasizes mathematical rigor, coordinate system consistency, optimization formulations, numerical stability, and theoretical soundness
3. **Senior Robotics Engineer**: Concentrates on calibration quality, sensor synchronization, state estimation robustness, real-world deployment challenges, and hardware constraints
4. **Senior Architect**: Addresses system design, modularity, scalability, technical debt, API contracts, and long-term maintainability

**Your Mission**: When the user asks questions, think from all four perspectives and provide:
- **Diverse viewpoints**: Each expert may have different priorities and concerns
- **Brutally honest assessment**: Point out flaws, risks, and weaknesses without sugarcoating
- **Concrete actionable advice**: Not just "improve quality" but specific technical recommendations
- **Critical thinking**: Challenge assumptions, identify edge cases, and question design decisions
- **Inspiration**: Suggest alternative approaches, recent research directions, and creative solutions

**Response Format**: When providing multi-perspective analysis:
1. Lead with the most critical concern that needs immediate attention
2. Present each expert's view when they materially differ
3. Synthesize into a unified recommendation when consensus exists
4. Be direct: "This is fundamentally broken because..." not "This could potentially be improved..."
5. Prioritize correctness over politeness, depth over brevity

## Project Overview

GlassEgo is an egocentric vision pipeline for AR glasses (Rokid Glass3) that processes visual-inertial data through calibration, VIO (Visual-Inertial Odometry), hand tracking, phase segmentation, and object tracking. The system follows a data collection → preprocessing → training flow.

**Key Features:**
- Visual-inertial calibration (camera intrinsics, IMU noise, camera-IMU extrinsics via Kalibr)
- Monocular VIO using Basalt
- 3D hand tracking with HaMeR and MediaPipe/ViTPose detection
- Phase segmentation for motion analysis
- Object tracking with DINO-SAM and CoTracker

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n glassego python=3.11 -y
conda activate glassego

# Full installation (excludes hardware and hand tracking by default)
bash setup.sh

# Include hand tracking packages (MediaPipe, WiLoR, HaMeR)
SKIP_HAND=0 bash setup.sh

# Include hardware packages (pyrealsense2, trossen-arm)
SKIP_HARDWARE=0 bash setup.sh
```

### Calibration Workflow
All commands run from repository root:

```bash
# 1. Generate ChArUco calibration board
python datacollection/rokidglass3/calibration/calibration_pipeline.py board

# 2. Calibrate camera intrinsics
python datacollection/rokidglass3/calibration/calibration_pipeline.py camera \
  --unit data/camera_calibration

# 3. Calibrate IMU noise parameters
python datacollection/rokidglass3/calibration/calibration_pipeline.py imu \
  --unit data/imu_calibration

# 4. Generate AprilGrid for extrinsics
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic-board

# 5. Calibrate camera-IMU extrinsics (requires Kalibr installation)
python datacollection/rokidglass3/calibration/calibration_pipeline.py extrinsic \
  --unit data/cam_imu_calibration

# 6. Validate calibration file
python datacollection/rokidglass3/calibration/calibration_pipeline.py validate
```

### Preprocessing Pipeline

**主预处理流程：**
```bash
# Run full preprocessing pipeline (VIO → hands → phases → objects)
python preprocess/pipeline.py
```
这是生产环境的预处理入口点，不是测试套件。

自动发现 `data/` 下的数据单元并顺序处理。每个单元必须包含恰好一个视频文件，VIO 阶段还需要 `camera.csv`、`imu.csv` 和 `calibration.yaml`。

**VIO 独立验证工具：**
```bash
# Run only VIO validation with ground truth comparison
python preprocess/basalt_pipeline.py \
  --unit data/1 \
  --ground_truth data/1/ground_truth.csv \
  --force
```

用于 VIO 轨迹评估，输出 ATE RMSE、RPE 指标。

**两者的区别：**
- `pipeline.py`：完整流水线协调器，处理所有发现的单元
- `basalt_pipeline.py`：VIO 验证工具，支持与 ground truth 对比

### Testing
```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_coordinate_frames
python -m unittest tests.test_hand_world_kinematics

# Run single test case
python tests/test_coordinate_frames.py
```

**关键测试模式：**
- 坐标系测试验证 Aria MPS 世界坐标系对齐和变换正确性
- 手部轨迹测试验证 SLERP 插值和世界位姿填充
- 数值比较使用 `np.testing.assert_allclose` 并设置合适的容差

### Model Weights

模型权重在首次运行时自动下载。预先下载：
```bash
PREDOWNLOAD=1 bash setup.sh
```

**手部跟踪模型：**
- HaMeR: 3D 手部重建（自动下载到 `~/.cache/`）
- MediaPipe 或 ViTPose: 2D 手部检测
- WiLoR: 可选的替代手部姿态估计器

**对象跟踪模型：**
- DINO: 对象检测（通过 transformers 库）
- SAM: 分割
- CoTracker: 点跟踪

## Architecture

### Data Flow

```
datacollection/ → data/<unit>/ → preprocess/ → data/<unit>/preprocess/ → training/
```

1. **datacollection/**: Raw sensor data acquisition and calibration tools
2. **data/<unit>/**: One recording session (video + sensor data + calibration)
3. **preprocess/**: Processing modules that read from data units and write to `data/<unit>/preprocess/`
4. **training/**: Consumes preprocessed outputs (not yet implemented)

### Directory Structure

**`datacollection/rokidglass3/`**
- `calibration/`: Offline calibration pipeline (camera, IMU, extrinsics)
- Device-specific data collection code (real-time collection not yet integrated)

**`preprocess/`**
- `pipeline.py`: Main preprocessing CLI and coordinator (orchestrates all stages)
- `basalt_pipeline.py`: Standalone VIO validation tool with trajectory evaluation
- `config/`: YAML configurations (default, sensors, hand_tracking, vio, phase_segmentation, object_tracking)
- `data_types/`: Shared data structures (Cam, CamData, Hands, VIOResult, PhaseSequence, ObjectTrackingResult)
- `vio/`: Basalt VIO integration (BasaltVIOGenerator, sensor synchronization, pose processing)
- `hand_tracking/`: HaMeR 3D hand reconstruction, trajectory optimization, visualization
- `object_tracking/`: DINO-SAM segmentation, CoTracker tracking, 3D triangulation
- `phase_segmentation/`: Motion phase detection for HumanEgo task structure

**`utils/`**
- `utils_math.py`: Math utilities (rotation representations, transformations, timing)
- `utils_media.py`: Video/image I/O, camera data loading (`build_cam_from_disk`)
- `utils_vis.py`: Visualization helpers

**`ui/`**
- User interface components (separate AGENTS.md)

### Key Data Structures

**ProcessUnit** (`preprocess/pipeline.py`):
- `unit_dir`: Path to data unit
- `video_path`: Path to video file
- `pose_path`: Optional path to pose file (poses.json, pose.json, camera_poses.json)

**VIOResult** (`preprocess/data_types/VIOTypes.py`):
- Camera poses aligned with video frames from Basalt VIO
- Includes trajectory path, report, and pose data

**Hands** (`preprocess/data_types/HandsTypes.py`):
- Per-frame 3D hand keypoints (21 points per hand in HaMeR/OpenPose order)
- Includes MANO parameters, trajectory optimization results

**Cam/CamData** (`preprocess/data_types/CamTypes.py`):
- `Cam.cam`: List of RGB frames (numpy arrays)
- `Cam.tss`: Frame timestamps aligned with camera poses

### Configuration System

Configurations are loaded from `preprocess/config/`:
- `default.yaml`: Global paths, output flags, data root
- `sensors.yaml`: Sensor parameters (camera, IMU)
- `hand_tracking.yaml`: Hand detection and tracking parameters
- `vio.yaml`: Basalt VIO settings
- `phase_segmentation.yaml`: Phase detection parameters
- `object_tracking.yaml`: Object tracking settings

All configs are merged in `PreprocessPipeline._load_preprocess_config()` using OmegaConf.

### Data Unit Protocol

Each data unit follows `DATA_STORAGE_PROTOCOL.md`:

```
data/<unit>/
├── video.mp4               # H.264 encoded, native resolution
├── camera.csv              # Frame timestamps (frame_idx, frame_id, rokid_timestamp_ns, device_monotonic_ns)
├── imu.csv                 # Raw IMU events (sensor_type, sequence, timestamp_ns, x, y, z)
├── calibration.yaml        # Camera intrinsics, distortion, T_cam_imu, IMU noise
└── preprocess/             # Output directory
    ├── hands/
    ├── vio/
    ├── phases/
    └── objects/
```

### Processing Stages

1. **VIO** (`process_vio`): Basalt monocular VIO produces camera poses aligned with video frames
2. **Hands** (`process_hands`): HaMeR generates 3D hand keypoints per frame with trajectory optimization
3. **Phases** (`process_phases`): Scores VIO and cached hand evidence into binary OPERATION/NON_OPERATION phases
4. **Objects** (`process_objects`): DINO-SAM + CoTracker for object tracking in phase windows

Each stage validates input types, checks configuration flags (`enabled`), and writes outputs to `data/<unit>/preprocess/<stage>/`.

## Development Practices

### Code Organization Principles

- **Root directory**: Only repository-level docs, dependencies, and setup scripts
- **Module boundaries**: Preprocessing modules receive config via parameters, never read global YAML directly
- **Data types**: Shared structures live in `preprocess/data_types/`, not scattered across modules
- **Utils**: Only for genuinely reusable functions; hand-specific logic stays in `hand_tracking/`

### Configuration Management

- Only add config keys that code actually reads
- Remove obsolete config when removing features
- Never store credentials, absolute paths, or model weights in config files
- Camera intrinsics and distortion must match actual video resolution

### Timestamp Conventions

- VIO uses `camera.csv` timestamps, not MP4 PTS
- IMU uses Android `SensorEvent.timestamp` (device monotonic clock)
- Time synchronization: `t_imu = a * t_rokid + b` estimated per recording session
- All timestamps in nanoseconds unless specified otherwise

### Hand Tracking Details

- 21 keypoints per hand in HaMeR/OpenPose order (declared in `HaMeRHandsGenerator`)
- Coordinate system: camera frame unless transformed
- Pipeline: Detection (MediaPipe or ViTPose) → Crop → HaMeR 3D reconstruction → Trajectory optimization
- Output alignment: `len(hands.hands) == len(cam.cam) == len(cam.tss)`

### VIO Integration

- Basalt requires calibrated camera intrinsics and IMU parameters in `calibration.yaml`
- VIO output: `T_world_cam` transforms per frame in CSV format
- Evaluation metrics: ATE RMSE, RPE translation/rotation RMSE against ground truth
- Basalt adapter handles conversion from GlassEgo format to Basalt's expected inputs

### Important Constraints

- One video file per data unit (enforced by `_load_pending_units`)
- Video must not be cropped/rotated/stabilized unless camera is recalibrated
- Frame alignment is critical: all per-frame outputs must match video frame count
- IMU data is raw (no interpolation, filtering, or coordinate transforms until VIO)

## Preprocessing Outputs

每个阶段写入 `data/<unit>/preprocess/<stage>/`：

**VIO 输出 (`vio/`)：**
- `poses.json`: 每帧相机位姿，Aria MPS 世界坐标系
- `basalt_trajectory.csv`: IMU 位姿，带世界坐标系元数据
- `vio_report.txt`: 处理统计和质量指标

**手部跟踪输出 (`hands/`)：**
- `hands.json`: 每帧每只手 21 个关键点（HaMeR/OpenPose 顺序）
- `hands_visualization.mp4`: 带关键点叠加的可视化视频
- `trajectory_analysis.json`: 优化结果和统计信息

**阶段分割输出 (`phases/`)：**
- `phases.json`: 二值操作/非操作标签、置信度、手部证据及时间戳

**对象跟踪输出 (`objects/`)：**
- `objects.json`: 跟踪的对象轨迹及 3D 位置

## Data Validation

处理前验证每个数据单元：

```bash
# 检查帧数是否匹配
ffprobe -count_frames -select_streams v:0 -show_entries stream=nb_read_frames video.mp4
wc -l camera.csv  # 应该是 nb_read_frames + 1（含表头）
```

**完整 VIO 流水线所需文件：**
- `video.mp4`（H.264 编码，原生分辨率）
- `camera.csv`（与视频对齐的帧时间戳）
- `imu.csv`（原始加速度计和陀螺仪事件）
- `calibration.yaml`（相机内参、T_cam_imu、IMU 噪声参数）

**最小有效性检查：**
1. `video.mp4` 可解码帧数等于 `camera.csv` 数据行数
2. `frame_idx` 连续，相机时间戳单调递增
3. 每种 IMU 的 `sequence` 和 `timestamp_ns` 单调递增
4. 标定分辨率与视频一致，IMU 时间范围覆盖整个视频
5. 持久化 VIO 和下游世界坐标产物声明 `aria_mps_x_right_y_up_z_backward`

## Debugging & Troubleshooting

**VIO 处理失败：**
- 检查 `calibration.yaml` 分辨率是否与视频分辨率完全匹配
- 验证 IMU 时间戳覆盖整个视频持续时间
- 确保相机时间戳单调递增
- Basalt 需要足够的视觉特征；纯色或模糊视频可能失败

**手部跟踪问题：**
- HaMeR 需要 GPU；CPU 处理会极慢
- 检测失败：检查手是否可见且未严重遮挡
- 轨迹优化需要帧间有足够的有效检测
- 21 点顺序必须与 `HaMeRHandsGenerator` 声明一致

**时间戳同步：**
- VIO 使用 `camera.csv` 时间戳，不使用 MP4 PTS
- 时间同步估计：`t_imu = a * t_rokid + b`，每次录制单独估计
- 所有时间戳单位为纳秒，除非另有说明

**坐标系问题：**
- 唯一持久化世界系：Aria MPS（`aria_mps_x_right_y_up_z_backward`）
- Basalt 原生世界系仅用于临时计算
- 所有 `*_world` 字段必须使用同一世界系
- 初始头部朝向平行于重力轴会导致对齐失败

**配置相关：**
- 只添加代码实际读取的配置项
- 删除功能时同步删除失效配置
- 不保存凭据、绝对路径或模型权重

## Notes

- `pipeline.py` is the production preprocessing entry point, not a test suite
- The `training/` directory structure exists but implementation is incomplete
- Real-time data collection is not yet integrated; current workflow assumes offline processing
- Model weights are auto-downloaded on first run unless `PREDOWNLOAD=1` is set during setup
- Calibration uses ChArUco for camera intrinsics and AprilGrid for extrinsics (Kalibr requirement)
- See `AGENTS.md` for detailed development workflow and coding standards (Chinese)
