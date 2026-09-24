# ROS 无关异步推理系统设计（初版）

## 1. 目标和范围

本设计把现有 HumanEgo 推理模板拆成两个进程：

1. **Robot Client**：运行在 ROS1 双臂机器人侧，负责采集状态、执行动作、维护实时控制和本地安全策略。
2. **Policy Server**：运行在有 GPU 的远端机器，负责感知预处理、ICT 构建和模型推理。

网络层只使用 protobuf 定义的通用数据结构，不依赖 ROS message、TF、MoveIt 或具体机器人 SDK。ROS1 只存在于 Robot Client 的适配器中。

本版本只实现最小闭环：

```text
OpenSession
  -> SendObservation
  -> GetActionChunk
  -> 本地 ActionQueue 执行
  -> 重复 SendObservation / GetActionChunk
```

协议定义见 [`proto/inference.proto`](../proto/inference.proto)。

## 2. 当前仓库分析

当前 `inference/run_inference.py` 是同步单进程闭环：

```text
Camera -> Perception -> ICTPolicy -> TrajectoryController -> RobotArm
```

它已经定义了重要的数据约定：

- 相机 optical frame 是共享参考坐标系。
- 位姿是 4×4 SE(3) 矩阵，位置单位为米。
- gripper 取值 `[0, 1]`，0 表示打开，1 表示闭合。
- `T_align` 用于 hand frame 和 end-effector frame 的转换。

当前限制：

- `TrajectoryController.execute_chunk()` 在推理和动作执行之间同步阻塞。
- `ReferencePerception.estimate_objects()` 仍未实现。
- `TrossenArm` 引用了仓库中不存在的 `RobotArmTrossen`。
- `services.proto` 只提供裸 `bytes data`，没有 observation/action 的关联、时间索引和过期语义。
- `policy.py` 把 HumanEgo 的 ICT 和 checkpoint 细节直接放在运行循环附近，不适合作为机器人端依赖。

因此第一阶段不应直接把当前 `run_inference.py` 远程化，而应先固定跨进程数据契约，再拆分运行时职责。

## 3. 参考的异步模型

Robot Client 维护一个固定频率的本地控制循环：

```text
每个 control tick:
  1. 从 ActionQueue 取当前 tick 的动作
  2. 做安全检查、平滑和限速
  3. 发给 ROS1 机器人
  4. 采集最新 Observation
  5. 队列低于阈值时异步发送 Observation
```

另一个接收线程不断调用 `GetActionChunk`，收到动作后按照 `target_tick` 合并到队列。服务端的观测缓存最多保留最新的一帧，避免推理处理旧数据。

这种设计的关键不是让服务端控制机器人，而是让服务端生成未来一小段动作，机器人端始终拥有可执行的本地动作。

## 4. 两端职责

### 4.1 Robot Client

本轮服务端目录：

```text
inference/
├── policy_server.py
├── policy.py
└── proto/
    ├── inference.proto
    ├── inference_pb2.py
    └── inference_pb2_grpc.py
```

职责：

- 从 ROS1 订阅双臂关节、末端位姿、夹爪状态和相机帧。
- 通过适配器转换为 `ArmState` 和 `CameraFrame`。
- 维护单调递增的 `control_tick` 和 `observation_id`。
- 将 RGB 编码为 JPEG，将深度编码为 16 位 PNG 或其他明确编码。
- 维护线程安全的 ActionQueue。
- 执行动作前做 workspace、速度、加速度、IK 和通信 watchdog 检查。
- 网络失败时执行本地的保持、减速停止或安全姿态策略。

Robot Client 不应该加载 HumanEgo checkpoint，也不应该构建 ICT。

### 4.2 Policy Server

服务端职责集中在 `policy_server.py`：

```text
policy_server.py
policy.py
```

职责：

- 校验协议版本和会话状态。
- 每个会话只保存最新 Observation。
- 将网络数据转换为本地 `Frame`、`ObjectState` 和手部位姿。
- 完成目标检测、深度 lifting、物体姿态估计、object latch 和 clean image。
- 调用 `ICTPolicy` 生成未来轨迹。
- 将预测结果转换为带 `target_tick` 的 `ActionChunk`。
- 为 chunk 设置 `source_observation_id` 和 `expires_at_tick`。

模型的 `T_align`、ICT token 顺序、checkpoint normalization 等都属于服务端内部配置，不放进通用网络协议。

## 5. 最小 protobuf 设计

当前版本只有四个 RPC：

| RPC | 作用 |
|---|---|
| `OpenSession` | 协商版本、任务、策略和动作时间步 |
| `SendObservation` | 发送机器人状态和压缩后的相机数据 |
| `GetActionChunk` | 获取一段带时间索引的动作 |
| `CloseSession` | 结束会话 |

### 5.1 Observation

Observation 包含：

- `session_id`
- `observation_id`
- `control_tick`
- `capture_time_ns`
- 双臂 `ArmState`
- 已校正的左右目 `CameraFrame`

网络层不暴露 ICT。这样更换策略模型时，不需要改变 Robot Client。

### 5.2 ActionChunk

ActionChunk 包含：

- `chunk_id`
- `source_observation_id`
- `start_tick`
- `action_period_ns`
- `expires_at_tick`
- 多个 `ActionStep`
- 每个动作的 `arm_id`、末端位姿和 gripper
- `done_probability` 与 `terminal`

所有动作必须带 `frame_id`。初版约定动作目标使用相机 optical frame；ROS1 适配器在本地转换到机器人 base frame，再调用 IK 或 Cartesian servo。

### 5.3 双目图像传输

protobuf 只承载压缩后的左右目图像 bytes，不承载 Python pickle 或整个对象序列化结果：

- left/right：推荐 JPEG，左右图像必须已经完成双目校正。
- 服务端从 `inference/config/params.yaml` 引用的 calibration 文件读取左右投影矩阵。
- 服务端使用 StereoSGBM 计算米制深度，不再从客户端接收 depth bytes。

后续如果单帧超过 gRPC 限制，再增加分片消息；初版优先把图像压缩和消息大小限制配置好。

## 6. 异步控制流程

### 6.1 Robot Client 控制循环

```text
control_loop:
  action = queue.pop(target_tick=current_tick)

  if action is None:
      safety_controller.hold_or_stop()
  else:
      safety_controller.validate(action)
      robot_adapter.execute(action)

  observation = observation_builder.capture()

  if queue.remaining() / chunk_size < refill_threshold:
      observation_sender.submit_latest(observation)
```

Observation sender 使用有限长度队列。网络阻塞时应丢弃旧 Observation，只保留最新一帧。

### 6.2 Policy Server 推理循环

```text
SendObservation:
  validate(session_id, observation_id, frame shape)
  latest_observation.replace(observation)
  return ObservationAck

GetActionChunk:
  wait for latest observation, up to max_wait_ms
  preprocess observation
  run policy inference
  build action chunk
  return chunk
```

服务端不应该排队处理所有 Observation。旧帧对闭环控制没有价值，会增加 action 延迟。

## 7. ActionQueue 合并规则

Robot Client 收到新 chunk 后：

1. 丢弃 `target_tick <= last_executed_tick` 的动作。
2. 丢弃超过 `expires_at_tick` 的动作。
3. 新 tick 直接插入队列。
4. 相同 tick 的动作进行合并。

合并方式：

- position：加权平均。
- rotation：使用 Slerp，不对四元数分量直接平均。
- gripper：使用最新动作或保持策略。
- STOP：优先级高于普通动作。

队列为空时不能阻塞等待远端，而应进入本地安全行为。

## 8. 坐标系和时间约定

### 坐标系

初版统一使用：

```text
camera_optical:
  +x right
  +y down
  +z forward
```

网络协议中的 `Pose.frame_id` 必须填写实际坐标系名称。模型内部的 anchor frame、hand frame 和 `T_align` 不作为网络层坐标系。

### 时间

- `control_tick`：机器人端单调递增的控制序号，负责动作排序。
- `capture_time_ns`：机器人端 monotonic clock，用于延迟统计。
- 不用两个主机的 monotonic timestamp 直接相减。
- 需要跨主机日志关联时，再额外记录 wall-clock 时间。

## 9. 初始参数建议

如果当前 HumanEgo checkpoint 的动作时间步是 100 ms，初版可以使用：

```yaml
robot_control_hz: 50
policy_action_period_ms: 100
action_horizon: 20
refill_threshold: 0.4
max_action_age_ms: 250
server_latest_observation_only: true
```

`robot_control_hz` 与 `policy_action_period_ms` 可以不同。若机器人 servo 是 50 Hz，而策略是 10 Hz，需要在客户端插值或保持策略动作，不能直接把原始策略步长当成 50 Hz 动作。

## 10. 安全策略

服务端只能生成策略动作，不能绕过 Robot Client 的安全层。Robot Client 必须本地执行：

- workspace 限制
- 单步位置和旋转限幅
- 速度、加速度限幅
- IK 失败处理
- gripper 限制
- 通信超时 watchdog
- 人工急停和本地安全姿态

远端断连时，机器人端必须能独立安全停止。

## 11. 实施顺序

### 阶段一：本地异步化

先不接网络，把当前 `TrajectoryController` 拆成：

- `ActionQueue`
- 固定频率 `ControlLoop`
- 推理 worker
- `SafetyController`

用 fake policy 验证队列不会空转、动作不会乱序。

### 阶段二：协议和服务端

- 生成 Python/C++ protobuf 类。
- 实现 `OpenSession`、`SendObservation`、`GetActionChunk`。
- Policy Server 先接 fake policy，再接 `ICTPolicy`。
- 增加 stale action、超时和重连测试。

### 阶段三：ROS1 适配

实现：

```text
ROS1 joint/tf/camera -> Observation
ActionChunk -> ROS1 Cartesian / gripper command
```

ROS1 端不应修改 proto，也不应引入模型专属字段。

### 阶段四：真实感知和安全联调

- 完成 `ReferencePerception.estimate_objects()`。
- 验证 hand-eye extrinsic 和 `T_align`。
- 逐步增加动作 horizon 和控制频率。
- 注入网络延迟、丢包、服务端重启和 IK 失败。

## 12. 验收标准

初版完成的判断标准：

- Robot Client 可以在没有 Policy Server 时安全保持或停止。
- ActionQueue 可以按 `target_tick` 丢弃旧动作。
- 服务端不会处理过期 Observation 队列。
- 双臂动作在同一 `ActionStep` 中保持同步。
- 重复收到相同 `chunk_id` 不会重复执行。
- 所有位姿都能通过 `frame_id` 明确解释。
- Python 端和 ROS1/C++ 端可以完成一次 protobuf round trip。
