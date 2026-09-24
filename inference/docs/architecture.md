# 异步推理架构图

## A. 当前仓库的同步结构

当前代码主要集中在 `inference/run_inference.py`，推理和动作执行位于同一个循环中：

```mermaid
flowchart LR
    CAM[Camera / CamRS]
    PER[ReferencePerception]
    ICT[ICTPolicy]
    CTRL[TrajectoryController]
    ARM[RobotArm Adapter]

    CAM -->|RGB-D Frame| PER
    PER -->|clean image + object states| ICT
    ARM -->|EE pose + gripper| ICT
    ICT -->|future trajectory| CTRL
    CTRL -->|blocking execute_chunk| ARM
    ARM -->|new state| CAM
```

当前结构的问题是：

- 推理耗时期间无法持续执行动作。
- `execute_chunk()` 将控制、平滑、夹爪和等待放在同一调用中。
- 相机、感知、策略和机器人驱动的生命周期由一个脚本统一管理。
- 网络协议无法直接插入两个独立进程之间。

## B. 初版双端异步结构

```mermaid
flowchart LR
    subgraph ROBOT[ROS1 Robot Client]
        ROS[ROS1 双臂节点]
        CAM[Camera Adapter]
        OBS[Observation Builder]
        SEND[Observation Sender]
        QUEUE[Thread-safe ActionQueue]
        CTRL[50 Hz Control Loop]
        SAFE[Safety / Watchdog]
        GRPC_C[gRPC Client]

        ROS --> OBS
        CAM --> OBS
        OBS --> SEND
        SEND --> GRPC_C
        GRPC_C --> QUEUE
        QUEUE --> CTRL
        CTRL --> SAFE
        SAFE --> ROS
    end

    subgraph SERVER[Remote Policy Server]
        GRPC_S[gRPC Server]
        LATEST[Latest Observation Slot]
        PRE[HumanEgo Observation Builder]
        TRACK[Scene Tracker / Object Latch]
        POLICY[ICTPolicy]
        BUILD[ActionChunk Builder]

        GRPC_S --> LATEST
        LATEST --> PRE
        PRE --> TRACK
        TRACK --> POLICY
        POLICY --> BUILD
        BUILD --> GRPC_S
    end

    GRPC_C -->|Observation| GRPC_S
    GRPC_S -->|ActionChunk| GRPC_C
```

## C. 一次异步控制周期

```mermaid
sequenceDiagram
    participant C as Robot Client
    participant Q as ActionQueue
    participant S as Policy Server
    participant P as ICTPolicy
    participant R as ROS1 Robot

    loop every control tick
        C->>Q: pop action for target_tick
        Q-->>C: action or empty
        C->>R: safety check + execute action
        C->>C: capture latest observation
    end

    C->>S: SendObservation(observation_id, control_tick)
    S-->>C: ObservationAck
    C->>S: GetActionChunk(last_executed_tick, queue_size)
    S->>P: preprocess + infer latest observation
    P-->>S: future trajectory
    S-->>C: ActionChunk(target_tick range)
    C->>Q: merge by target_tick
```

## D. 数据边界

```mermaid
flowchart TB
    subgraph WIRE[Network Contract: inference.proto]
        O[Observation]
        A[ActionChunk]
        P[Pose / ArmState / CameraFrame]
    end

    subgraph CLIENT_ONLY[Robot Client Internal]
        ROSMSG[ROS messages]
        IK[IK / MoveIt / Servo]
        SAFETY[Local safety limits]
    end

    subgraph SERVER_ONLY[Policy Server Internal]
        ICTTOK[ICT token layout]
        ALIGN[T_align]
        PERCEPTION[Detector / Depth lifting / Inpainting]
        CKPT[Checkpoint and normalization]
    end

    ROSMSG --> P
    P --> O
    A --> IK
    A --> SAFETY
    O --> PERCEPTION
    PERCEPTION --> ICTTOK
    ALIGN --> ICTTOK
    CKPT --> ICTTOK
```

网络层只传通用机器人状态、相机数据和时间索引动作。ROS 类型、ICT token、`T_align` 和 checkpoint 配置分别留在两端内部。

