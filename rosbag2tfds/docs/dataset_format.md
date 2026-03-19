# Delivery OpenX Dataset Format

本文档描述 ROS bag 转换为 TFDS (TensorFlow Datasets) 后的数据格式。

## 数据集结构

```
delivery_openx/1.0.0/
├── train/
│   ├── delivery_openx-train.tfrecord-00000-of-XXXXX
│   ├── delivery_openx-train.tfrecord-00001-of-XXXXX
│   └── ...
├── dataset_info.json
├── features.json
└── metadata.json
```

每个 TFRecord 文件包含一个 episode（对应一个 ROS bag）。

---

## Episode 结构

每个 episode 包含：
- `steps`: 时间步序列（每步包含 observation 和 action）
- `episode_metadata`: episode 级别的元数据

---

## Step 结构

### Observation（观测）

| 字段 | 形状 | 类型 | 描述 |
|------|------|------|------|
| `image` | (H, W, 3) | uint8 | 主相机 RGB 图像 (JPEG) |
| `image_aux_1` | (H, W, 3) | uint8 | 辅助相机 1 RGB 图像 |
| `image_aux_2` | (H, W, ?) | uint8/uint16 | 辅助相机 2 (可能是深度图) |
| `state/joint_position` | (28,) | float32 | 关节位置 (rad) |
| `state/joint_velocity` | (28,) | float32 | 关节速度 (rad/s) |
| `state/joint_torque` | (28,) | float32 | 关节力矩 (N·m) |
| `state/eef_position` | (12,) | float32 | 末端执行器位置/姿态 |
| `state/eef_velocity` | (12,) | float32 | 末端执行器速度 |
| `state/eef_effort` | (12,) | float32 | 末端执行器力/力矩 |
| `state/tcp_position_left` | (3,) | float32 | 左臂 TCP 位置 [x,y,z] (m) |
| `state/tcp_position_right` | (3,) | float32 | 右臂 TCP 位置 [x,y,z] (m) |
| `state/tcp_orientation_left` | (4,) | float32 | 左臂 TCP 四元数 [x,y,z,w] |
| `state/tcp_orientation_right` | (4,) | float32 | 右臂 TCP 四元数 [x,y,z,w] |
| `timestamp` | () | int64 | ROS 时间戳 (ns) |
| `natural_language_instruction` | - | string | Episode 级任务指令 |
| `subtask_language_instruction` | - | string | 当前子任务指令 |
| `camera_extrinsics_json` | - | string | 相机外参 (JSON) |

### Action（动作）

| 字段 | 形状 | 类型 | 描述 |
|------|------|------|------|
| **Open-X 格式 (RT-1/RT-1-X)** |
| `world_vector` | (3,) | float32 | TCP 位置增量 [Δx,Δy,Δz] (m) |
| `rotation_delta` | (3,) | float32 | TCP 旋转增量 [Δroll,Δpitch,Δyaw] (rad) |
| `world_vector_left` | (3,) | float32 | 左臂 TCP 位置增量 |
| `world_vector_right` | (3,) | float32 | 右臂 TCP 位置增量 |
| `rotation_delta_left` | (3,) | float32 | 左臂 TCP 旋转增量 |
| `rotation_delta_right` | (3,) | float32 | 右臂 TCP 旋转增量 |
| `gripper_closedness_action` | (1,) | float32 | 夹爪开合 (占位符，始终为 0) |
| `terminate_episode` | () | float32 | 终止标志 (1.0=结束, 0.0=继续) |
| **原始指令 (低级控制)** |
| `agent/joint_position` | (28,) | float32 | 指令关节位置 (rad) |
| `agent/joint_velocity` | (28,) | float32 | 指令关节速度 (rad/s) |
| `agent/joint_torque` | (28,) | float32 | 指令关节力矩 (N·m) |
| `agent/eef_position` | (12,) | float32 | 指令末端执行器位置 |
| `agent/eef_velocity` | (12,) | float32 | 指令末端执行器速度 |

### 控制标志

| 字段 | 类型 | 描述 |
|------|------|------|
| `reward` | float32 | 奖励 (始终为 0) |
| `discount` | float32 | 折扣 (始终为 1) |
| `is_first` | bool | 是否为 episode 第一步 |
| `is_last` | bool | 是否为 episode 最后一步 |
| `is_terminal` | bool | 是否终止 (同 is_last) |

---

## Episode Metadata

| 字段 | 类型 | 描述 |
|------|------|------|
| `episode_id` | string | Episode 唯一标识 (来自 bag 文件名) |
| `bag_path` | string | 原始 ROS bag 路径 |
| `eef_type` | string | 末端执行器类型 (`dexhand` 或 `leju_claw`) |
| `eef_dim` | int32 | 末端执行器维度 (2 或 12) |
| `timeline` | string | 时间线对齐参考 (`camera`) |
| `num_steps` | int32 | Episode 步数 |
| `camera_info_json` | string | 相机内参 (JSON) |
| `camera_intrinsics_json` | string | 相机内参矩阵 (JSON) |
| `camera_extrinsics_json` | string | 相机外参 (JSON) |
| `sidecar_json` | string | 原始 sidecar JSON |
| `joint_names_json` | string | 关节名称列表 (JSON) |

---

## Delta 计算公式

**遵循 Open-X 标准：`action[t] = state[t+1] - state[t]`**

### world_vector (位置增量)
```python
world_vector[t] = tcp_position[t+1] - tcp_position[t]
```

### rotation_delta (旋转增量)
```python
# 四元数相对旋转
q_delta = q[t+1] * q[t]^(-1)

# 转换为欧拉角 [roll, pitch, yaw]
rotation_delta[t] = quaternion_to_euler(q_delta)
```

---

## 关节顺序 (28 DOF)

```
[0-6]   : 左臂 (7 DOF) - left_arm_joint_1 ~ left_arm_joint_7
[7-13]  : 右臂 (7 DOF) - right_arm_joint_1 ~ right_arm_joint_7
[14-19] : 左手 (6 DOF) - left_hand joints
[20-25] : 右手 (6 DOF) - right_hand joints
[26-27] : 头部 (2 DOF) - head_yaw, head_pitch
```

---

## 使用示例

```python
import tensorflow_datasets as tfds

# 加载数据集
ds = tfds.load('delivery_openx', data_dir='/path/to/dataset')

# 遍历 episodes
for episode in ds['train']:
    steps = episode['steps']
    metadata = episode['episode_metadata']
    
    for step in steps:
        obs = step['observation']
        action = step['action']
        
        image = obs['image']                    # RGB 图像
        tcp_pos = obs['state']['tcp_position_left']  # TCP 位置
        world_vec = action['world_vector']      # 位置增量
        rot_delta = action['rotation_delta']    # 旋转增量
```

