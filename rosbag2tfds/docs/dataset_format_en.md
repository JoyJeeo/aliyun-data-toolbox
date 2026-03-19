# Delivery OpenX Dataset Format

This document describes the data format after converting ROS bags to TFDS (TensorFlow Datasets).

## Dataset Structure

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

Each TFRecord file contains one episode (corresponding to one ROS bag).

---

## Episode Structure

Each episode contains:
- `steps`: Sequence of time steps (each step contains observation and action)
- `episode_metadata`: Episode-level metadata

---

## Step Structure

### Observation

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `image` | (H, W, 3) | uint8 | Primary camera RGB image (JPEG) |
| `image_aux_1` | (H, W, 3) | uint8 | Auxiliary camera 1 RGB image |
| `image_aux_2` | (H, W, ?) | uint8/uint16 | Auxiliary camera 2 (may be depth) |
| `state/joint_position` | (28,) | float32 | Joint positions (rad) |
| `state/joint_velocity` | (28,) | float32 | Joint velocities (rad/s) |
| `state/joint_torque` | (28,) | float32 | Joint torques (N·m) |
| `state/eef_position` | (12,) | float32 | End-effector position/pose |
| `state/eef_velocity` | (12,) | float32 | End-effector velocity |
| `state/eef_effort` | (12,) | float32 | End-effector force/torque |
| `state/tcp_position_left` | (3,) | float32 | Left arm TCP position [x,y,z] (m) |
| `state/tcp_position_right` | (3,) | float32 | Right arm TCP position [x,y,z] (m) |
| `state/tcp_orientation_left` | (4,) | float32 | Left arm TCP quaternion [x,y,z,w] |
| `state/tcp_orientation_right` | (4,) | float32 | Right arm TCP quaternion [x,y,z,w] |
| `timestamp` | () | int64 | ROS timestamp (ns) |
| `natural_language_instruction` | - | string | Episode-level task instruction |
| `subtask_language_instruction` | - | string | Current subtask instruction |
| `camera_extrinsics_json` | - | string | Camera extrinsics (JSON) |

### Action

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| **Open-X Format (RT-1/RT-1-X)** |
| `world_vector` | (3,) | float32 | TCP position delta [Δx,Δy,Δz] (m) |
| `rotation_delta` | (3,) | float32 | TCP rotation delta [Δroll,Δpitch,Δyaw] (rad) |
| `world_vector_left` | (3,) | float32 | Left arm TCP position delta |
| `world_vector_right` | (3,) | float32 | Right arm TCP position delta |
| `rotation_delta_left` | (3,) | float32 | Left arm TCP rotation delta |
| `rotation_delta_right` | (3,) | float32 | Right arm TCP rotation delta |
| `gripper_closedness_action` | (1,) | float32 | Gripper action (placeholder, always 0) |
| `terminate_episode` | () | float32 | Termination flag (1.0=end, 0.0=continue) |
| **Raw Commands (Low-level Control)** |
| `agent/joint_position` | (28,) | float32 | Commanded joint positions (rad) |
| `agent/joint_velocity` | (28,) | float32 | Commanded joint velocities (rad/s) |
| `agent/joint_torque` | (28,) | float32 | Commanded joint torques (N·m) |
| `agent/eef_position` | (12,) | float32 | Commanded end-effector position |
| `agent/eef_velocity` | (12,) | float32 | Commanded end-effector velocity |

### Control Flags

| Field | Type | Description |
|-------|------|-------------|
| `reward` | float32 | Reward (always 0) |
| `discount` | float32 | Discount (always 1) |
| `is_first` | bool | True on first step of episode |
| `is_last` | bool | True on last step of episode |
| `is_terminal` | bool | True when terminated (same as is_last) |

---

## Episode Metadata

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique episode identifier (from bag filename) |
| `bag_path` | string | Original ROS bag path |
| `eef_type` | string | End-effector type (`dexhand` or `leju_claw`) |
| `eef_dim` | int32 | End-effector dimension (2 or 12) |
| `timeline` | string | Timeline alignment reference (`camera`) |
| `num_steps` | int32 | Number of steps in episode |
| `camera_info_json` | string | Camera intrinsics (JSON) |
| `camera_intrinsics_json` | string | Camera intrinsic matrix (JSON) |
| `camera_extrinsics_json` | string | Camera extrinsics (JSON) |
| `sidecar_json` | string | Original sidecar JSON |
| `joint_names_json` | string | Joint names list (JSON) |

---

## Delta Computation Formula

**Following Open-X standard: `action[t] = state[t+1] - state[t]`**

### world_vector (Position Delta)
```python
world_vector[t] = tcp_position[t+1] - tcp_position[t]
```

### rotation_delta (Rotation Delta)
```python
# Quaternion relative rotation
q_delta = q[t+1] * q[t]^(-1)

# Convert to Euler angles [roll, pitch, yaw]
rotation_delta[t] = quaternion_to_euler(q_delta)
```

---

## Joint Order (28 DOF)

```
[0-6]   : Left arm  (7 DOF) - left_arm_joint_1 ~ left_arm_joint_7
[7-13]  : Right arm (7 DOF) - right_arm_joint_1 ~ right_arm_joint_7
[14-19] : Left hand (6 DOF) - left_hand joints
[20-25] : Right hand (6 DOF) - right_hand joints
[26-27] : Head (2 DOF) - head_yaw, head_pitch
```

---

## Usage Example

```python
import tensorflow_datasets as tfds

# Load dataset
ds = tfds.load('delivery_openx', data_dir='/path/to/dataset')

# Iterate over episodes
for episode in ds['train']:
    steps = episode['steps']
    metadata = episode['episode_metadata']
    
    for step in steps:
        obs = step['observation']
        action = step['action']
        
        image = obs['image']                         # RGB image
        tcp_pos = obs['state']['tcp_position_left']  # TCP position
        world_vec = action['world_vector']           # Position delta
        rot_delta = action['rotation_delta']         # Rotation delta
```

