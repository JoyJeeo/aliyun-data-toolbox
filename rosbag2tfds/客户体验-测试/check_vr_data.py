#!/usr/bin/env python3
import sys
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 支持命令行参数指定 shard 和 episode
shard_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# 读取 TFRecord 文件
tfrecord_path = f"/data/delivery_openx/1.0.0/train/delivery_openx-train.tfrecord-{shard_idx:05d}-of-00005"
print(f"读取 shard {shard_idx}, episode {episode_idx}")

# 定义特征描述 (TFDS格式使用 steps/ 前缀)
feature_description = {
    "steps/observation/state/vr_tcp_pose": tf.io.VarLenFeature(tf.float32),
    "steps/action/agent/vr_tcp_input_pose": tf.io.VarLenFeature(tf.float32),
}

# 读取数据 - 直接解析 VarLenFeature 并 reshape 为 (n_steps, 14)
dataset = tf.data.TFRecordDataset(tfrecord_path)
for i, raw_record in enumerate(dataset):
    if i < episode_idx:
        continue
    if i > episode_idx:
        break
    example = tf.io.parse_single_example(raw_record, feature_description)
    vr_tcp_pose_sparse = example["steps/observation/state/vr_tcp_pose"]
    vr_tcp_input_pose_sparse = example["steps/action/agent/vr_tcp_input_pose"]

    vr_tcp_pose_flat = tf.sparse.to_dense(vr_tcp_pose_sparse).numpy()
    vr_tcp_input_pose_flat = tf.sparse.to_dense(vr_tcp_input_pose_sparse).numpy()

    n_steps = len(vr_tcp_pose_flat) // 14
    vr_tcp_pose = vr_tcp_pose_flat.reshape(n_steps, 14)
    vr_tcp_input_pose = vr_tcp_input_pose_flat.reshape(n_steps, 14)

print(f"读取了 {len(vr_tcp_pose)} 个 steps")
print(f"vr_tcp_pose shape: {vr_tcp_pose.shape}")
print(f"vr_tcp_input_pose shape: {vr_tcp_input_pose.shape}")
print(f"\nvr_tcp_pose 非零值数量: {np.count_nonzero(vr_tcp_pose)}")
print(f"vr_tcp_input_pose 非零值数量: {np.count_nonzero(vr_tcp_input_pose)}")

# 计算全部帧的统计信息
print("\n=== vr_tcp_pose 全帧统计 ===")
std = np.std(vr_tcp_pose, axis=0)
mean = np.mean(vr_tcp_pose, axis=0)
min_val = np.min(vr_tcp_pose, axis=0)
max_val = np.max(vr_tcp_pose, axis=0)
print(f"  Left Position  - std: {std[:3]}, range: [{min_val[:3]} ~ {max_val[:3]}]")
print(f"  Left Quaternion - std: {std[3:7]}, range: [{min_val[3:7]} ~ {max_val[3:7]}]")
print(f"  Right Position  - std: {std[7:10]}, range: [{min_val[7:10]} ~ {max_val[7:10]}]")
print(f"  Right Quaternion - std: {std[10:14]}, range: [{min_val[10:14]} ~ {max_val[10:14]}]")

print("\n=== vr_tcp_input_pose 全帧统计 ===")
std_in = np.std(vr_tcp_input_pose, axis=0)
min_in = np.min(vr_tcp_input_pose, axis=0)
max_in = np.max(vr_tcp_input_pose, axis=0)
print(f"  Left Position  - std: {std_in[:3]}, range: [{min_in[:3]} ~ {max_in[:3]}]")
print(f"  Left Quaternion - std: {std_in[3:7]}, range: [{min_in[3:7]} ~ {max_in[3:7]}]")
print(f"  Right Position  - std: {std_in[7:10]}, range: [{min_in[7:10]} ~ {max_in[7:10]}]")
print(f"  Right Quaternion - std: {std_in[10:14]}, range: [{min_in[10:14]} ~ {max_in[10:14]}]")

# 判断是否为"平"的数据
is_flat = np.all(std < 0.001)
is_flat_input = np.all(std_in < 0.001)
print(f"\nvr_tcp_pose 是否为常量: {is_flat}")
print(f"vr_tcp_input_pose 是否为常量: {is_flat_input}")

# 创建图表
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('VR TCP Data Visualization', fontsize=14)
time_axis = np.arange(len(vr_tcp_pose))

# Row 0: Position (obs)
axes[0, 0].plot(time_axis, vr_tcp_pose[:, 0], 'r-', label='X')
axes[0, 0].plot(time_axis, vr_tcp_pose[:, 1], 'g-', label='Y')
axes[0, 0].plot(time_axis, vr_tcp_pose[:, 2], 'b-', label='Z')
axes[0, 0].set_title('vr_tcp_pose - Left Position (obs)')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time_axis, vr_tcp_pose[:, 7], 'r-', label='X')
axes[0, 1].plot(time_axis, vr_tcp_pose[:, 8], 'g-', label='Y')
axes[0, 1].plot(time_axis, vr_tcp_pose[:, 9], 'b-', label='Z')
axes[0, 1].set_title('vr_tcp_pose - Right Position (obs)')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Row 1: Quaternion (obs)
axes[1, 0].plot(time_axis, vr_tcp_pose[:, 3], 'r-', label='qx')
axes[1, 0].plot(time_axis, vr_tcp_pose[:, 4], 'g-', label='qy')
axes[1, 0].plot(time_axis, vr_tcp_pose[:, 5], 'b-', label='qz')
axes[1, 0].plot(time_axis, vr_tcp_pose[:, 6], 'k-', label='qw')
axes[1, 0].set_title('vr_tcp_pose - Left Quaternion (obs)')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(time_axis, vr_tcp_pose[:, 10], 'r-', label='qx')
axes[1, 1].plot(time_axis, vr_tcp_pose[:, 11], 'g-', label='qy')
axes[1, 1].plot(time_axis, vr_tcp_pose[:, 12], 'b-', label='qz')
axes[1, 1].plot(time_axis, vr_tcp_pose[:, 13], 'k-', label='qw')
axes[1, 1].set_title('vr_tcp_pose - Right Quaternion (obs)')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

# Row 2: Position (action)
axes[2, 0].plot(time_axis, vr_tcp_input_pose[:, 0], 'r-', label='X')
axes[2, 0].plot(time_axis, vr_tcp_input_pose[:, 1], 'g-', label='Y')
axes[2, 0].plot(time_axis, vr_tcp_input_pose[:, 2], 'b-', label='Z')
axes[2, 0].set_title('vr_tcp_input_pose - Left Position (action)')
axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(time_axis, vr_tcp_input_pose[:, 7], 'r-', label='X')
axes[2, 1].plot(time_axis, vr_tcp_input_pose[:, 8], 'g-', label='Y')
axes[2, 1].plot(time_axis, vr_tcp_input_pose[:, 9], 'b-', label='Z')
axes[2, 1].set_title('vr_tcp_input_pose - Right Position (action)')
axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

# Row 3: Quaternion (action)
axes[3, 0].plot(time_axis, vr_tcp_input_pose[:, 3], 'r-', label='qx')
axes[3, 0].plot(time_axis, vr_tcp_input_pose[:, 4], 'g-', label='qy')
axes[3, 0].plot(time_axis, vr_tcp_input_pose[:, 5], 'b-', label='qz')
axes[3, 0].plot(time_axis, vr_tcp_input_pose[:, 6], 'k-', label='qw')
axes[3, 0].set_title('vr_tcp_input_pose - Left Quaternion (action)')
axes[3, 0].set_xlabel('Step')
axes[3, 0].legend(); axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].plot(time_axis, vr_tcp_input_pose[:, 10], 'r-', label='qx')
axes[3, 1].plot(time_axis, vr_tcp_input_pose[:, 11], 'g-', label='qy')
axes[3, 1].plot(time_axis, vr_tcp_input_pose[:, 12], 'b-', label='qz')
axes[3, 1].plot(time_axis, vr_tcp_input_pose[:, 13], 'k-', label='qw')
axes[3, 1].set_title('vr_tcp_input_pose - Right Quaternion (action)')
axes[3, 1].set_xlabel('Step')
axes[3, 1].legend(); axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/data/vr_tcp_data_plot.png", dpi=150)
print(f"\n图表已保存到: /data/vr_tcp_data_plot.png")

