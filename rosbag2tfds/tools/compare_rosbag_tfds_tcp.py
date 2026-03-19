#!/usr/bin/env python3
"""Compare TCP data between rosbag and TFDS to verify conversion correctness.

Usage:
    python compare_rosbag_tfds_tcp.py /data/xxx.bag /data/delivery_openx/1.0.0 --save /data/compare.png
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import rosbag
import tensorflow as tf
from scipy.spatial.transform import Rotation

from tf_reader import TFBuffer


def compute_tcp_from_rosbag(bag_path: str, base_frame: str = "base_link",
                             tcp_frame_left: str = "zarm_l7_link",
                             tcp_frame_right: str = "zarm_r7_link"):
    """Compute TCP poses (position + quaternion) directly from rosbag TF data."""
    print(f"Reading TF from rosbag: {bag_path}")

    bag = rosbag.Bag(bag_path)
    tf_buffer = TFBuffer()

    # Read all TF
    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for tf_msg in msg.transforms:
            ts_ns = tf_msg.header.stamp.secs * 1_000_000_000 + tf_msg.header.stamp.nsecs
            tf_buffer.add_transform(tf_msg, ts_ns)

    timestamps = tf_buffer.get_timestamps()
    print(f"  Found {len(timestamps)} TF timestamps")

    # Compute TCP for each timestamp
    pos_left = []
    pos_right = []
    quat_left = []
    quat_right = []
    valid_ts = []

    for ts in timestamps:
        left = tf_buffer.get_transform(base_frame, tcp_frame_left, ts)
        right = tf_buffer.get_transform(base_frame, tcp_frame_right, ts)

        if left is not None and right is not None:
            pos_left.append(left[0])
            pos_right.append(right[0])
            quat_left.append(left[1])
            quat_right.append(right[1])
            valid_ts.append(ts)

    print(f"  Valid TCP samples: {len(valid_ts)}")
    return (np.array(valid_ts),
            np.array(pos_left), np.array(pos_right),
            np.array(quat_left), np.array(quat_right))


def read_tcp_from_tfds(dataset_path: str, episode_idx: int = 0):
    """Read TCP data (position + quaternion) from TFDS."""
    train_dir = Path(dataset_path) / "train"
    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found")

    # Read episode
    all_episodes = []
    for f in tfrecord_files:
        dataset = tf.data.TFRecordDataset(f)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            all_episodes.append(example.features.feature)

    if episode_idx >= len(all_episodes):
        raise ValueError(f"Episode {episode_idx} not found")

    feat = all_episodes[episode_idx]

    def get_field(name, dim):
        if name in feat:
            flat = np.array(feat[name].float_list.value)
            n = len(flat) // dim
            return flat[:n*dim].reshape(n, dim) if n > 0 else None
        return None

    pos_left = get_field("steps/observation/state/tcp_position_left", 3)
    pos_right = get_field("steps/observation/state/tcp_position_right", 3)
    quat_left = get_field("steps/observation/state/tcp_orientation_left", 4)
    quat_right = get_field("steps/observation/state/tcp_orientation_right", 4)

    # Get timestamps
    timestamps = None
    if "steps/observation/timestamp" in feat:
        timestamps = np.array(feat["steps/observation/timestamp"].int64_list.value)

    print(f"TFDS episode {episode_idx}: {len(pos_left) if pos_left is not None else 0} steps")
    return timestamps, pos_left, pos_right, quat_left, quat_right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to rosbag file")
    parser.add_argument("tfds_path", help="Path to TFDS dataset")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # Read from rosbag (now returns position and quaternion)
    bag_ts, bag_pos_left, bag_pos_right, bag_quat_left, bag_quat_right = compute_tcp_from_rosbag(args.bag_path)

    # Read from TFDS
    tfds_ts, tfds_pos_left, tfds_pos_right, tfds_quat_left, tfds_quat_right = read_tcp_from_tfds(args.tfds_path, args.episode)

    print(f"\nComparison:")
    print(f"  Rosbag: {len(bag_pos_left)} samples")
    print(f"  TFDS:   {len(tfds_pos_left)} samples")

    # Match by timestamps
    if tfds_ts is not None and len(tfds_ts) > 0:
        print(f"\n  TFDS timestamp range: {tfds_ts[0]} - {tfds_ts[-1]}")
        print(f"  Bag  timestamp range: {bag_ts[0]} - {bag_ts[-1]}")

        # For each TFDS timestamp, find nearest bag timestamp
        matched_pos_left = []
        matched_pos_right = []
        matched_quat_left = []
        matched_quat_right = []
        for ts in tfds_ts:
            idx = np.argmin(np.abs(bag_ts - ts))
            matched_pos_left.append(bag_pos_left[idx])
            matched_pos_right.append(bag_pos_right[idx])
            matched_quat_left.append(bag_quat_left[idx])
            matched_quat_right.append(bag_quat_right[idx])
        bag_pos_left = np.array(matched_pos_left)
        bag_pos_right = np.array(matched_pos_right)
        bag_quat_left = np.array(matched_quat_left)
        bag_quat_right = np.array(matched_quat_right)
        print(f"  Matched {len(bag_pos_left)} samples by timestamp")

    # Compare positions
    n = min(len(bag_pos_left), len(tfds_pos_left))
    diff_pos_left = np.linalg.norm(bag_pos_left[:n] - tfds_pos_left[:n], axis=1)
    diff_pos_right = np.linalg.norm(bag_pos_right[:n] - tfds_pos_right[:n], axis=1)

    print(f"\n  Left TCP pos diff:  mean={diff_pos_left.mean():.6f}m, max={diff_pos_left.max():.6f}m")
    print(f"  Right TCP pos diff: mean={diff_pos_right.mean():.6f}m, max={diff_pos_right.max():.6f}m")

    # Compare quaternions (if available)
    if tfds_quat_left is not None and tfds_quat_right is not None:
        diff_quat_left = np.abs(bag_quat_left[:n] - tfds_quat_left[:n])
        diff_quat_right = np.abs(bag_quat_right[:n] - tfds_quat_right[:n])
        print(f"  Left TCP quat diff:  mean={diff_quat_left.mean():.6f}, max={diff_quat_left.max():.6f}")
        print(f"  Right TCP quat diff: mean={diff_quat_right.mean():.6f}, max={diff_quat_right.max():.6f}")

    if args.save:
        import matplotlib.pyplot as plt

        # 7 rows: 3 for position (X,Y,Z) + 4 for quaternion (qx,qy,qz,qw)
        fig, axes = plt.subplots(7, 2, figsize=(16, 20))
        fig.suptitle("Rosbag(blue) vs TFDS(red) TCP Comparison", fontsize=14)

        # Position plots (rows 0-2)
        pos_labels = ["X", "Y", "Z"]
        for i, label in enumerate(pos_labels):
            axes[i, 0].plot(bag_pos_left[:n, i], 'b-', alpha=0.7, linewidth=0.8, label='rosbag')
            axes[i, 0].plot(tfds_pos_left[:n, i], 'r--', alpha=0.7, linewidth=0.8, label='tfds')
            axes[i, 0].set_ylabel(f"Pos {label} (m)")
            axes[i, 0].legend(loc='upper right', fontsize=8)
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].set_title("Left TCP")

            axes[i, 1].plot(bag_pos_right[:n, i], 'b-', alpha=0.7, linewidth=0.8, label='rosbag')
            axes[i, 1].plot(tfds_pos_right[:n, i], 'r--', alpha=0.7, linewidth=0.8, label='tfds')
            axes[i, 1].set_ylabel(f"Pos {label} (m)")
            axes[i, 1].legend(loc='upper right', fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 1].set_title("Right TCP")

        # Quaternion plots (rows 3-6)
        quat_labels = ["qx", "qy", "qz", "qw"]
        if tfds_quat_left is not None and tfds_quat_right is not None:
            for i, label in enumerate(quat_labels):
                row = i + 3
                axes[row, 0].plot(bag_quat_left[:n, i], 'b-', alpha=0.7, linewidth=0.8, label='rosbag')
                axes[row, 0].plot(tfds_quat_left[:n, i], 'r--', alpha=0.7, linewidth=0.8, label='tfds')
                axes[row, 0].set_ylabel(label)
                axes[row, 0].legend(loc='upper right', fontsize=8)
                axes[row, 0].grid(True, alpha=0.3)

                axes[row, 1].plot(bag_quat_right[:n, i], 'b-', alpha=0.7, linewidth=0.8, label='rosbag')
                axes[row, 1].plot(tfds_quat_right[:n, i], 'r--', alpha=0.7, linewidth=0.8, label='tfds')
                axes[row, 1].set_ylabel(label)
                axes[row, 1].legend(loc='upper right', fontsize=8)
                axes[row, 1].grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("Sample")
        axes[-1, 1].set_xlabel("Sample")
        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()

