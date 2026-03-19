#!/usr/bin/env python3
"""Compare joint data between rosbag and converted TFDS."""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import rosbag
import tensorflow_datasets as tfds

# Joint name order (28 joints)
JOINT_NAMES = [
    # Left arm (7)
    "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
    "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
    # Right arm (7)
    "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
    "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    # Waist (3)
    "joint_waist_pitch", "joint_waist_roll", "joint_waist_yaw",
    # Head (2)
    "joint_head_yaw", "joint_head_pitch",
    # Left leg (6)
    "joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3",
    "joint_left_leg_4", "joint_left_leg_5", "joint_left_leg_6",
    # Right leg (6) - currently zeros
    "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3",
    "joint_right_leg_4", "joint_right_leg_5", "joint_right_leg_6",
]


def read_joints_from_bag(bag_path: str, topic: str = "/sensors_data_raw"):
    """Read joint positions from rosbag."""
    print(f"Reading joint data from rosbag: {bag_path}")
    bag = rosbag.Bag(bag_path)
    
    timestamps = []
    joint_positions = []
    
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        # Use message timestamp if available
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            ts_ns = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        else:
            ts_ns = t.secs * 1_000_000_000 + t.nsecs
        
        if hasattr(msg, 'joint_data') and hasattr(msg.joint_data, 'joint_q'):
            timestamps.append(ts_ns)
            joint_positions.append(list(msg.joint_data.joint_q))
    
    bag.close()
    print(f"  Found {len(timestamps)} samples")
    return np.array(timestamps), np.array(joint_positions)


def read_joints_from_tfds(tfds_path: str):
    """Read joint positions from TFDS dataset."""
    import glob
    import tensorflow as tf

    print(f"Reading joint data from TFDS: {tfds_path}")

    train_dir = Path(tfds_path) / "train"
    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {train_dir}")

    timestamps = []
    joint_positions = []

    for f in tfrecord_files:
        dataset = tf.data.TFRecordDataset(f)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feat = example.features.feature

            # Get timestamps
            if "steps/observation/timestamp" in feat:
                ts_list = list(feat["steps/observation/timestamp"].int64_list.value)
                timestamps.extend(ts_list)

            # Get joint positions (flat array, reshape to Nx28)
            if "steps/observation/state/joint_position" in feat:
                flat = np.array(feat["steps/observation/state/joint_position"].float_list.value)
                n = len(flat) // 28
                if n > 0:
                    jp = flat[:n*28].reshape(n, 28)
                    joint_positions.extend(jp)

    print(f"  Found {len(timestamps)} samples")
    return np.array(timestamps), np.array(joint_positions)


def main():
    parser = argparse.ArgumentParser(description="Compare joint data between rosbag and TFDS")
    parser.add_argument("bag_path", help="Path to rosbag file")
    parser.add_argument("tfds_path", help="Path to TFDS dataset directory")
    parser.add_argument("--topic", default="/sensors_data_raw", help="Joint state topic")
    parser.add_argument("--joints", default="0,1,2,3,4,5,6", help="Joint indices to plot (comma-separated)")
    parser.add_argument("--save", help="Save plot to file")
    args = parser.parse_args()
    
    # Read data
    bag_ts, bag_joints = read_joints_from_bag(args.bag_path, args.topic)
    tfds_ts, tfds_joints = read_joints_from_tfds(args.tfds_path)
    
    if len(bag_joints) == 0 or len(tfds_joints) == 0:
        print("Error: No joint data found!")
        sys.exit(1)
    
    # Match by timestamp
    print(f"\nMatching by timestamp...")
    print(f"  Bag  ts range: {bag_ts[0]} - {bag_ts[-1]}")
    print(f"  TFDS ts range: {tfds_ts[0]} - {tfds_ts[-1]}")

    # Find overlapping range
    overlap_start = max(bag_ts[0], tfds_ts[0])
    overlap_end = min(bag_ts[-1], tfds_ts[-1])
    print(f"  Overlap range: {overlap_start} - {overlap_end}")

    # Filter TFDS samples within overlap range
    valid_mask = (tfds_ts >= overlap_start) & (tfds_ts <= overlap_end)
    tfds_ts_valid = tfds_ts[valid_mask]
    tfds_joints_valid = tfds_joints[valid_mask]
    print(f"  TFDS samples in overlap: {len(tfds_ts_valid)} / {len(tfds_ts)}")

    matched_bag_joints = []
    matched_indices = []
    for ts in tfds_ts_valid:
        idx = np.argmin(np.abs(bag_ts - ts))
        matched_bag_joints.append(bag_joints[idx])
        matched_indices.append(idx)
    matched_bag_joints = np.array(matched_bag_joints)

    # Check matched indices distribution
    if len(matched_indices) > 0:
        print(f"  Matched bag indices: min={min(matched_indices)}, max={max(matched_indices)}")
        print(f"  Bag total samples: {len(bag_ts)}")

    tfds_joints = tfds_joints_valid
    print(f"  Final matched samples: {len(matched_bag_joints)}")
    
    # Parse joint indices
    joint_indices = [int(j) for j in args.joints.split(",")]
    n_joints = len(joint_indices)
    
    # Compute diff
    diff = np.abs(matched_bag_joints - tfds_joints)
    print(f"\nMax absolute difference per joint:")
    for i in joint_indices:
        print(f"  {JOINT_NAMES[i]}: {diff[:, i].max():.6f} rad")
    
    # Plot
    n_cols = 2
    n_rows = (n_joints + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for plot_idx, joint_idx in enumerate(joint_indices):
        ax = axes[plot_idx]
        ax.plot(matched_bag_joints[:, joint_idx], 'b-', label='rosbag', linewidth=1.5)
        ax.plot(tfds_joints[:, joint_idx], 'r--', label='tfds', linewidth=1.5)
        ax.set_ylabel(JOINT_NAMES[joint_idx])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(len(joint_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Rosbag (blue) vs TFDS (red) Joint Position Comparison", fontsize=12)
    plt.xlabel("Sample")
    plt.tight_layout()
    
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

