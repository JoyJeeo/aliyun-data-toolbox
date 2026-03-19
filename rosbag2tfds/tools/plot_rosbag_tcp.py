#!/usr/bin/env python3
"""Plot all arm link positions from rosbag TF data.

Usage:
    python plot_rosbag_tcp.py /path/to/xxx.bag --save /path/to/output.png
    python plot_rosbag_tcp.py /path/to/xxx.bag  # Display interactively
    python plot_rosbag_tcp.py /path/to/xxx.bag --all-links  # Show all arm links
"""

import argparse
from pathlib import Path

import numpy as np
import rosbag

from tf_reader import TFBuffer


def compute_link_poses(bag_path: str, base_frame: str, link_frames: list):
    """Compute link positions and orientations from rosbag TF data."""
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

    # Compute poses for each link
    positions = {frame: [] for frame in link_frames}
    quaternions = {frame: [] for frame in link_frames}
    valid_ts = []

    for ts in timestamps:
        all_valid = True
        poses = {}
        for frame in link_frames:
            transform = tf_buffer.get_transform(base_frame, frame, ts)
            if transform is not None:
                poses[frame] = transform  # (position, quaternion)
            else:
                all_valid = False
                break

        if all_valid:
            for frame in link_frames:
                positions[frame].append(poses[frame][0])
                quaternions[frame].append(poses[frame][1])
            valid_ts.append(ts)

    print(f"  Valid samples: {len(valid_ts)}")
    bag.close()

    # Convert to numpy arrays
    for frame in link_frames:
        positions[frame] = np.array(positions[frame])
        quaternions[frame] = np.array(quaternions[frame])

    return np.array(valid_ts), positions, quaternions


def main():
    parser = argparse.ArgumentParser(description="Plot rosbag link positions and orientations")
    parser.add_argument("bag_path", help="Path to rosbag file")
    parser.add_argument("--base-frame", default="base_link", help="Base frame")
    parser.add_argument("--all-links", action="store_true", help="Show all arm links (l1-l7, r1-r7)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    args = parser.parse_args()

    # Define link frames - only TCP (l7, r7) for now
    left_links = ["zarm_l7_link"]
    right_links = ["zarm_r7_link"]
    all_links = left_links + right_links

    # Read poses from rosbag
    ts, positions, quaternions = compute_link_poses(args.bag_path, args.base_frame, all_links)

    if len(ts) == 0:
        print("No data found!")
        return

    # Convert timestamps to seconds from start
    ts_sec = (ts - ts[0]) / 1e9

    print(f"\nData summary:")
    print(f"  Duration: {ts_sec[-1]:.2f}s")
    print(f"  Samples: {len(ts)}")

    # Plot
    import matplotlib.pyplot as plt

    # Create figure with 7 rows: 3 for position (X,Y,Z) + 4 for quaternion (x,y,z,w)
    fig, axes = plt.subplots(7, 2, figsize=(16, 20))
    fig.suptitle(f"Rosbag TCP Poses (base_link frame)\n{Path(args.bag_path).name}", fontsize=12)

    left_link = left_links[0]
    right_link = right_links[0]

    # Position plots (rows 0-2)
    pos_labels = ['X', 'Y', 'Z']
    for i, label in enumerate(pos_labels):
        axes[i, 0].plot(ts_sec, positions[left_link][:, i], 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel(f"Pos {label} (m)")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Left TCP (zarm_l7_link)")

        axes[i, 1].plot(ts_sec, positions[right_link][:, i], 'r-', linewidth=0.8)
        axes[i, 1].set_ylabel(f"Pos {label} (m)")
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("Right TCP (zarm_r7_link)")

    # Quaternion plots (rows 3-6)
    quat_labels = ['qx', 'qy', 'qz', 'qw']
    for i, label in enumerate(quat_labels):
        row = i + 3
        axes[row, 0].plot(ts_sec, quaternions[left_link][:, i], 'b-', linewidth=0.8)
        axes[row, 0].set_ylabel(label)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(ts_sec, quaternions[right_link][:, i], 'r-', linewidth=0.8)
        axes[row, 1].set_ylabel(label)
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nSaved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

