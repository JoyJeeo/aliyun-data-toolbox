#!/usr/bin/env python3
"""Check left/right TCP timestamp alignment in TFDS dataset.

Usage:
    python check_tcp_alignment.py /data/delivery_openx/1.0.0
    python check_tcp_alignment.py /data/delivery_openx/1.0.0 --save /data/tcp_alignment.png
"""

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def check_tcp_alignment(dataset_path: str, episode_idx: int = 0, save_path: str = None):
    """Check TCP data alignment in TFDS dataset."""
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        train_dir = Path(dataset_path)
    
    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))
    if not tfrecord_files:
        tfrecord_files = sorted(glob.glob(str(Path(dataset_path) / "**" / "*.tfrecord*"), recursive=True))
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {dataset_path}")

    print(f"Found {len(tfrecord_files)} TFRecord files")

    # Read episode
    all_episodes = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feat = example.features.feature
            
            ep_id = "unknown"
            if "episode_metadata/episode_id" in feat:
                ep_id = feat["episode_metadata/episode_id"].bytes_list.value[0].decode("utf-8")
            
            num_steps = 0
            if "episode_metadata/num_steps" in feat:
                num_steps = int(feat["episode_metadata/num_steps"].int64_list.value[0])
            
            all_episodes.append({"id": ep_id, "num_steps": num_steps, "features": feat})

    print(f"Total episodes: {len(all_episodes)}")
    
    if episode_idx >= len(all_episodes):
        print(f"Episode {episode_idx} not found")
        return

    episode = all_episodes[episode_idx]
    feat = episode["features"]
    num_steps = episode["num_steps"]
    
    print(f"\nChecking episode {episode_idx}: {episode['id']} ({num_steps} steps)")
    print("=" * 70)
    
    # Extract TCP data
    def get_field(field_name, dim):
        if field_name in feat:
            flat = np.array(feat[field_name].float_list.value)
            n = len(flat) // dim
            if n > 0:
                return flat[:n * dim].reshape(n, dim)
        return None
    
    tcp_pos_left = get_field("steps/observation/state/tcp_position_left", 3)
    tcp_pos_right = get_field("steps/observation/state/tcp_position_right", 3)
    tcp_ori_left = get_field("steps/observation/state/tcp_orientation_left", 4)
    tcp_ori_right = get_field("steps/observation/state/tcp_orientation_right", 4)
    
    print(f"\ntcp_position_left:  {tcp_pos_left.shape if tcp_pos_left is not None else 'NOT FOUND'}")
    print(f"tcp_position_right: {tcp_pos_right.shape if tcp_pos_right is not None else 'NOT FOUND'}")
    print(f"tcp_orientation_left:  {tcp_ori_left.shape if tcp_ori_left is not None else 'NOT FOUND'}")
    print(f"tcp_orientation_right: {tcp_ori_right.shape if tcp_ori_right is not None else 'NOT FOUND'}")
    
    if tcp_pos_left is None or tcp_pos_right is None:
        print("TCP data not found")
        return
    
    # Calculate velocity (diff between frames)
    vel_left = np.diff(tcp_pos_left, axis=0)
    vel_right = np.diff(tcp_pos_right, axis=0)
    
    # Calculate correlation between left and right movement
    print("\n" + "=" * 70)
    print("Left/Right TCP Velocity Correlation:")
    print("=" * 70)
    
    for axis, name in enumerate(["X", "Y", "Z"]):
        corr = np.corrcoef(vel_left[:, axis], vel_right[:, axis])[0, 1]
        print(f"  {name}-axis correlation: {corr:.4f}")
    
    # Check for time offset
    print("\n" + "=" * 70)
    print("Cross-correlation (time offset check):")
    print("=" * 70)
    
    for axis, name in enumerate(["X", "Y", "Z"]):
        # Normalize
        l = vel_left[:, axis] - vel_left[:, axis].mean()
        r = vel_right[:, axis] - vel_right[:, axis].mean()
        if np.std(l) > 0 and np.std(r) > 0:
            l = l / np.std(l)
            r = r / np.std(r)
            # Cross-correlation
            xcorr = np.correlate(l, r, mode='full')
            lags = np.arange(-len(l) + 1, len(l))
            best_lag = lags[np.argmax(xcorr)]
            print(f"  {name}-axis: best lag = {best_lag} steps")
    
    # Plot
    if save_path:
        _plot_tcp(tcp_pos_left, tcp_pos_right, episode["id"], save_path, tcp_ori_left, tcp_ori_right)


def _plot_tcp(tcp_left, tcp_right, episode_id, save_path, ori_left=None, ori_right=None):
    """Plot TCP positions and orientations - left hand on left column, right hand on right column."""
    import matplotlib.pyplot as plt

    # 7 rows: X, Y, Z, qx, qy, qz, qw
    fig, axes = plt.subplots(7, 2, figsize=(14, 18))
    fig.suptitle(f"TCP Position & Orientation - {episode_id}", fontsize=14)

    time_steps = np.arange(len(tcp_left))

    # Row labels
    row_labels = ["Pos X (m)", "Pos Y (m)", "Pos Z (m)", "qx", "qy", "qz", "qw"]

    # Left column: Left hand
    axes[0, 0].set_title("Left TCP", fontsize=12, fontweight='bold')
    for i in range(3):
        axes[i, 0].plot(time_steps, tcp_left[:, i], 'b-', linewidth=1.0, alpha=0.9)
        axes[i, 0].set_ylabel(row_labels[i], fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

    if ori_left is not None:
        for i in range(4):
            axes[i+3, 0].plot(time_steps[:len(ori_left)], ori_left[:, i], 'b-', linewidth=1.0, alpha=0.9)
            axes[i+3, 0].set_ylabel(row_labels[i+3], fontsize=10)
            axes[i+3, 0].grid(True, alpha=0.3)

    # Right column: Right hand
    axes[0, 1].set_title("Right TCP", fontsize=12, fontweight='bold')
    for i in range(3):
        axes[i, 1].plot(time_steps, tcp_right[:, i], 'r-', linewidth=1.0, alpha=0.9)
        axes[i, 1].set_ylabel(row_labels[i], fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)

    if ori_right is not None:
        for i in range(4):
            axes[i+3, 1].plot(time_steps[:len(ori_right)], ori_right[:, i], 'r-', linewidth=1.0, alpha=0.9)
            axes[i+3, 1].set_ylabel(row_labels[i+3], fontsize=10)
            axes[i+3, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Sample", fontsize=11)
    axes[-1, 1].set_xlabel("Sample", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Check TCP alignment")
    parser.add_argument("dataset_path", type=str, help="TFDS dataset path")
    parser.add_argument("episode_idx", type=int, nargs="?", default=0, help="Episode index")
    parser.add_argument("--save", type=str, default=None, help="Save plot")
    args = parser.parse_args()

    check_tcp_alignment(args.dataset_path, args.episode_idx, args.save)


if __name__ == "__main__":
    main()

