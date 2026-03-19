#!/usr/bin/env python3
"""Check joint data in TFDS dataset.

Usage:
    python check_tfds_joints.py /path/to/dataset/delivery_openx/1.0.0
    python check_tfds_joints.py /path/to/dataset/delivery_openx/1.0.0 1  # specify episode index
    python check_tfds_joints.py /path/to/dataset/delivery_openx/1.0.0 --plot  # show plot
    python check_tfds_joints.py /path/to/dataset/delivery_openx/1.0.0 --save joints.png
"""

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

# 28 DOF joint names
JOINT_NAMES = [
    # Left leg (0-5)
    "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
    # Right leg (6-11)
    "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
    # Left arm (12-18)
    "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
    # Right arm (19-25)
    "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
    # Head (26-27)
    "head_yaw", "head_pitch",
]

JOINT_GROUPS = [
    ("Left Leg [0-5]", range(0, 6)),
    ("Right Leg [6-11]", range(6, 12)),
    ("Left Arm [12-18]", range(12, 19)),
    ("Right Arm [19-25]", range(19, 26)),
    ("Head [26-27]", range(26, 28)),
]


def check_tfds_joints(dataset_path: str, episode_idx: int = 0, plot: bool = False, save_path: str = None,
                      save_per_joint_vel: str = None, save_per_joint_pos: str = None):
    """Check joint data in TFDS dataset."""
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        train_dir = Path(dataset_path)

    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))
    if not tfrecord_files:
        tfrecord_files = sorted(glob.glob(str(Path(dataset_path) / "**" / "*.tfrecord*"), recursive=True))

    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {dataset_path}")

    print(f"Found {len(tfrecord_files)} TFRecord files")

    # Read all episodes (each TFRecord record = one episode with flattened steps)
    all_episodes = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            feat = example.features.feature

            # Get episode_id
            ep_id = "unknown"
            if "episode_metadata/episode_id" in feat:
                ep_id = feat["episode_metadata/episode_id"].bytes_list.value[0].decode("utf-8")

            # Get num_steps
            num_steps = 0
            if "episode_metadata/num_steps" in feat:
                num_steps = int(feat["episode_metadata/num_steps"].int64_list.value[0])

            all_episodes.append({"id": ep_id, "num_steps": num_steps, "features": feat})

    print(f"Total episodes: {len(all_episodes)}")

    # Show all episodes
    print("\n" + "=" * 70)
    print("Episode Summary:")
    print("=" * 70)
    for i, ep in enumerate(all_episodes):
        print(f"  [{i}] {ep['id']}: {ep['num_steps']} steps")

    if episode_idx >= len(all_episodes):
        print(f"Episode {episode_idx} not found")
        return

    episode = all_episodes[episode_idx]
    feat = episode["features"]
    num_steps = episode["num_steps"]

    print(f"\n" + "=" * 70)
    print(f"Checking episode {episode_idx}: {episode['id']} ({num_steps} steps)")
    print("=" * 70)

    # Extract joint data (flattened: num_steps * 28 values)
    joint_fields = [
        ("steps/observation/state/joint_position", 28, "obs_pos"),
        ("steps/observation/state/joint_velocity", 28, "obs_vel"),
        ("steps/observation/state/joint_torque", 28, "obs_tau"),
        ("steps/action/agent/joint_position", 28, "act_pos"),
        ("steps/action/agent/joint_velocity", 28, "act_vel"),
        ("steps/action/agent/joint_torque", 28, "act_tau"),
    ]

    data = {}
    print("\nJoint Data Analysis:")

    for field, dim, key in joint_fields:
        if field in feat:
            flat_values = np.array(feat[field].float_list.value)
            n_steps = len(flat_values) // dim
            if n_steps > 0:
                values = flat_values[:n_steps * dim].reshape(n_steps, dim)
                data[key] = values

                # Check which dimensions have non-zero data
                non_zero_dims = [d for d in range(dim) if np.any(values[:, d] != 0)]
                zero_dims = [d for d in range(dim) if not np.any(values[:, d] != 0)]

                print(f"\n  {field}:")
                print(f"    Shape: {values.shape}")
                print(f"    Non-zero dims: {len(non_zero_dims)}/28")
                if zero_dims:
                    print(f"    All-zero dims: {zero_dims[:10]}{'...' if len(zero_dims) > 10 else ''}")
                print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
            else:
                print(f"\n  {field}: empty data")
        else:
            print(f"\n  {field}: NOT FOUND")
    
    # Per-group analysis
    print("\n" + "=" * 70)
    print("Per-Group Analysis (observation/state/joint_position):")
    print("=" * 70)

    if "obs_pos" in data:
        obs_pos = data["obs_pos"]
        for group_name, indices in JOINT_GROUPS:
            group_data = obs_pos[:, list(indices)]
            non_zero = np.any(group_data != 0, axis=0).sum()
            print(f"\n  {group_name}:")
            print(f"    Active joints: {non_zero}/{len(indices)}")
            print(f"    Range: [{group_data.min():.4f}, {group_data.max():.4f}]")

            # Show per-joint stats
            for i, j in enumerate(indices):
                jdata = obs_pos[:, j]
                status = "Y" if np.any(jdata != 0) else "N"
                print(f"      [{j:2d}] {JOINT_NAMES[j]:12s}: {status} min={jdata.min():8.4f}, max={jdata.max():8.4f}, std={jdata.std():.4f}")

    # Action vs Observation comparison
    print("\n" + "=" * 70)
    print("Action vs Observation Comparison:")
    print("=" * 70)

    if "obs_pos" in data and "act_pos" in data:
        obs_pos = data["obs_pos"]
        act_pos = data["act_pos"]
        min_len = min(len(obs_pos), len(act_pos))

        diff = np.abs(obs_pos[:min_len] - act_pos[:min_len])
        print(f"\n  |action - observation| joint_position:")
        print(f"    Mean diff: {diff.mean():.6f} rad")
        print(f"    Max diff:  {diff.max():.6f} rad")

        # Find joints with largest diff
        max_diff_per_joint = diff.max(axis=0)
        top_diff_joints = np.argsort(max_diff_per_joint)[-5:][::-1]
        print(f"\n    Top 5 joints with largest diff:")
        for j in top_diff_joints:
            print(f"      [{j:2d}] {JOINT_NAMES[j]:12s}: max_diff={max_diff_per_joint[j]:.4f} rad")

    # Plot
    if plot or save_path:
        _plot_joints(data, episode["id"], save_path)

    # Per-joint plots
    if save_per_joint_vel:
        _plot_per_joint(data, episode["id"], save_per_joint_vel, data_type="vel")
    if save_per_joint_pos:
        _plot_per_joint(data, episode["id"], save_per_joint_pos, data_type="pos")


def _plot_joints(data: dict, episode_id: str, save_path: str = None):
    """Plot joint data in multi-panel layout."""
    import matplotlib.pyplot as plt

    # Define panels to plot: (data_key, title, ylabel)
    panels = [
        ("obs_pos", "observation/state/joint_position", "Position (rad)"),
        ("act_pos", "action/agent/joint_position", "Position (rad)"),
        ("obs_vel", "observation/state/joint_velocity", "Velocity (rad/s)"),
        ("act_vel", "action/agent/joint_velocity", "Velocity (rad/s)"),
        ("obs_tau", "observation/state/joint_torque", "Torque (Nm)"),
        ("act_tau", "action/agent/joint_torque", "Torque (Nm)"),
    ]

    # Filter to only panels with data
    available_panels = [(k, t, y) for k, t, y in panels if k in data]

    if not available_panels:
        print("No joint data to plot")
        return

    n_panels = len(available_panels)
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"Joint Data - {episode_id}", fontsize=14)

    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 28))

    for idx, (key, title, ylabel) in enumerate(available_panels):
        ax = axes[idx]
        arr = data[key]
        num_steps = len(arr)
        time_steps = np.arange(num_steps)

        # Plot all 28 joints
        for j in range(min(28, arr.shape[1])):
            if np.any(arr[:, j] != 0):  # Only plot non-zero joints
                ax.plot(time_steps, arr[:, j], color=colors[j],
                       linewidth=1.0, alpha=0.8, label=JOINT_NAMES[j] if j < 7 else None)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Step", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add box around position/velocity panels to highlight
        if "position" in title or "velocity" in title:
            for spine in ax.spines.values():
                spine.set_edgecolor('red' if "action" in title else 'blue')
                spine.set_linewidth(2)

    # Hide unused axes
    for idx in range(len(available_panels), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def _plot_per_joint(data: dict, episode_id: str, save_path: str = None, data_type: str = "vel"):
    """Plot each joint separately with obs vs action comparison.

    Args:
        data: Dictionary containing joint data
        episode_id: Episode identifier string
        save_path: Path to save the plot
        data_type: "vel" for velocity, "pos" for position
    """
    import matplotlib.pyplot as plt

    if data_type == "pos":
        obs_data = data.get("obs_pos")
        act_data = data.get("act_pos")
        ylabel = "Pos (rad)"
        title_prefix = "Position"
    else:
        obs_data = data.get("obs_vel")
        act_data = data.get("act_vel")
        ylabel = "Vel (rad/s)"
        title_prefix = "Velocity"

    if obs_data is None:
        print(f"No observation joint {data_type} data to plot")
        return

    num_steps = len(obs_data)
    time_steps = np.arange(num_steps)

    # 7 cols x 4 rows = 28 joints
    fig, axes = plt.subplots(4, 7, figsize=(24, 12), sharex=True)
    fig.suptitle(f"Per-Joint {title_prefix}: Obs(blue) vs Action(red) - {episode_id}", fontsize=14)

    axes = axes.flatten()

    for j in range(28):
        ax = axes[j]

        # Plot observation (blue)
        ax.plot(time_steps, obs_data[:, j], 'b-', linewidth=1.2, label='obs', alpha=0.8)

        # Plot action (red) if available
        if act_data is not None:
            ax.plot(time_steps[:len(act_data)], act_data[:, j], 'r--', linewidth=1.0, label='act', alpha=0.7)

        ax.set_title(f"[{j}] {JOINT_NAMES[j]}", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Only add legend to first plot
        if j == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Add common labels
    for ax in axes[-7:]:
        ax.set_xlabel("Step", fontsize=8)
    for i in range(4):
        axes[i * 7].set_ylabel(ylabel, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-joint {data_type} plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Check joint data in TFDS dataset")
    parser.add_argument("dataset_path", type=str, help="TFDS dataset path")
    parser.add_argument("episode_idx", type=int, nargs="?", default=0, help="Episode index")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--save-per-joint-vel", type=str, default=None, help="Save per-joint velocity comparison plot")
    parser.add_argument("--save-per-joint-pos", type=str, default=None, help="Save per-joint position comparison plot")
    args = parser.parse_args()

    check_tfds_joints(args.dataset_path, args.episode_idx, args.plot, args.save,
                      args.save_per_joint_vel, args.save_per_joint_pos)


if __name__ == "__main__":
    main()

