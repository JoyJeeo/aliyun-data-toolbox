#!/usr/bin/env python3
"""检查末端执行器（夹爪/灵巧手）数据并绘图 - 从 TFRecord 读取。"""

import sys
import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots")


def read_tfds_eef(dataset_path: str, episode_idx: int = 0):
    """从 TFDS 读取 eef 数据。"""
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {train_dir}")

    print(f"📁 Found {len(tfrecord_files)} TFRecord files")

    # Read all episodes
    all_episodes = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            features = example.features.feature

            ep_id = features['episode_metadata/episode_id'].bytes_list.value[0].decode('utf-8') if 'episode_metadata/episode_id' in features else f"ep_{len(all_episodes)}"
            eef_type = features['episode_metadata/eef_type'].bytes_list.value[0].decode('utf-8') if 'episode_metadata/eef_type' in features else "unknown"
            num_steps = features['episode_metadata/num_steps'].int64_list.value[0] if 'episode_metadata/num_steps' in features else 0

            all_episodes.append({'id': ep_id, 'eef_type': eef_type, 'num_steps': num_steps, 'features': features})

    print(f"📊 Total episodes: {len(all_episodes)}")

    # Show all episodes
    print("\n" + "=" * 60)
    print("� Episode Summary:")
    print("=" * 60)
    for i, ep in enumerate(all_episodes):
        marker = "→" if i == episode_idx else " "
        print(f"  {marker}[{i}] {ep['id']}: eef_type={ep['eef_type']}, steps={ep['num_steps']}")

    if episode_idx >= len(all_episodes):
        print(f"❌ Episode {episode_idx} not found")
        return None, None

    episode = all_episodes[episode_idx]
    features = episode['features']

    # Extract eef data
    eef_fields = {
        'obs_eef_position': ('steps/observation/state/eef_position', 12),
        'obs_eef_velocity': ('steps/observation/state/eef_velocity', 12),
        'obs_eef_effort': ('steps/observation/state/eef_effort', 12),
        'action_eef_position': ('steps/action/agent/eef_position', 12),
        'action_eef_velocity': ('steps/action/agent/eef_velocity', 12),
    }

    data = {}
    for name, (key, dim) in eef_fields.items():
        if key in features:
            values = np.array(features[key].float_list.value).reshape(-1, dim)
            data[name] = values
        else:
            data[name] = np.array([])

    return episode, data


def plot_eef_from_tfds(episode: dict, data: dict, output_dir: str):
    """绘制 TFDS 中的 eef 数据。"""
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available, skipping plots")
        return

    ep_id = episode['id']
    eef_type = episode['eef_type']

    # Determine which dims have data
    obs_pos = data.get('obs_eef_position', np.array([]))
    if len(obs_pos) == 0:
        print("⚠️ No eef_position data to plot")
        return

    # For leju_claw, only dims 0-1 have data
    if eef_type == 'leju_claw':
        active_dims = [0, 1]
        dim_names = ['left_claw', 'right_claw']
    else:
        # dexhand: all 12 dims
        active_dims = list(range(12))
        dim_names = [f'dim_{i}' for i in range(12)]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Episode: {ep_id} (eef_type={eef_type})', fontsize=14)

    # Observation - Position
    ax = axes[0, 0]
    for i, d in enumerate(active_dims):
        ax.plot(obs_pos[:, d], label=dim_names[i], alpha=0.8, linewidth=1.5)
    ax.set_title('Observation: eef_position')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Observation - Velocity
    ax = axes[1, 0]
    obs_vel = data.get('obs_eef_velocity', np.array([]))
    if len(obs_vel) > 0:
        for i, d in enumerate(active_dims):
            ax.plot(obs_vel[:, d], label=dim_names[i], alpha=0.8, linewidth=1.5)
    ax.set_title('Observation: eef_velocity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)

    # Observation - Effort
    ax = axes[2, 0]
    obs_eff = data.get('obs_eef_effort', np.array([]))
    if len(obs_eff) > 0:
        for i, d in enumerate(active_dims):
            ax.plot(obs_eff[:, d], label=dim_names[i], alpha=0.8, linewidth=1.5)
    ax.set_title('Observation: eef_effort')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effort')
    ax.grid(True, alpha=0.3)

    # Action - Position
    ax = axes[0, 1]
    act_pos = data.get('action_eef_position', np.array([]))
    if len(act_pos) > 0:
        for i, d in enumerate(active_dims):
            ax.plot(act_pos[:, d], label=dim_names[i], alpha=0.8, linewidth=1.5)
    ax.set_title('Action: eef_position')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Action - Velocity
    ax = axes[1, 1]
    act_vel = data.get('action_eef_velocity', np.array([]))
    if len(act_vel) > 0:
        for i, d in enumerate(active_dims):
            ax.plot(act_vel[:, d], label=dim_names[i], alpha=0.8, linewidth=1.5)
    ax.set_title('Action: eef_velocity')
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)

    # Stats summary
    ax = axes[2, 1]
    ax.axis('off')
    stats_text = f"""
Episode: {ep_id}
EEF Type: {eef_type}
Steps: {len(obs_pos)}

Observation eef_position:
  Dim 0 (left):  min={obs_pos[:, 0].min():.2f}, max={obs_pos[:, 0].max():.2f}
  Dim 1 (right): min={obs_pos[:, 1].min():.2f}, max={obs_pos[:, 1].max():.2f}

Action eef_position:
  Dim 0 (left):  min={act_pos[:, 0].min():.2f}, max={act_pos[:, 0].max():.2f}
  Dim 1 (right): min={act_pos[:, 1].min():.2f}, max={act_pos[:, 1].max():.2f}
"""
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path(output_dir) / f"eef_tfds_{ep_id}.png"
    plt.savefig(output_path, dpi=150)
    print(f"  📊 Plot saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_eef_data.py /path/to/dataset/delivery_openx/1.0.0 [episode_idx] [output_dir]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_dir = sys.argv[3] if len(sys.argv) > 3 else dataset_path

    print(f"\n📂 Dataset: {dataset_path}")
    print(f"� Episode index: {episode_idx}")

    episode, data = read_tfds_eef(dataset_path, episode_idx)

    if episode is not None:
        print(f"\n� Plotting episode: {episode['id']}...")
        plot_eef_from_tfds(episode, data, output_dir)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()

