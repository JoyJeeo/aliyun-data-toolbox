#!/usr/bin/env python3
"""
Verify world_vector and rotation_delta calculations in TFDS dataset.

Usage:
    python verify_deltas.py /path/to/dataset/delivery_openx/1.0.0

This script checks:
1. world_vector[t] ≈ tcp_pos[t+1] - tcp_pos[t]
2. rotation_delta has no discontinuities (wrap-around)
3. Plots curves for visual inspection
"""

import glob
import json
import sys
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("Please install tensorflow: pip install tensorflow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping plots")


def load_episode_from_tfrecord(dataset_path: str, episode_idx: int = 0):
    """Load a single episode from TFRecord files directly.

    In RLDS format, each TFRecord contains one episode with all steps flattened.
    """
    # Find train directory
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    # Find all TFRecord files
    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {train_dir}")

    print(f"📁 Found {len(tfrecord_files)} TFRecord files")

    # Read all records (each record = one episode in RLDS format)
    all_episodes = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            features = example.features.feature

            # Get episode_id and num_steps
            if 'episode_metadata/episode_id' in features:
                ep_id = features['episode_metadata/episode_id'].bytes_list.value[0].decode('utf-8')
            else:
                ep_id = f"episode_{len(all_episodes)}"

            if 'episode_metadata/num_steps' in features:
                num_steps = features['episode_metadata/num_steps'].int64_list.value[0]
            else:
                # Infer from data length
                if 'steps/action/world_vector_left' in features:
                    num_steps = len(features['steps/action/world_vector_left'].float_list.value) // 3
                else:
                    num_steps = 0

            all_episodes.append({
                'id': ep_id,
                'num_steps': num_steps,
                'features': features
            })

    print(f"� Total episodes: {len(all_episodes)}")

    # Get the requested episode
    if episode_idx >= len(all_episodes):
        raise ValueError(f"Episode {episode_idx} not found, only {len(all_episodes)} episodes available")

    episode = all_episodes[episode_idx]
    print(f"📋 Loading episode: {episode['id']} ({episode['num_steps']} steps)")

    return [episode['features']], episode['num_steps']


def extract_arrays_from_features(steps, num_steps: int):
    """Extract relevant arrays from TFRecord features (RLDS format).

    In RLDS format, each episode is stored as a single TFRecord with all steps flattened.
    For example, tcp_position_left has shape [num_steps * 3] for 3D positions.
    """
    # Get data from the first (and only) record - it contains all steps
    features = steps[0]

    # Extract tcp_position_left: shape [num_steps * 3] -> [num_steps, 3]
    if 'steps/observation/state/tcp_position_left' in features:
        tcp_pos_flat = list(features['steps/observation/state/tcp_position_left'].float_list.value)
        tcp_pos_left = np.array(tcp_pos_flat).reshape(-1, 3)
    else:
        tcp_pos_left = np.zeros((num_steps, 3))

    # Extract tcp_orientation_left: shape [num_steps * 4] -> [num_steps, 4]
    if 'steps/observation/state/tcp_orientation_left' in features:
        tcp_quat_flat = list(features['steps/observation/state/tcp_orientation_left'].float_list.value)
        tcp_quat_left = np.array(tcp_quat_flat).reshape(-1, 4)
    else:
        tcp_quat_left = np.zeros((num_steps, 4))
        tcp_quat_left[:, 3] = 1  # default quaternion [0,0,0,1]

    # Extract world_vector_left: shape [num_steps * 3] -> [num_steps, 3]
    if 'steps/action/world_vector_left' in features:
        wv_flat = list(features['steps/action/world_vector_left'].float_list.value)
        world_vector = np.array(wv_flat).reshape(-1, 3)
    else:
        world_vector = np.zeros((num_steps, 3))

    # Extract rotation_delta_left: shape [num_steps * 3] -> [num_steps, 3]
    if 'steps/action/rotation_delta_left' in features:
        rd_flat = list(features['steps/action/rotation_delta_left'].float_list.value)
        rotation_delta = np.array(rd_flat).reshape(-1, 3)
    else:
        rotation_delta = np.zeros((num_steps, 3))

    return {
        'tcp_pos_left': tcp_pos_left,
        'tcp_quat_left': tcp_quat_left,
        'world_vector': world_vector,
        'rotation_delta': rotation_delta,
    }


def verify_world_vector(data):
    """Verify world_vector[t] = tcp_pos[t+1] - tcp_pos[t]."""
    tcp_pos = data['tcp_pos_left']
    world_vector = data['world_vector']
    
    n_steps = len(tcp_pos)
    errors = []
    
    print("\n" + "=" * 60)
    print("📐 Verifying world_vector = tcp_pos[t+1] - tcp_pos[t]")
    print("=" * 60)
    
    for t in range(n_steps - 1):
        expected = tcp_pos[t + 1] - tcp_pos[t]
        actual = world_vector[t]
        error = np.linalg.norm(expected - actual)
        errors.append(error)
        
        if error > 1e-5:
            print(f"  ⚠️  Step {t}: error = {error:.6f}")
            print(f"      Expected: {expected}")
            print(f"      Actual:   {actual}")
    
    # Check last step (should be zeros)
    if np.linalg.norm(world_vector[-1]) > 1e-5:
        print(f"  ⚠️  Last step world_vector should be zeros: {world_vector[-1]}")
    
    max_error = max(errors) if errors else 0
    mean_error = np.mean(errors) if errors else 0
    
    if max_error < 1e-5:
        print(f"  ✅ All {n_steps - 1} steps verified! Max error: {max_error:.2e}")
    else:
        print(f"  ❌ Verification failed! Max error: {max_error:.2e}, Mean: {mean_error:.2e}")
    
    return errors


def check_rotation_discontinuities(data, threshold=0.5):
    """Check for discontinuities in rotation_delta (wrap-around detection)."""
    rotation_delta = data['rotation_delta']

    print("\n" + "=" * 60)
    print("🔄 Checking rotation_delta for wrap-around discontinuities")
    print("=" * 60)

    # Check 1: Large jumps in rotation_delta values between adjacent steps
    discontinuities = []
    for t in range(1, len(rotation_delta)):
        diff = np.abs(rotation_delta[t] - rotation_delta[t - 1])
        if np.any(diff > threshold):
            discontinuities.append((t, diff))

    if discontinuities:
        print(f"  ⚠️  Found {len(discontinuities)} jumps > {threshold} rad:")
        for t, diff in discontinuities[:10]:  # Show first 10
            print(f"      Step {t}: jump = [{diff[0]:.3f}, {diff[1]:.3f}, {diff[2]:.3f}] rad")
        if len(discontinuities) > 10:
            print(f"      ... and {len(discontinuities) - 10} more")
    else:
        print(f"  ✅ No large jumps found (threshold: {threshold} rad)")

    # Check 2: Values near ±π (sign of wrap-around issues)
    near_pi_count = np.sum(np.abs(rotation_delta) > 2.5)  # > ~143 degrees
    if near_pi_count > 0:
        print(f"  ⚠️  {near_pi_count} values near ±π (potential wrap-around)")
    else:
        print(f"  ✅ No values near ±π")

    # Statistics
    print(f"\n  📊 rotation_delta statistics:")
    print(f"      min:  [{rotation_delta[:, 0].min():.4f}, {rotation_delta[:, 1].min():.4f}, {rotation_delta[:, 2].min():.4f}]")
    print(f"      max:  [{rotation_delta[:, 0].max():.4f}, {rotation_delta[:, 1].max():.4f}, {rotation_delta[:, 2].max():.4f}]")
    print(f"      mean: [{rotation_delta[:, 0].mean():.4f}, {rotation_delta[:, 1].mean():.4f}, {rotation_delta[:, 2].mean():.4f}]")

    return discontinuities


def plot_curves(data, output_path: str = None):
    """Plot world_vector and rotation_delta curves."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # World vector
    ax = axes[0, 0]
    wv = data['world_vector']
    ax.plot(wv[:, 0], label='x', alpha=0.8)
    ax.plot(wv[:, 1], label='y', alpha=0.8)
    ax.plot(wv[:, 2], label='z', alpha=0.8)
    ax.set_title('world_vector (left arm)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Delta position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotation delta with ±π reference lines
    ax = axes[0, 1]
    rd = data['rotation_delta']
    ax.plot(rd[:, 0], label='roll', alpha=0.8)
    ax.plot(rd[:, 1], label='pitch', alpha=0.8)
    ax.plot(rd[:, 2], label='yaw', alpha=0.8)
    # Add ±π reference lines to show wrap-around threshold
    ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='±π')
    ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
    ax.set_title('rotation_delta (left arm) - check for ±π jumps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Delta rotation (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotation delta - full scale view (like customer's image)
    ax = axes[0, 2]
    ax.plot(rd[:, 0], label='roll', alpha=0.8)
    ax.plot(rd[:, 1], label='pitch', alpha=0.8)
    ax.plot(rd[:, 2], label='yaw', alpha=0.8)
    ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
    ax.set_ylim(-4, 4)  # Fixed scale to match customer's image
    ax.set_title('rotation_delta (fixed scale ±4 rad)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Delta rotation (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TCP position
    ax = axes[1, 0]
    tcp = data['tcp_pos_left']
    ax.plot(tcp[:, 0], label='x', alpha=0.8)
    ax.plot(tcp[:, 1], label='y', alpha=0.8)
    ax.plot(tcp[:, 2], label='z', alpha=0.8)
    ax.set_title('TCP position (left arm)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify: computed vs actual world_vector
    ax = axes[1, 1]
    computed_wv = np.diff(tcp, axis=0)
    actual_wv = wv[:-1]
    ax.plot(np.linalg.norm(computed_wv - actual_wv, axis=1), 'r-', alpha=0.8)
    ax.set_title('world_vector error: |computed - actual|')
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (m)')
    ax.grid(True, alpha=0.3)

    # Rotation delta jumps between steps
    ax = axes[1, 2]
    rd_diff = np.abs(np.diff(rd, axis=0))
    ax.plot(rd_diff[:, 0], label='roll jump', alpha=0.8)
    ax.plot(rd_diff[:, 1], label='pitch jump', alpha=0.8)
    ax.plot(rd_diff[:, 2], label='yaw jump', alpha=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='threshold')
    ax.set_title('rotation_delta step-to-step jumps')
    ax.set_xlabel('Step')
    ax.set_ylabel('|delta[t] - delta[t-1]| (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\n📊 Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_deltas.py /path/to/dataset/delivery_openx/1.0.0 [episode_idx]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"\n📂 Dataset: {dataset_path}")
    print(f"📋 Episode: {episode_idx}")

    # Load episode from TFRecord files directly
    steps, num_steps = load_episode_from_tfrecord(dataset_path, episode_idx)
    data = extract_arrays_from_features(steps, num_steps)

    print(f"📊 Steps: {len(data['tcp_pos_left'])}")

    # Verify
    verify_world_vector(data)
    check_rotation_discontinuities(data)

    # Plot
    output_path = Path(dataset_path) / f"delta_verification_ep{episode_idx}.png"
    plot_curves(data, str(output_path))


if __name__ == "__main__":
    main()

