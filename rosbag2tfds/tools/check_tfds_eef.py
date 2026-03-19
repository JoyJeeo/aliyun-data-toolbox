#!/usr/bin/env python3
"""检查 TFDS 数据集中的末端执行器数据。"""

import sys
import glob
from pathlib import Path

import numpy as np
import tensorflow as tf


def check_tfds_eef(dataset_path: str, episode_idx: int = 0):
    """检查 TFDS 中的 eef 数据。"""
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")

    tfrecord_files = sorted(glob.glob(str(train_dir / "*.tfrecord*")))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {train_dir}")

    print(f"📁 Found {len(tfrecord_files)} TFRecord files")

    # Read episode
    all_episodes = []
    for tfrecord_file in tfrecord_files:
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            features = example.features.feature
            
            if 'episode_metadata/episode_id' in features:
                ep_id = features['episode_metadata/episode_id'].bytes_list.value[0].decode('utf-8')
            else:
                ep_id = f"episode_{len(all_episodes)}"
            
            if 'episode_metadata/eef_type' in features:
                eef_type = features['episode_metadata/eef_type'].bytes_list.value[0].decode('utf-8')
            else:
                eef_type = "unknown"
            
            if 'episode_metadata/eef_dim' in features:
                eef_dim = features['episode_metadata/eef_dim'].int64_list.value[0]
            else:
                eef_dim = 0
            
            if 'episode_metadata/num_steps' in features:
                num_steps = features['episode_metadata/num_steps'].int64_list.value[0]
            else:
                num_steps = 0
            
            all_episodes.append({
                'id': ep_id,
                'eef_type': eef_type,
                'eef_dim': eef_dim,
                'num_steps': num_steps,
                'features': features
            })

    print(f"📊 Total episodes: {len(all_episodes)}")
    
    # Show all episodes
    print("\n" + "=" * 60)
    print("📋 Episode Summary:")
    print("=" * 60)
    for i, ep in enumerate(all_episodes):
        print(f"  [{i}] {ep['id']}: eef_type={ep['eef_type']}, eef_dim={ep['eef_dim']}, steps={ep['num_steps']}")

    if episode_idx >= len(all_episodes):
        print(f"❌ Episode {episode_idx} not found")
        return

    episode = all_episodes[episode_idx]
    features = episode['features']
    num_steps = episode['num_steps']
    
    print(f"\n" + "=" * 60)
    print(f"🔍 Checking episode {episode_idx}: {episode['id']}")
    print("=" * 60)
    
    # Check observation eef fields
    eef_fields = [
        ("observation/state/eef_position", 12),
        ("observation/state/eef_velocity", 12),
        ("observation/state/eef_effort", 12),
        ("action/agent/eef_position", 12),
        ("action/agent/eef_velocity", 12),
    ]
    
    print("\n📊 EEF Data Analysis:")
    for field, dim in eef_fields:
        key = f"steps/{field}"
        if key in features:
            values = np.array(features[key].float_list.value).reshape(-1, dim)
            
            # Check which dimensions have non-zero data
            non_zero_dims = []
            for d in range(dim):
                if np.any(values[:, d] != 0):
                    non_zero_dims.append(d)
            
            print(f"\n  {field}:")
            print(f"    Shape: {values.shape}")
            print(f"    Non-zero dimensions: {non_zero_dims if non_zero_dims else 'NONE (all zeros!)'}")
            print(f"    Min: {values.min():.6f}, Max: {values.max():.6f}")
            
            # Show per-dimension stats for non-zero dims
            if non_zero_dims:
                for d in non_zero_dims[:6]:  # Show first 6
                    print(f"    Dim[{d}]: min={values[:, d].min():.4f}, max={values[:, d].max():.4f}, mean={values[:, d].mean():.4f}")
        else:
            print(f"\n  ❌ {field}: NOT FOUND")
    
    # Check for both leju_claw and dexhand patterns
    print("\n" + "=" * 60)
    print("🔎 EEF Type Detection:")
    print("=" * 60)
    
    obs_eef_key = "steps/observation/state/eef_position"
    if obs_eef_key in features:
        values = np.array(features[obs_eef_key].float_list.value).reshape(-1, 12)
        
        # leju_claw pattern: only dims 0-1 have data
        dims_01_active = np.any(values[:, :2] != 0)
        dims_2_11_active = np.any(values[:, 2:] != 0)
        
        if dims_01_active and not dims_2_11_active:
            print("  ✅ Pattern matches leju_claw (only dims 0-1 have data)")
        elif dims_2_11_active:
            print("  ✅ Pattern matches dexhand (dims 2-11 have data)")
        elif not dims_01_active and not dims_2_11_active:
            print("  ❌ All dimensions are zero! No EEF data!")
        else:
            print("  ⚠️  Mixed pattern")
        
        print(f"\n  Dims 0-1 (leju_claw): min={values[:, :2].min():.4f}, max={values[:, :2].max():.4f}")
        print(f"  Dims 2-11 (dexhand): min={values[:, 2:].min():.4f}, max={values[:, 2:].max():.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_tfds_eef.py /path/to/dataset/delivery_openx/1.0.0 [episode_idx]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    check_tfds_eef(dataset_path, episode_idx)


if __name__ == "__main__":
    main()

