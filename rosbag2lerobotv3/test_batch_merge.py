#!/usr/bin/env python3
"""
测试批次合并功能
"""

import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "/home/zhangyutao/Documents/Work/Code/Contest/aliyun-data-toolbox/rosbag2lerobotv3")

from kuavo_data.converter.pipeline.batch_merger import (
    merge_all_batches,
    merge_parquet_files,
    merge_metadata,
    merge_meta_files,
    get_batch_dirs
)


def test_batch_merger():
    """测试批次合并功能"""
    print("=" * 60)
    print("测试批次合并功能")
    print("=" * 60)

    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        print(f"\n✓ 测试目录: {test_dir}")

        # 创建模拟的 batch 目录结构
        num_batches = 3
        for i in range(num_batches):
            batch_dir = test_dir / f"batch_{i:04d}"
            batch_dir.mkdir()

            # 创建 metadata.json
            metadata = {
                "episode_id": f"test_episode_{i}",
                "label_info": {
                    "action_config": [
                        {
                            "skill": f"action_{i}",
                            "start_frame": 0,
                            "end_frame": 99,
                            "timestamp_utc": f"2026-02-04T15:22:{i*10:02d}+08:00",
                            "is_mistake": False
                        }
                    ]
                },
                "total_frames": 100
            }

            with open(batch_dir / "metadata.json", "w") as f:
                import json
                json.dump(metadata, f, indent=2)

            # 创建 data 目录和 parquet 文件
            data_dir = batch_dir / "data" / "chunk-000"
            data_dir.mkdir(parents=True)

            # 创建简单的 parquet 文件用于测试
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                # 创建一个简单的表格
                data = {
                    'timestamp': [float(j) for j in range(100)],
                    'frame_index': list(range(100)),
                    'index': list(range(100)),
                }
                table = pa.table(data)
                pq.write_table(table, data_dir / "episode_000000.parquet")
                print(f"  ✓ 创建 parquet: {data_dir / 'episode_000000.parquet'}")
            except ImportError:
                print("  ⚠️  跳过 parquet 创建（需要安装 pyarrow）")

            # 创建 meta 目录和 info.json
            meta_dir = batch_dir / "meta"
            meta_dir.mkdir()

            info = {
                "total_frames": 100,
                "total_videos": 3,
                "features": {
                    "observation.images.head_cam_h": {
                        "dtype": "video",
                        "shape": [3, 480, 848],
                        "names": ["channels", "height", "width"],
                        "info": {
                            "video.height": 480,
                            "video.width": 848,
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "video.fps": 30,
                            "video.channels": 3,
                            "has_audio": False
                        }
                    }
                }
            }

            with open(meta_dir / "info.json", "w") as f:
                json.dump(info, f, indent=2)

            # 创建 tasks.jsonl
            with open(meta_dir / "tasks.jsonl", "w") as f:
                f.write('{"task": "DEBUG"}\n')

        print(f"\n✓ 创建了 {num_batches} 个 batch 目录")

        # 测试 get_batch_dirs
        print("\n[1/4] 测试 get_batch_dirs...")
        batch_dirs = get_batch_dirs(test_dir)
        print(f"  找到 {len(batch_dirs)} 个 batch: {[d.name for d in batch_dirs]}")
        assert len(batch_dirs) == num_batches, "batch 目录数量不正确"

        # 测试 merge_all_batches
        print("\n[2/4] 测试 merge_all_batches...")
        output_dir = test_dir / "merged_output"
        try:
            merge_all_batches(
                input_dir=str(test_dir),
                output_dir=str(output_dir)
            )
            print("  ✓ 合并成功")
        except Exception as e:
            print(f"  ✗ 合并失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 验证输出
        print("\n[3/4] 验证输出文件...")
        assert output_dir.exists(), "输出目录不存在"

        # 检查 metadata.json
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists(), "metadata.json 不存在"
        print(f"  ✓ metadata.json 存在")

        import json
        with open(metadata_path, "r") as f:
            merged_metadata = json.load(f)

        print(f"  总帧数: {merged_metadata.get('total_frames')}")
        print(f"  动作数量: {len(merged_metadata.get('label_info', {}).get('action_config', []))}")

        # 检查 meta 目录
        meta_dir = output_dir / "meta"
        assert meta_dir.exists(), "meta 目录不存在"

        episodes_path = meta_dir / "episodes.jsonl"
        assert episodes_path.exists(), "episodes.jsonl 不存在"
        print(f"  ✓ episodes.jsonl 存在")

        info_path = meta_dir / "info.json"
        assert info_path.exists(), "info.json 不存在"
        print(f"  ✓ info.json 存在")

        tasks_path = meta_dir / "tasks.jsonl"
        assert tasks_path.exists(), "tasks.jsonl 不存在"
        print(f"  ✓ tasks.jsonl 存在")

        # 检查 parameters 目录（如果有）
        params_dir = output_dir / "parameters"
        if params_dir.exists():
            print(f"  ✓ parameters 目录存在")

        # 验证动作配置
        print("\n[4/4] 验证动作配置...")
        actions = merged_metadata.get('label_info', {}).get('action_config', [])
        print(f"  动作数量: {len(actions)}")

        if len(actions) > 0:
            first = actions[0]
            last = actions[-1]

            # 第一个动作应该从 0 开始
            assert first.get('start_frame') == 0, "第一个动作应该从帧 0 开始"
            print(f"  ✓ 第一个动作从帧 0 开始")

            # 最后一个动作应该到总帧数-1
            total_frames = merged_metadata.get('total_frames', 0)
            assert last.get('end_frame') == total_frames - 1, f"最后一个动作应该到帧 {total_frames - 1}"
            print(f"  ✓ 最后一个动作到帧 {total_frames - 1}")

            # 检查动作区间连续性
            for i in range(1, len(actions)):
                prev = actions[i-1]
                curr = actions[i]
                assert curr.get('start_frame') == prev.get('end_frame'), \
                    f"动作 {i} 的起始帧应该等于前一个动作的结束帧"
            print(f"  ✓ 动作区间连续")

        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过！")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = test_batch_merger()
    sys.exit(0 if success else 1)
