#!/usr/bin/env python3
"""
测试 metadata 时间裁剪功能

使用方法:
    python test_crop_function.py

测试场景:
    1. 无 metadata -> 不裁剪，处理所有帧
    2. 完整 metadata (0.0-1.0) -> 处理所有帧
    3. 部分 metadata (0.1-0.9) -> 只处理中间部分
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from kuavo_data.common.kuavo_dataset import KuavoRosbagReader, KuavoMsgProcesser
from kuavo_data.common.chunk_process import ChunkedRosbagProcessor

def create_processor(bag_file, crop_range=None):
    """创建处理器，手动设置参数（不依赖 kuavo 模块）"""
    msg_processer = KuavoMsgProcesser()

    # 简化的 topic_process_map（只包含关键话题）
    topic_process_map = {
        "head_cam_h": {
            "topic": "/cam_h/color/image_raw/compressed",
            "msg_process_fn": msg_processer.process_color_image,
        },
        "observation.state": {
            "topic": "/sensors_data_raw",
            "msg_process_fn": msg_processer.process_joint_state,
        },
        "action": {
            "topic": "/joint_cmd",
            "msg_process_fn": msg_processer.process_joint_cmd,
        }
    }

    processor = ChunkedRosbagProcessor(
        msg_processer=msg_processer,
        topic_process_map=topic_process_map,
        camera_names=["head_cam_h"],
        train_hz=10,
        main_timeline_fps=30,
        sample_drop=10,
        only_half_up_body=True,
        crop_range=crop_range
    )

    return processor

def test_crop_no_metadata(bag_file):
    """测试1：无 metadata，应该处理所有帧"""
    print("\n=== 测试1：无 metadata ===")
    print(f"Bag file: {bag_file}")

    processor = create_processor(bag_file, crop_range=None)

    main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)

    print(f"总帧数: {len(main_timestamps)}")
    print(f"主时间线: {main_timeline}")

    return len(main_timestamps)

def test_crop_full_range(bag_file):
    """测试2：完整 metadata (0.0-1.0)，应该处理所有帧"""
    print("\n=== 测试2：完整 metadata (0.0-1.0) ===")
    print(f"Bag file: {bag_file}")

    processor = create_processor(bag_file, crop_range=(0.0, 1.0))

    main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)

    print(f"总帧数: {len(main_timestamps)}")
    print(f"主时间线: {main_timeline}")

    return len(main_timestamps)

def test_crop_partial_range(bag_file):
    """测试3：部分 metadata (0.1-0.9)，应该裁剪掉前后各10%"""
    print("\n=== 测试3：部分 metadata (0.1-0.9) ===")
    print(f"Bag file: {bag_file}")

    processor = create_processor(bag_file, crop_range=(0.1, 0.9))

    main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)

    print(f"总帧数: {len(main_timestamps)}")
    print(f"主时间线: {main_timeline}")

    return len(main_timestamps)

def test_crop_from_metadata(bag_file, metadata_file):
    """测试4：从 metadata.json 读取裁剪范围"""
    print("\n=== 测试4：从 metadata.json 读取裁剪范围 ===")
    print(f"Bag file: {bag_file}")
    print(f"Metadata: {metadata_file}")

    # 读取 metadata.json
    import json
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 从 metadata 获取裁剪范围
    marks = metadata.get("marks", [])
    if marks:
        start_positions = [m.get("startPosition", 0.0) for m in marks]
        end_positions = [m.get("endPosition", 1.0) for m in marks]
        min_start = min(start_positions)
        max_end = max(end_positions)
        crop_range = (min_start, max_end)
        print(f"从 marks 计算裁剪范围: {crop_range}")
    else:
        print("Warning: metadata 中没有 marks")
        crop_range = None

    # 测试裁剪
    processor = create_processor(bag_file, crop_range=crop_range)

    main_timeline, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)

    print(f"总帧数: {len(main_timestamps)}")
    print(f"主时间线: {main_timeline}")

    return len(main_timestamps), crop_range

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 metadata 时间裁剪功能")
    parser.add_argument("--bag", type=str, required=True, help="ROSbag 文件路径")
    parser.add_argument("--metadata", type=str, help="metadata.json 文件路径")

    args = parser.parse_args()

    print("=" * 60)
    print("Metadata 时间裁剪功能测试")
    print("=" * 60)

    # 测试1
    total_frames = test_crop_no_metadata(args.bag)

    # 测试2
    full_frames = test_crop_full_range(args.bag)

    # 验证测试1和2结果一致
    assert total_frames == full_frames, "测试1和2应该结果一致"
    print("✓ 测试1和2结果一致")

    # 测试3
    partial_frames = test_crop_partial_range(args.bag)

    # 验证部分裁剪后帧数减少
    assert partial_frames < total_frames, "部分裁剪后帧数应该减少"
    reduction = (1 - partial_frames / total_frames) * 100
    print(f"✓ 裁剪后帧数减少: {reduction:.1f}%")

    # 测试4（如果有 metadata）
    if args.metadata:
        metadata_frames, crop_range = test_crop_from_metadata(args.bag, args.metadata)

        # 验证 metadata 裁剪结果
        print(f"✓ metadata 裁剪范围: {crop_range[0]:.3f}-{crop_range[1]:.3f}")
        if crop_range[0] > 0 or crop_range[1] < 1.0:
            print(f"  预期帧数减少: {(1 - (crop_range[1] - crop_range[0])) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
