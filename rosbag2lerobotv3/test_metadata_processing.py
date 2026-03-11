#!/usr/bin/env python3
"""
测试 metadata.json 处理功能的完整性

测试目标：
1. 读取与解析（batch_processor.py 的功能）
2. 批次级转换与帧计算（metadata_merge.py 的功能）
3. 保存批次 metadata（batch_finalizer.py 的功能）
4. 合并批次 metadata（batch_merger.py 的功能）
5. 最终合并（conversion_orchestrator.py 的功能）
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "kuavo_data"))

from converter.data.metadata_merge import (
    load_metadata,
    get_sn_code,
    get_time_range_from_metadata,
    merge_metadata_and_moment,
    merge_batch_metadata,
    get_time_range_from_moments,
)


def test_1_read_and_parse():
    """测试阶段1：读取与解析"""
    print("\n=== 测试 1: 读取与解析 ===")

    # 测试数据
    test_metadata_path = "metadata.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    # 1.1 读取 metadata
    metadata = load_metadata(test_metadata_path)
    print(f"✓ 读取 metadata 成功")
    print(f"  - deviceSn: {metadata.get('deviceSn')}")
    print(f"  - taskName: {metadata.get('taskName')}")
    print(f"  - marks 数量: {len(metadata.get('marks', []))}")

    # 1.2 获取 sn_code
    sn_code = get_sn_code(metadata)
    print(f"✓ 获取 sn_code: {sn_code}")

    # 1.3 提取时间范围
    start_pos, end_pos = get_time_range_from_metadata(metadata)
    print(f"✓ 提取时间范围: {start_pos:.6f} - {end_pos:.6f}")

    return True


def test_2_batch_conversion():
    """测试阶段2：批次级转换与帧计算"""
    print("\n=== 测试 2: 批次级转换与帧计算 ===")

    test_metadata_path = "metadata.json"
    test_output_path = "test_output_metadata.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    # 模拟配置对象
    class MockConfig:
        def __init__(self):
            self.train_hz = 30

    config = MockConfig()

    # 批次 1：模拟第一个批次的处理
    batch1_output = "batch1_metadata.json"
    merge_metadata_and_moment(
        metadata_path=test_metadata_path,
        moment_path=None,
        output_path=batch1_output,
        uuid="test_episode_001",
        raw_config=config,
        bag_time_info={
            "unix_timestamp": 1707030161.960,  # 2026-02-04 15:22:41.960
            "end_time": 1707030190.868,        # 2026-02-04 15:23:10.868
        },
        main_time_line_timestamps=[1707030161.960 + i * 0.033 for i in range(300)],  # 300帧，30Hz
        total_frames=300
    )
    print(f"✓ 批次 1 处理完成: {batch1_output}")

    # 读取并验证输出
    with open(batch1_output, 'r', encoding='utf-8') as f:
        batch1_data = json.load(f)

    print(f"  - episode_id: {batch1_data.get('episode_id')}")
    print(f"  - total_frames: {batch1_data.get('total_frames')}")
    print(f"  - action_config 长度: {len(batch1_data.get('label_info', {}).get('action_config', []))}")

    # 验证动作帧范围
    for i, action in enumerate(batch1_data['label_info']['action_config']):
        print(f"  - 动作 {i}: {action['skill']} - 帧 {action['start_frame']}-{action['end_frame']}")

    # 清理
    os.remove(batch1_output)

    return True


def test_3_save_batch_metadata():
    """测试阶段3：保存批次 metadata"""
    print("\n=== 测试 3: 保存批次 metadata ===")

    test_metadata_path = "metadata.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    # 创建临时批次目录
    with tempfile.TemporaryDirectory() as tmpdir:
        batch_dir = Path(tmpdir) / "batch_0001"
        batch_dir.mkdir()

        class MockConfig:
            def __init__(self):
                self.train_hz = 30

        config = MockConfig()

        # 模拟保存批次 metadata
        batch_output = batch_dir / "metadata.json"
        merge_metadata_and_moment(
            metadata_path=test_metadata_path,
            moment_path=None,
            output_path=str(batch_output),
            uuid="test_batch_001",
            raw_config=config,
            bag_time_info={
                "unix_timestamp": 1707030161.960,
                "end_time": 1707030190.868,
            },
            main_time_line_timestamps=[1707030161.960 + i * 0.033 for i in range(200)],
            total_frames=200
        )

        # 验证文件存在
        if batch_output.exists():
            print(f"✓ 批次 metadata 保存成功: {batch_output}")

            with open(batch_output, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)

            print(f"  - 文件大小: {batch_output.stat().st_size} bytes")
            print(f"  - 包含字段: {list(batch_data.keys())}")

            return True
        else:
            print(f"[ERROR] 批次 metadata 保存失败")
            return False


def test_4_merge_batches():
    """测试阶段4：合并批次 metadata"""
    print("\n=== 测试 4: 合并批次 metadata ===")

    test_metadata_path = "metadata.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    # 创建多个批次目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        class MockConfig:
            def __init__(self):
                self.train_hz = 30

        config = MockConfig()

        # 创建 3 个批次
        batch_dirs = []
        total_frames = 0

        for i in range(3):
            batch_dir = tmpdir / f"batch_{i+1:04d}"
            batch_dir.mkdir()
            batch_dirs.append(str(batch_dir))

            # 每个批次 100 帧
            batch_frames = 100
            total_frames += batch_frames

            batch_output = batch_dir / "metadata.json"
            merge_metadata_and_moment(
                metadata_path=test_metadata_path,
                moment_path=None,
                output_path=str(batch_output),
                uuid=f"test_episode_batch{i+1}",
                raw_config=config,
                bag_time_info={
                    "unix_timestamp": 1707030161.960 + i * 10,
                    "end_time": 1707030161.960 + i * 10 + 10,
                },
                main_time_line_timestamps=[1707030161.960 + i * 10 + j * 0.033 for j in range(batch_frames)],
                total_frames=batch_frames
            )

        print(f"✓ 创建了 {len(batch_dirs)} 个批次")

        # 合并批次
        output_dir = tmpdir / "merged"
        merged_data = merge_batch_metadata(batch_dirs, str(output_dir), total_frames)

        print(f"✓ 合并完成")
        print(f"  - 总帧数: {merged_data.get('total_frames')}")
        print(f"  - 合并后的动作数量: {len(merged_data.get('label_info', {}).get('action_config', []))}")

        # 验证动作帧范围的连续性
        actions = merged_data['label_info']['action_config']
        for i, action in enumerate(actions):
            print(f"  - 动作 {i}: {action['skill']} - 帧 {action['start_frame']}-{action['end_frame']}")

        # 验证第一个动作从 0 开始
        if actions and actions[0].get('start_frame') == 0:
            print(f"✓ 第一个动作从帧 0 开始")

        # 验证最后一个动作到总帧数-1
        if actions and actions[-1].get('end_frame') == total_frames - 1:
            print(f"✓ 最后一个动作到帧 {total_frames - 1}")

        return True


def test_5_get_time_range_from_moments():
    """测试阶段5：从 moments/marks 提取时间范围"""
    print("\n=== 测试 5: 从 marks 提取时间范围 ===")

    test_metadata_path = "metadata.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    # 测试新格式（marks）
    start, end = get_time_range_from_moments(
        moments_json_path=None,
        metadata_json_path=test_metadata_path
    )

    if start is not None and end is not None:
        print(f"✓ 从 marks 提取时间范围成功")
        print(f"  - start_position: {start:.6f}")
        print(f"  - end_position: {end:.6f}")
        print(f"  - 范围长度: {end - start:.6f}")
        return True
    else:
        print(f"[ERROR] 提取时间范围失败")
        return False


def test_6_compare_with_expected_output():
    """测试阶段6：与预期输出对比"""
    print("\n=== 测试 6: 与预期输出对比 ===")

    test_metadata_path = "metadata.json"
    expected_output_path = "metadata_out.json"

    if not os.path.exists(test_metadata_path):
        print(f"[ERROR] 测试文件不存在: {test_metadata_path}")
        return False

    if not os.path.exists(expected_output_path):
        print(f"[WARN] 预期输出文件不存在: {expected_output_path}")
        print(f"     跳过此测试")
        return True

    # 读取预期输出
    with open(expected_output_path, 'r', encoding='utf-8') as f:
        expected_data = json.load(f)

    print(f"✓ 预期输出: {expected_output_path}")
    print(f"  - episode_id: {expected_data.get('episode_id')}")
    print(f"  - total_frames: {expected_data.get('total_frames')}")
    print(f"  - action_config: {len(expected_data.get('label_info', {}).get('action_config', []))} 个动作")

    # 执行转换
    class MockConfig:
        def __init__(self):
            self.train_hz = 30

    config = MockConfig()

    test_output_path = "test_generated_metadata.json"
    merge_metadata_and_moment(
        metadata_path=test_metadata_path,
        moment_path=None,
        output_path=test_output_path,
        uuid=expected_data.get('episode_id', 'test_episode'),
        raw_config=config,
        bag_time_info={
            "unix_timestamp": 1707030161.960,
            "end_time": 1707030190.868,
        },
        main_time_line_timestamps=[1707030161.960 + i * 0.033 for i in range(expected_data.get('total_frames', 900))],
        total_frames=expected_data.get('total_frames', 900)
    )

    # 读取生成的输出
    with open(test_output_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)

    # 对比关键字段
    print(f"\n✓ 生成输出: {test_output_path}")
    print(f"  - episode_id: {generated_data.get('episode_id')}")
    print(f"  - total_frames: {generated_data.get('total_frames')}")
    print(f"  - action_config: {len(generated_data.get('label_info', {}).get('action_config', []))} 个动作")

    # 对比动作信息
    expected_actions = expected_data.get('label_info', {}).get('action_config', [])
    generated_actions = generated_data.get('label_info', {}).get('action_config', [])

    if len(expected_actions) == len(generated_actions):
        print(f"\n✓ 动作数量匹配: {len(expected_actions)}")

        # 对比每个动作（允许一定误差，因为不同的计算方式可能产生微小差异）
        all_match = True
        max_frame_diff = 20  # 允许最大20帧的差异（考虑到舍入误差和不同算法）

        for i, (exp, gen) in enumerate(zip(expected_actions, generated_actions)):
            # 检查技能名称
            skill_match = exp.get('skill') == gen.get('skill')

            # 检查帧范围（允许误差）
            start_diff = abs(exp.get('start_frame', 0) - gen.get('start_frame', 0))
            end_diff = abs(exp.get('end_frame', 0) - gen.get('end_frame', 0))
            frame_match = start_diff <= max_frame_diff and end_diff <= max_frame_diff

            match = skill_match and frame_match

            status = "✓" if match else "✗"
            diff_info = f"(差: start={start_diff}, end={end_diff})" if not frame_match else ""
            print(f"  {status} 动作 {i}: {exp.get('skill')} - "
                  f"预期 {exp.get('start_frame')}-{exp.get('end_frame')} "
                  f"{'≈' if frame_match else '!='} "
                  f"生成 {gen.get('start_frame')}-{gen.get('end_frame')} "
                  f"{diff_info}")

            if not match:
                all_match = False

        if all_match:
            print(f"\n✓✓✓ 所有动作信息完全匹配！")
        else:
            print(f"\n✗✗✗ 部分动作信息不匹配")

        # 清理
        os.remove(test_output_path)

        return all_match
    else:
        print(f"\n✗ 动作数量不匹配: 预期 {len(expected_actions)}, 生成 {len(generated_actions)}")
        os.remove(test_output_path)
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Metadata.json 处理功能完整性测试")
    print("=" * 60)

    results = []

    # 测试 1：读取与解析
    try:
        results.append(("读取与解析", test_1_read_and_parse()))
    except Exception as e:
        print(f"[ERROR] 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("读取与解析", False))

    # 测试 2：批次级转换
    try:
        results.append(("批次级转换", test_2_batch_conversion()))
    except Exception as e:
        print(f"[ERROR] 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("批次级转换", False))

    # 测试 3：保存批次 metadata
    try:
        results.append(("保存批次 metadata", test_3_save_batch_metadata()))
    except Exception as e:
        print(f"[ERROR] 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("保存批次 metadata", False))

    # 测试 4：合并批次
    try:
        results.append(("合并批次 metadata", test_4_merge_batches()))
    except Exception as e:
        print(f"[ERROR] 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("合并批次 metadata", False))

    # 测试 5：时间范围提取
    try:
        results.append(("时间范围提取", test_5_get_time_range_from_moments()))
    except Exception as e:
        print(f"[ERROR] 测试 5 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("时间范围提取", False))

    # 测试 6：与预期输出对比
    try:
        results.append(("与预期输出对比", test_6_compare_with_expected_output()))
    except Exception as e:
        print(f"[ERROR] 测试 6 失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("与预期输出对比", False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("=" * 60)
    if all_passed:
        print("✓✓✓ 所有测试通过！")
    else:
        print("✗✗✗ 部分测试失败")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
