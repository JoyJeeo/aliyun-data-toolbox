#!/usr/bin/env python3
"""
合并多个 batch 数据为一个连续的 episode
适配 v3 代码结构
"""

import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    print("请先安装 pyarrow: pip install pyarrow")
    exit(1)

logger = logging.getLogger(__name__)


def get_batch_dirs(base_path: Path) -> list:
    """获取所有 batch 目录，按名称排序"""
    batch_dirs = sorted([d for d in base_path.iterdir()
                        if d.is_dir() and d.name.startswith('batch_')])
    return batch_dirs


def merge_parquet_files(batch_dirs: list, output_dir: Path) -> int:
    """合并所有 parquet 文件，返回总帧数"""
    logger.info("[1/3] 合并 Parquet 数据...")

    all_tables = []
    frame_offset = 0
    timestamp_offset = 0.0

    for batch_dir in batch_dirs:
        parquet_path = batch_dir / "data" / "chunk-000" / "episode_000000.parquet"
        if not parquet_path.exists():
            logger.warning("%s 不存在，跳过", parquet_path)
            continue

        table = pq.read_table(parquet_path)
        num_rows = table.num_rows

        # 获取当前 batch 的最大时间戳（用于计算下一个 batch 的偏移）
        max_timestamp = 0.0
        if 'timestamp' in table.column_names:
            timestamps = table.column('timestamp').to_pylist()
            max_timestamp = max(timestamps) if timestamps else 0.0

        logger.info(
            "%s: %s 帧 (frame_offset=%s, ts_offset=%.3fs)",
            batch_dir.name,
            num_rows,
            frame_offset,
            timestamp_offset,
        )

        # 更新 frame_index 列
        if 'frame_index' in table.column_names:
            frame_indices = [i + frame_offset for i in range(num_rows)]
            table = table.set_column(
                table.schema.get_field_index('frame_index'),
                'frame_index',
                pa.array(frame_indices)
            )

        # 更新 index 列
        if 'index' in table.column_names:
            indices = [i + frame_offset for i in range(num_rows)]
            table = table.set_column(
                table.schema.get_field_index('index'),
                'index',
                pa.array(indices)
            )

        # 更新 timestamp 列（添加偏移使时间戳连续）
        if 'timestamp' in table.column_names:
            new_timestamps = [t + timestamp_offset for t in timestamps]
            table = table.set_column(
                table.schema.get_field_index('timestamp'),
                'timestamp',
                pa.array(new_timestamps, type=pa.float32())
            )

        all_tables.append(table)
        frame_offset += num_rows
        # 下一个 batch 的时间戳偏移 = 当前偏移 + 当前最大时间戳 + 一帧间隔(1/30s)
        timestamp_offset += max_timestamp + (1.0 / 30.0)

    # 合并所有表
    if not all_tables:
        raise RuntimeError("没有可合并的 parquet 文件,请检查 batch 目录内容")
    merged_table = pa.concat_tables(all_tables)

    # 保存合并后的 parquet
    output_data_dir = output_dir / "data" / "chunk-000"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_data_dir / "episode_000000.parquet"
    pq.write_table(merged_table, output_parquet)

    logger.info("合并完成: %s 帧 -> %s", merged_table.num_rows, output_parquet)
    return frame_offset


def merge_metadata(batch_dirs: list, output_dir: Path, total_frames: int):
    """合并 metadata.json"""
    logger.info("[2/3] 合并 metadata.json ...")

    # 读取第一个 batch 的 metadata 作为基础
    base_metadata = None
    all_actions = []
    frame_offset = 0

    for batch_dir in batch_dirs:
        metadata_path = batch_dir / "metadata.json"
        parquet_path = batch_dir / "data" / "chunk-000" / "episode_000000.parquet"

        if not metadata_path.exists():
            logger.warning("缺少 batch metadata: %s", metadata_path)
            continue

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if base_metadata is None:
            base_metadata = metadata.copy()

        # 获取当前 batch 的帧数
        num_frames = 0
        if parquet_path.exists():
            table = pq.read_table(parquet_path)
            num_frames = table.num_rows

        # 处理 action_config
        for action in metadata.get('label_info', {}).get('action_config', []):
            action_copy = action.copy()
            # 更新帧范围
            if action_copy.get('start_frame') is not None:
                action_copy['start_frame'] += frame_offset
            if action_copy.get('end_frame') is not None:
                action_copy['end_frame'] += frame_offset
            all_actions.append(action_copy)

        frame_offset += num_frames

    if base_metadata:
        # 合并并去重 action_config（基于 skill 和时间戳），同时优先保留有有效帧范围的标注
        # 如果同一个 (skill, timestamp_utc) 存在多次：
        #   - 如果已有的是 None-None，而新的有 start/end_frame,则用新的替换
        #   - 否则保持第一次出现的
        merged_actions = {}

        def _has_valid_frames(a: dict) -> bool:
            return a.get("start_frame") is not None and a.get("end_frame") is not None

        for action in all_actions:
            key = (action.get('skill'), action.get('timestamp_utc'))
            if key not in merged_actions:
                merged_actions[key] = action
            else:
                old = merged_actions[key]
                # 如果旧的是 None-None，新的是有效帧,则用新覆盖旧
                if (not _has_valid_frames(old)) and _has_valid_frames(action):
                    merged_actions[key] = action

        unique_actions = list(merged_actions.values())

        # ---- 将动作按时间排序，并尽量保证帧区间连续（与原始标注风格一致）----
        if unique_actions:
            # 1) 按 timestamp_utc 排序，保证时间顺序一致
            unique_actions.sort(key=lambda a: a.get("timestamp_utc") or "")

            # 2) 规范第一个动作的起止帧
            first = unique_actions[0]
            # 起始帧缺失则从 0 开始
            if first.get("start_frame") is None:
                first["start_frame"] = 0
            else:
                first["start_frame"] = max(
                    0, min(int(first["start_frame"]), total_frames - 1)
                )
            # 结束帧至少不小于起始帧，且不超过 total_frames-1
            if first.get("end_frame") is None:
                first["end_frame"] = first["start_frame"]
            else:
                first["end_frame"] = max(
                    first["start_frame"],
                    min(int(first["end_frame"]), total_frames - 1),
                )

            # 3) 后续动作：起点强制衔接上一个动作的 end_frame
            for i in range(1, len(unique_actions)):
                prev = unique_actions[i - 1]
                cur = unique_actions[i]

                prev_end = prev.get("end_frame")
                if prev_end is None:
                    prev_end = prev.get("start_frame", 0)
                prev_end = max(0, min(int(prev_end), total_frames - 1))

                cur["start_frame"] = prev_end
                if cur.get("end_frame") is None:
                    cur["end_frame"] = prev_end
                else:
                    cur["end_frame"] = max(
                        prev_end, min(int(cur["end_frame"]), total_frames - 1)
                    )

            # 4) 最后一个动作：结束帧拉到整段 episode 末尾
            last = unique_actions[-1]
            last["end_frame"] = total_frames - 1

        base_metadata['label_info']['action_config'] = unique_actions
        base_metadata['total_frames'] = total_frames

        # 保存
        output_metadata = output_dir / "metadata.json"
        with open(output_metadata, 'w', encoding='utf-8') as f:
            json.dump(base_metadata, f, ensure_ascii=False, indent=4)

        logger.info("metadata 合并完成 -> %s", output_metadata)
        logger.info("共 %s 个动作标注", len(unique_actions))


def merge_meta_files(
    batch_dirs: list,
    output_dir: Path,
    total_frames: int,
    cam_stats: Optional[Dict] = None,
    video_height: Optional[int] = None,
    video_width: Optional[int] = None,
):
    """合并 meta 目录下的文件

    video_height / video_width: 实际保存视频的高和宽，用于 info.json 中相机特征的 shape 与 info。
    若不传则从第一个 batch 的 info.json 中读取；若仍无则默认 480 x 848。
    """
    logger.info("[3/3] 合并 meta 目录下的元数据文件...")

    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    if cam_stats is None:
        cam_stats = {}

    # 1. 合并 episodes.jsonl
    episodes_data = {
        "episode_index": 0,
        "tasks": ["DEBUG"],
        "task_label": "DEBUG",
        "length": total_frames,
        "action_config": None
    }
    with open(output_meta_dir / "episodes.jsonl", 'w', encoding='utf-8') as f:
        f.write(json.dumps(episodes_data, ensure_ascii=False) + '\n')
    logger.info("写入 episodes.jsonl")

    # 2. 合并 info.json（使用第一个 batch 的,更新帧数；相机 shape 使用真实保存视频高宽）
    first_info = None
    for batch_dir in batch_dirs:
        info_path = batch_dir / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                first_info = json.load(f)
            break

    if first_info:
        first_info['total_frames'] = total_frames
        first_info['total_videos'] = 3
        for cam in cam_stats.keys():
            # 优先使用 cam_stats 中记录的实际视频高宽（来自首帧尺寸）
            stats_for_cam = cam_stats.get(cam, {})
            h = stats_for_cam.get("height")
            w = stats_for_cam.get("width")

            # 若 cam_stats 中无尺寸信息,则使用传入的 video_height / video_width
            if h is None:
                h = video_height
            if w is None:
                w = video_width

            # 仍为空则尝试从首个 batch 的 feature 中读取 shape 或 info
            if h is None or w is None:
                existing = first_info.get("features", {}).get(cam, {})
                shape = existing.get("shape")
                info_v = existing.get("info") or {}
                if shape and len(shape) >= 3:
                    if h is None:
                        h = int(shape[1])
                    if w is None:
                        w = int(shape[2])
                if h is None:
                    h = info_v.get("video.height")
                if w is None:
                    w = info_v.get("video.width")

            # 最后兜底：默认 480x848
            if h is None:
                h = 480
            if w is None:
                w = 848

            first_info["features"][cam] = {
                "dtype": "video",
                "shape": [3, h, w],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 30,
                    "video.channels": 3,
                    "has_audio": False
                }
            }
        with open(output_meta_dir / "info.json", 'w', encoding='utf-8') as f:
            json.dump(first_info, f, ensure_ascii=False, indent=4)
        logger.info("写入 info.json (total_frames=%s)", total_frames)

    # 3. 合并 tasks.jsonl
    tasks_path = batch_dirs[0] / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy(tasks_path, output_meta_dir / "tasks.jsonl")
        logger.info("拷贝 tasks.jsonl")

    # 4. 合并 parameters 目录（仅复制第一个 batch 的）
    first_params_dir = batch_dirs[0] / "parameters"
    output_params_dir = output_dir / "parameters"
    if first_params_dir.exists():
        shutil.copytree(first_params_dir, output_params_dir, dirs_exist_ok=True)
        logger.info("拷贝 parameters 目录 -> %s", output_params_dir)


def merge_all_batches(input_dir: str, output_dir: str):
    """合并所有 batch 为一个完整的 episode

    Args:
        input_dir: 包含 batch 目录的输入路径
        output_dir: 输出目录路径
    """
    base_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    logger.info("输入路径: %s", base_path)
    logger.info("输出路径: %s", output_path)

    # 获取所有 batch 目录
    batch_dirs = get_batch_dirs(base_path)
    if not batch_dirs:
        logger.error("未找到 batch 目录")
        return

    logger.info("找到 %s 个 batch 目录:", len(batch_dirs))
    for d in batch_dirs:
        logger.info("  - %s", d.name)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 合并 parquet 数据
    total_frames = merge_parquet_files(batch_dirs, output_path)

    # 合并 metadata.json
    merge_metadata(batch_dirs, output_path, total_frames)

    # 合并 meta 目录
    merge_meta_files(batch_dirs, output_path, total_frames, {})

    logger.info("=" * 50)
    logger.info("合并完成")
    logger.info("输出目录: %s", output_path)
    logger.info("总帧数: %s", total_frames)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description='合并多个 batch 为一个连续的 episode')
    parser.add_argument('--input', '-i', type=str, default='.',
                        help='包含 batch 目录的输入路径 (默认: 当前目录)')
    parser.add_argument('--output', '-o', type=str, default='merged_batch',
                        help='输出目录名称 (默认: merged_batch)')
    args = parser.parse_args()

    merge_all_batches(args.input, args.output)
