#!/usr/bin/env python3
import json

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("metadata_out.json", "r", encoding="utf-8") as f:
    expected = json.load(f)

total_frames = expected["total_frames"]

# 方法1: fractional position (我们当前使用的方式)
print("方法1: fractional position * total_frames")
print("-" * 50)
for i, mark in enumerate(metadata["marks"]):
    sp = mark["startPosition"]
    ep = mark["endPosition"]
    calc_start = int(sp * total_frames)
    calc_end = int(ep * total_frames)
    print(f"{i+1}. {mark['skillAtomic']:12s}: {calc_start:3d}-{calc_end:3d}")

# 方法2: 使用动作持续时间比例来分配帧
print("\n\n方法2: 按持续时间比例分配")
print("-" * 50)
total_duration = sum(m["duration"] for m in metadata["marks"])
print(f"总持续时间: {total_duration:.3f}s")

frame_accum = 0
for i, mark in enumerate(metadata["marks"]):
    ratio = mark["duration"] / total_duration
    frame_count = int(ratio * total_frames)
    start = frame_accum
    end = frame_accum + frame_count
    frame_accum = end
    print(f"{i+1}. {mark['skillAtomic']:12s}: {start:3d}-{end:3d} (持续时间: {mark['duration']:.3f}s, 比例: {ratio*100:.2f}%)")

# 方法3: 线性映射 startPosition/endPosition 到 0-total_frames
print("\n\n方法3: 线性映射 fractional position")
print("-" * 50)
min_pos = min(m["startPosition"] for m in metadata["marks"])
max_pos = max(m["endPosition"] for m in metadata["marks"])
pos_range = max_pos - min_pos

print(f"Position 范围: {min_pos:.6f} - {max_pos:.6f} (range={pos_range:.6f})")

for i, mark in enumerate(metadata["marks"]):
    # 将 [min_pos, max_pos] 映射到 [0, total_frames]
    norm_start = (mark["startPosition"] - min_pos) / pos_range
    norm_end = (mark["endPosition"] - min_pos) / pos_range
    start = int(norm_start * total_frames)
    end = int(norm_end * total_frames)
    print(f"{i+1}. {mark['skillAtomic']:12s}: {start:3d}-{end:3d}")

# 预期输出
print("\n\n预期输出:")
print("-" * 50)
for i, act in enumerate(expected["label_info"]["action_config"]):
    print(f"{i+1}. {act['skill']:12s}: {act['start_frame']:3d}-{act['end_frame']:3d}")
