#!/usr/bin/env python3
"""分析预期输出的帧数计算逻辑"""

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = __import__("json").load(f)

with open("metadata_out.json", "r", encoding="utf-8") as f:
    expected = __import__("json").load(f)

total_frames = expected["total_frames"]
print(f"Total frames: {total_frames}")

print("\nMarks from metadata.json:")
for i, mark in enumerate(metadata["marks"]):
    sp = mark["startPosition"]
    ep = mark["endPosition"]
    skill = mark["skillAtomic"]
    
    # 我们当前的计算方式
    calc_start = int(sp * total_frames)
    calc_end = int(ep * total_frames)
    
    # 预期的值
    exp_start = expected["label_info"]["action_config"][i]["start_frame"]
    exp_end = expected["label_info"]["action_config"][i]["end_frame"]
    
    print(f"\n{i+1}. {skill}:")
    print(f"   startPosition: {sp:.6f} -> 帧 {calc_start} (预期: {exp_start})")
    print(f"   endPosition:   {ep:.6f} -> 帧 {calc_end} (预期: {exp_end})")
    print(f"   差异: start={exp_start-calc_start}, end={exp_end-calc_end}")

print("\n\n尝试推导预期的计算方式：")
print("观察发现：预期输出可能使用了动作区间的归一化（确保连续覆盖）")
print("而非简单的 fractional position * total_frames")

# 尝试另一种算法：按动作区间比例分配
actions = expected["label_info"]["action_config"]
print(f"\n动作区间分布：")
for i, act in enumerate(actions):
    duration = act["end_frame"] - act["start_frame"]
    ratio = duration / total_frames
    print(f"  {i+1}. {act['skill']}: {act['start_frame']}-{act['end_frame']} ({duration}帧, {ratio*100:.2f}%)")
