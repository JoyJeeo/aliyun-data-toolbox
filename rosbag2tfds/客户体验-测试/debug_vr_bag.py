#!/usr/bin/env python3
import rosbag
import sys

bag_path = "/data/A02-A03-zctest-test_scene-2222222-ceshi04-P4_206-leju_claw-20260227151009-v002.bag"
bag = rosbag.Bag(bag_path, 'r')

# 获取所有话题
topics = bag.get_type_and_topic_info().topics
print("=== 所有包含 ik_fk 的话题 ===")
for topic in sorted(topics.keys()):
    if "ik_fk" in topic.lower():
        info = topics[topic]
        print(f"  {topic}: {info.msg_type} ({info.message_count} msgs)")

# 读取几条 VR 数据
print("\n=== /ik_fk_result/eef_pose 前3条消息 ===")
count = 0
for topic, msg, t in bag.read_messages(topics=["/ik_fk_result/eef_pose"]):
    if count >= 3:
        break
    print(f"[{count}] t={t.to_sec():.3f}")
    lp = getattr(msg, 'left_pose', None)
    rp = getattr(msg, 'right_pose', None)
    if lp:
        print(f"    left_pose.pos_xyz: {list(getattr(lp, 'pos_xyz', []))}")
        print(f"    left_pose.quat_xyzw: {list(getattr(lp, 'quat_xyzw', []))}")
    if rp:
        print(f"    right_pose.pos_xyz: {list(getattr(rp, 'pos_xyz', []))}")
        print(f"    right_pose.quat_xyzw: {list(getattr(rp, 'quat_xyzw', []))}")
    # 打印所有属性名
    print(f"    msg attrs: {[a for a in dir(msg) if not a.startswith('_')]}")
    if lp:
        print(f"    left_pose attrs: {[a for a in dir(lp) if not a.startswith('_')]}")
    count += 1

print("\n=== /ik_fk_result/input_pos 前3条消息 ===")
count = 0
for topic, msg, t in bag.read_messages(topics=["/ik_fk_result/input_pos"]):
    if count >= 3:
        break
    data = list(getattr(msg, 'data', []))
    print(f"[{count}] t={t.to_sec():.3f}, len={len(data)}, data={data[:14]}")
    count += 1

bag.close()
print("\n=== 调试完成 ===")

