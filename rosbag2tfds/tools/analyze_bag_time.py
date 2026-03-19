#!/usr/bin/env python3
import rosbag
import sys

bag_path = sys.argv[1]
bag = rosbag.Bag(bag_path, 'r')

# 获取各话题的时间范围
topic_stats = {}
for topic, msg, t in bag.read_messages():
    ts = t.to_sec()
    if topic not in topic_stats:
        topic_stats[topic] = {'min': ts, 'max': ts, 'count': 0}
    topic_stats[topic]['min'] = min(topic_stats[topic]['min'], ts)
    topic_stats[topic]['max'] = max(topic_stats[topic]['max'], ts)
    topic_stats[topic]['count'] += 1

bag.close()

print('各话题的时间范围：')
print('=' * 100)

# 按话题名排序
for topic in sorted(topic_stats.keys()):
    stats = topic_stats[topic]
    duration = stats['max'] - stats['min']
    min_ts = stats['min']
    max_ts = stats['max']
    count = stats['count']
    print(f"{topic}")
    print(f"  起始: {min_ts:.6f}s, 结束: {max_ts:.6f}s, 时长: {duration:.3f}s, 帧数: {count}")
    print()

