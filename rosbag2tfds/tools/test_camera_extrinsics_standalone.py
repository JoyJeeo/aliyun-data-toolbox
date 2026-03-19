#!/usr/bin/env python3
"""
独立测试脚本：从 rosbag 的 TF 话题中提取相机外参，并检查是否为动态外参。
不依赖任何其他模块，可以直接运行。

依赖:
    - ROS Noetic (需要 rosbag 模块)
    - numpy
    - scipy

用法:
    # 在有 ROS 环境的机器上运行
    source /opt/ros/noetic/setup.bash
    python3 test_camera_extrinsics_standalone.py \
        --bag /path/to/your.bag \
        --camera_links camera,l_hand_camera,r_hand_camera

示例:
    python3 test_camera_extrinsics_standalone.py \
        --bag ~/Desktop/vr_humanoid_20251211_162228.bag \
        --camera_links camera,l_hand_camera,r_hand_camera \
        --base_frame base_link \
        --sample_count 20 \
        --output camera_test_result.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rosbag
from scipy.spatial.transform import Rotation


class SimpleTFBuffer:
    """简化的 TF Buffer，支持链路查询与方向求逆"""
    def __init__(self):
        # 边： (parent, child) -> {timestamp_ns: TransformStamped}
        self.edges = {}
        # 邻接：parent -> set(children)
        self.adj = {}
        # 反向邻接：child -> set(parents)
        self.rev_adj = {}
        self.all_timestamps_list = []

    def add_transform(self, transform_msg, timestamp_ns: int):
        parent = transform_msg.header.frame_id
        child = transform_msg.child_frame_id
        key = (parent, child)
        if key not in self.edges:
            self.edges[key] = {}
        self.edges[key][timestamp_ns] = transform_msg
        self.adj.setdefault(parent, set()).add(child)
        self.rev_adj.setdefault(child, set()).add(parent)
        self.all_timestamps_list.append(timestamp_ns)

    def has_frame(self, frame_id: str) -> bool:
        return frame_id in self.adj or frame_id in self.rev_adj

    def all_frames(self) -> List[str]:
        return sorted(set(list(self.adj.keys()) + list(self.rev_adj.keys())))

    def get_all_timestamps(self) -> np.ndarray:
        if not self.all_timestamps_list:
            return np.array([], dtype=np.int64)
        return np.unique(np.array(self.all_timestamps_list, dtype=np.int64))

    def _nearest_transform(self, key, timestamp_ns: int, tolerance_ns: int, mode: str = "nearest_within"):
        """
        找边 key 在给定时间的 transform。
        mode:
          - "nearest_within": 仅在容差内的最近（原行为）
          - "nearest_any": 不限容差，取最近
          - "latest_before": 取不晚于 timestamp_ns 的最近（常用于静态/低频）
        """
        if key not in self.edges:
            return None
        frame_transforms = self.edges[key]
        if not frame_transforms:
            return None

        chosen_ts = None
        if mode == "latest_before":
            # 选 <= timestamp_ns 的最大 ts；若没有，回退到最小 ts（静态）
            candidates = [ts for ts in frame_transforms.keys() if ts <= timestamp_ns]
            if candidates:
                chosen_ts = max(candidates)
            else:
                chosen_ts = min(frame_transforms.keys())
        elif mode == "nearest_any":
            min_diff = float('inf')
            for ts in frame_transforms.keys():
                diff = abs(ts - timestamp_ns)
                if diff < min_diff:
                    min_diff = diff
                    chosen_ts = ts
        else:  # nearest_within
            closest_ts = None
            min_diff = float('inf')
            for ts in frame_transforms.keys():
                diff = abs(ts - timestamp_ns)
                if diff < min_diff and diff <= tolerance_ns:
                    min_diff = diff
                    closest_ts = ts
            chosen_ts = closest_ts

        if chosen_ts is None:
            return None
        return frame_transforms[chosen_ts]

    @staticmethod
    def _tf_to_matrix(transform_msg):
        t = transform_msg.transform.translation
        r = transform_msg.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat = np.array([r.x, r.y, r.z, r.w], dtype=np.float64)
        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    @staticmethod
    def _invert_matrix(T):
        R = T[:3, :3]
        p = T[:3, 3]
        T_inv = np.eye(4, dtype=np.float64)
        R_inv = R.T
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = -R_inv @ p
        return T_inv

    def _find_path(self, start: str, goal: str) -> Optional[List[Tuple[str, str, bool]]]:
        """
        在 TF 图中寻找从 start 到 goal 的路径。
        返回列表 [(u, v, forward)]：
          - forward=True 表示存在边 (u->v)；
          - forward=False 表示使用反向边 (v->u)，需要求逆。
        """
        from collections import deque
        # BFS
        q = deque()
        q.append(start)
        prev = {start: None}
        # 记录边方向
        edge_dir = {}

        while q:
            u = q.popleft()
            if u == goal:
                break
            # 正向邻居
            for v in self.adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    edge_dir[(u, v)] = True
                    q.append(v)
            # 反向邻居（允许走反向边）
            for v in self.rev_adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    edge_dir[(v, u)] = False  # 我们从 u 到 v 走的是 (v->u) 的逆
                    q.append(v)

        if goal not in prev:
            return None

        # 回溯路径
        path_nodes = []
        cur = goal
        while cur is not None:
            path_nodes.append(cur)
            cur = prev[cur]
        path_nodes.reverse()  # start -> goal

        # 转为边列表附带方向
        path_edges = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            forward = edge_dir.get((u, v))
            if forward is None:
                # 说明走的是反向边
                forward = False
            path_edges.append((u, v, forward))
        return path_edges

    def debug_path(self, target_frame: str, source_frame: str) -> Optional[List[Tuple[str, str, bool]]]:
        path = self._find_path(target_frame, source_frame)
        if not path:
            print(f"   ✖ 没有从 '{target_frame}' 到 '{source_frame}' 的连通路径")
            # 打印 source_frame 的父/子，帮助定位命名问题
            print(f"   🔎 '{source_frame}' 的父节点: {sorted(self.rev_adj.get(source_frame, []))}")
            print(f"   🔎 '{source_frame}' 的子节点: {sorted(self.adj.get(source_frame, []))}")
            return None
        pretty = " -> ".join([target_frame] + [p[1] for p in path])
        print(f"   ✅ 路径: {pretty}  （{'连乘正向/逆向混合' }）")
        return path

    def get_transform(
        self,
        target_frame: str,
        source_frame: str,
        timestamp_ns: int,
        tolerance_ns: int = 100000000,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取从 source_frame 到 target_frame 的变换，支持经由 TF 链路的组合与方向求逆。
        返回 (position, quaternion) 或 None。
        """
        if source_frame == target_frame:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

        # 在图中寻找路径：target_frame -> source_frame（矩阵将按该顺序连乘）
        path = self._find_path(target_frame, source_frame)
        if not path:
            return None

        # 累乘矩阵，从 target 到 source
        T_total = np.eye(4, dtype=np.float64)
        for u, v, forward in path:
            if forward:
                key = (u, v)
                # 先尝试容差内；失败则 latest_before；再失败则 nearest_any
                msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_within")
                if msg is None:
                    msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="latest_before")
                if msg is None:
                    msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_any")
                if msg is None:
                    print(f"   ✖ 边 {u}->{v} 无任何可用变换")
                    return None
                T = self._tf_to_matrix(msg)
            else:
                key = (v, u)
                msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_within")
                if msg is None:
                    msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="latest_before")
                if msg is None:
                    msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_any")
                if msg is None:
                    print(f"   ✖ 边 {v}->{u}（逆向）无任何可用变换")
                    return None
                T = self._invert_matrix(self._tf_to_matrix(msg))
            T_total = T @ T_total

        # 提取 position 与 quaternion
        pos = T_total[:3, 3]
        rot = Rotation.from_matrix(T_total[:3, :3]).as_quat()  # x,y,z,w
        return pos.astype(np.float32), rot.astype(np.float32)


def read_tf_from_bag(bag_path: Path, tf_topic: str = "/tf", tf_static_topic: str = "/tf_static") -> SimpleTFBuffer:
    """从 bag 文件读取 TF 数据（支持 /tf 和 /tf_static）"""
    tf_buffer = SimpleTFBuffer()
    try:
        bag = rosbag.Bag(str(bag_path), 'r')
    except Exception as e:
        print(f"❌ 错误: 无法打开 bag 文件: {e}")
        return tf_buffer

    try:
        topic_info = bag.get_type_and_topic_info()[1]
        available_tf_topics = []
        if (tf_topic in topic_info):
            available_tf_topics.append(tf_topic)
        if (tf_static_topic in topic_info):
            available_tf_topics.append(tf_static_topic)

        if not available_tf_topics:
            print(f"❌ 错误: TF topics '{tf_topic}' 和 '{tf_static_topic}' 都不存在于 bag 文件中")
            print(f"   可用的 topics: {list(topic_info.keys())[:10]}...")
            return tf_buffer

        print(f"📡 找到 TF topics: {available_tf_topics}")
        transform_count = 0

        for _, msg, _ in bag.read_messages(topics=available_tf_topics):
            if hasattr(msg, 'transforms'):
                for transform in msg.transforms:
                    ts_ns = int(transform.header.stamp.to_nsec())
                    tf_buffer.add_transform(transform, ts_ns)
                    transform_count += 1

        print(f"✅ 读取了 {transform_count} 个 TF transforms")
        print(f"   来自 topics: {', '.join(available_tf_topics)}")
    except Exception as e:
        print(f"❌ 错误: 读取 TF 数据失败: {e}")
    finally:
        bag.close()

    return tf_buffer


def get_camera_extrinsics_from_tf(
    tf_buffer: SimpleTFBuffer,
    base_frame: str,
    camera_frame: str,
    timestamp_ns: int,
    tolerance_ns: int = 500_000_000,  # 将容差提升到 0.5s，防止时间对不齐
) -> Optional[Dict]:
    """从 TF buffer 提取相机外参"""
    # 先打印路径便于诊断
    tf_buffer.debug_path(base_frame, camera_frame)
    pose = tf_buffer.get_transform(base_frame, camera_frame, timestamp_ns, tolerance_ns=tolerance_ns)
    if pose is None:
        return None
    
    pos, quat = pose
    
    # 转换四元数到 RPY
    rot = Rotation.from_quat(quat)
    rpy = rot.as_euler('xyz')
    
    # 构建 4x4 变换矩阵
    rot_matrix = rot.as_matrix()
    transform_matrix = [
        [float(rot_matrix[0, 0]), float(rot_matrix[0, 1]), float(rot_matrix[0, 2]), float(pos[0])],
        [float(rot_matrix[1, 0]), float(rot_matrix[1, 1]), float(rot_matrix[1, 2]), float(pos[1])],
        [float(rot_matrix[2, 0]), float(rot_matrix[2, 1]), float(rot_matrix[2, 2]), float(pos[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    
    return {
        "parent_link": base_frame,
        "child_link": camera_frame,
        "xyz": [float(x) for x in pos.tolist()],
        "rpy": [float(x) for x in rpy.tolist()],
        "transform_matrix": transform_matrix,
    }


def parse_camera_links(camera_links_str: Optional[str]) -> List[str]:
    """解析逗号分隔的相机 link 列表"""
    if not camera_links_str:
        return []
    return [link.strip() for link in camera_links_str.split(",") if link.strip()]


def check_extrinsics_dynamic(
    extrinsics_history: List[Dict],
    threshold: float = 0.001,
) -> bool:
    """
    检查外参是否为动态（是否随时间变化）
    """
    if len(extrinsics_history) < 2:
        return False

    positions = [np.array(ext["xyz"]) for ext in extrinsics_history]

    max_delta = 0.0
    for i in range(1, len(positions)):
        delta = float(np.linalg.norm(positions[i] - positions[i-1]))  # 保证为 Python float
        max_delta = max(max_delta, delta)

    return bool(max_delta > threshold)  # 转为原生 bool


def main():
    parser = argparse.ArgumentParser(description="测试从 rosbag TF 话题提取相机外参")
    parser.add_argument("--bag", required=True, help="ROS bag 文件路径")
    parser.add_argument(
        "--camera_links",
        required=True,
        help="相机 link 名称，逗号分隔，例如: camera,l_hand_camera,r_hand_camera"
    )
    parser.add_argument("--base_frame", default="base_link", help="基准坐标系名称")
    parser.add_argument("--tf_topic", default="/tf", help="TF 话题名称")
    parser.add_argument("--tf_static_topic", default="/tf_static", help="TF Static 话题名称")
    parser.add_argument("--sample_count", type=int, default=10, help="采样帧数（用于检测动态外参）")
    parser.add_argument("--output", help="输出 JSON 文件路径（可选）")

    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"❌ 错误: Bag 文件不存在: {bag_path}")
        return 1

    camera_links = parse_camera_links(args.camera_links)
    if not camera_links:
        print("❌ 错误: 未指定相机 link")
        return 1

    print(f"📦 正在读取 bag 文件: {bag_path}")
    print(f"🎯 基准坐标系: {args.base_frame}")
    print(f"📸 相机 links: {camera_links}")
    print(f"📊 采样帧数: {args.sample_count}")
    print()

    # 读取 TF 数据
    print(f"🔍 正在读取 TF 话题: {args.tf_topic} 和 {args.tf_static_topic}")
    tf_buffer = read_tf_from_bag(bag_path, args.tf_topic, args.tf_static_topic)

    if not tf_buffer.all_frames():
        print(f"❌ 错误: TF buffer 为空")
        return 1

    print(f"   可用的 frames: {sorted(tf_buffer.all_frames())}")
    print()

    # 检查每个相机 link
    results = {}

    for camera_link in camera_links:
        print(f"{'='*60}")
        print(f"📸 检查相机: {camera_link}")
        print(f"{'='*60}")

        # 检查 frame 是否存在
        if not tf_buffer.has_frame(camera_link):
            print(f"❌ 错误: Frame '{camera_link}' 不存在于 TF 数据中")
            print(f"   可用的 frames: {sorted(tf_buffer.all_frames())}")
            results[camera_link] = {"status": "not_found"}
            print()
            continue

        # 获取时间范围
        all_timestamps = sorted(tf_buffer.get_all_timestamps())
        if not all_timestamps:
            print(f"❌ 错误: TF buffer 中没有时间戳")
            results[camera_link] = {"status": "no_timestamps"}
            print()
            continue

        print(f"✅ Frame 存在")
        print(f"   时间范围: {all_timestamps[0]} ~ {all_timestamps[-1]} ns")
        print(f"   总帧数: {len(all_timestamps)}")

        # 采样多个时间点提取外参
        sample_indices = np.linspace(0, len(all_timestamps) - 1, min(args.sample_count, len(all_timestamps)), dtype=int)
        extrinsics_history = []

        print(f"\n🔍 采样 {len(sample_indices)} 个时间点提取外参...")

        for idx in sample_indices:
            ts_ns = all_timestamps[idx]
            try:
                extrinsics = get_camera_extrinsics_from_tf(
                    tf_buffer,
                    args.base_frame,
                    camera_link,
                    ts_ns
                )
                if extrinsics:
                    extrinsics_history.append(extrinsics)
            except Exception as e:
                print(f"⚠️  警告: 时间点 {ts_ns} 提取失败: {e}")

        if not extrinsics_history:
            print(f"❌ 错误: 无法提取任何外参数据")
            results[camera_link] = {"status": "extraction_failed"}
            print()
            continue

        print(f"✅ 成功提取 {len(extrinsics_history)} 个时间点的外参")

        # 检查是否为动态外参
        is_dynamic = check_extrinsics_dynamic(extrinsics_history)

        # 显示第一帧的外参
        first_extrinsics = extrinsics_history[0]
        print(f"\n📊 第一帧外参:")
        print(f"   Parent: {first_extrinsics['parent_link']}")
        print(f"   Child:  {first_extrinsics['child_link']}")
        print(f"   XYZ:    {first_extrinsics['xyz']}")
        print(f"   RPY:    {first_extrinsics['rpy']}")

        # 如果是动态的，显示变化范围
        if is_dynamic:
            positions = np.array([ext["xyz"] for ext in extrinsics_history])
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            pos_range = pos_max - pos_min

            print(f"\n🔄 外参类型: 动态 (DYNAMIC)")
            print(f"   位置变化范围:")
            print(f"     X: [{pos_min[0]:.4f}, {pos_max[0]:.4f}] (Δ={pos_range[0]:.4f} m)")
            print(f"     Y: [{pos_min[1]:.4f}, {pos_max[1]:.4f}] (Δ={pos_range[1]:.4f} m)")
            print(f"     Z: [{pos_min[2]:.4f}, {pos_max[2]:.4f}] (Δ={pos_range[2]:.4f} m)")
        else:
            print(f"\n📌 外参类型: 静态 (STATIC)")

        results[camera_link] = {
            "status": "success",
            "is_dynamic": bool(is_dynamic),  # 防御性转换
            "sample_count": int(len(extrinsics_history)),
            "first_extrinsics": {
                # 防止 numpy 类型混入
                "parent_link": str(first_extrinsics["parent_link"]),
                "child_link": str(first_extrinsics["child_link"]),
                "xyz": [float(x) for x in first_extrinsics["xyz"]],
                "rpy": [float(x) for x in first_extrinsics["rpy"]],
                "transform_matrix": [[float(v) for v in row] for row in first_extrinsics["transform_matrix"]],
            },
        }

        print()

    # 输出总结
    print(f"{'='*60}")
    print(f"📋 总结")
    print(f"{'='*60}")

    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    dynamic_count = sum(1 for r in results.values() if r.get("is_dynamic", False))

    print(f"✅ 成功提取: {success_count}/{len(camera_links)}")
    print(f"🔄 动态外参: {dynamic_count}/{success_count if success_count > 0 else len(camera_links)}")

    # 保存结果到 JSON 文件
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {output_path}")

    return 0 if success_count == len(camera_links) else 1


if __name__ == "__main__":
    sys.exit(main())

