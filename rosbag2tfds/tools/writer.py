#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rlds_writer_v2_3_final.py

- 主时间线默认使用头部主摄（/cam_h/color/image_raw/compressed），可选 --timeline joint
- 读取三路 RGB & 三路 Depth（均为 CompressedImage/CompressedDepth），直接写入压缩字节，不解码
- 读取 /sensors_data_raw（q, v, tau, kp, kd）作为 state（observation）并对齐到主时间线
- 读取 /joint_cmd（q, v, tau, kp, kd）作为 action 并对齐到主时间线
- 自动检测末端类型（夹爪 leju_claw / 灵巧手 dexhand），也可 --eef 强制指定
- 相机内参 /cam_*/color/camera_info 仅写入 rlds_metadata.json（不进入每帧）
- 将 sidecar json（与 bag 同名或通过 --meta 指定）中的 marks 按 position 映射为 step 区间
- 分片写入 TFRecord（可 --compress gzip），输出 rlds_metadata.json 记录所有信息

依赖：
  - ROS Noetic (rosbag)
  - numpy
  - tensorflow (仅用 tf.train.Example/TFRecordWriter)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

# Note: sidecar_utils and synchronization imports removed as they were only used by main().
# writer.py is now used only as a library module by rosbag_to_openx.py.

try:
    import rosbag  # ROS1
except Exception as e:
    rosbag = None
    print("ERROR: rosbag is required; install ROS Noetic. Detail:", e)


# ---------------- TFRecord helpers ----------------
def _int64(v: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _float_list(arr) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in arr]))


def _bytes(b: bytes) -> tf.train.Feature:
    if isinstance(b, memoryview):
        b = b.tobytes()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(b)]))


def _str(s: str) -> tf.train.Feature:
    return _bytes(s.encode("utf-8"))


class ShardedWriter:
    """
    TFDS-compliant sharded TFRecord writer.

    Writes TFRecords with naming convention:
    {dataset_name}-{split}.tfrecord-{shard_index:05d}-of-{total_shards:05d}

    Note: Total shard count is determined after all writes are complete,
    so files are renamed in close() to include the final count.

    Best practices for shard count (as per TFDS guidelines):
    - Use 10-100 shards to balance file size and parallelism
    - Small datasets (< 1GB): 10-20 shards
    - Medium datasets (1-10GB): 20-50 shards
    - Large datasets (> 10GB): 50-100 shards
    """
    def __init__(self, out_dir: Path, dataset_name: str = "rlds_dataset",
                 split: str = "train", shard_bytes: int = 256 * 1024 * 1024,
                 compress: str = "none", min_shards: int = 10, max_shards: int = 100):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.split = split
        self.shard_bytes = int(shard_bytes)
        self.compress = compress.lower()
        self.min_shards = min_shards
        self.max_shards = max_shards
        self._writer = None
        self._cur = 0
        self._idx = 0
        self._temp_paths: List[str] = []  # Temporary paths before renaming
        self._final_paths: List[str] = []  # Final TFDS-compliant paths
        self._total_bytes_written = 0  # Track total bytes for shard calculation

    def _opts(self):
        return tf.io.TFRecordOptions(compression_type="GZIP") if self.compress == "gzip" else None

    def _open(self):
        if self._writer is not None:
            self._writer.close()
        # Use temporary naming during writing
        suffix = ".tfrecord" + (".gz" if self.compress == "gzip" else "")
        temp_path = self.out_dir / f"_temp_{self.dataset_name}-{self.split}-{self._idx:05d}{suffix}"
        self._writer = tf.io.TFRecordWriter(str(temp_path), options=self._opts())
        self._temp_paths.append(str(temp_path))
        self._idx += 1
        self._cur = 0

    def write(self, ex: tf.train.Example):
        b = ex.SerializeToString()
        if (self._writer is None) or (self._cur + len(b) > self.shard_bytes):
            self._open()
        self._writer.write(b)
        # add a small overhead to approximate record framing
        record_size = len(b) + 16
        self._cur += record_size
        self._total_bytes_written += record_size

    def close(self):
        """Close writer and rename files to TFDS-compliant format."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        # Rename temporary files to final TFDS format
        total_shards = len(self._temp_paths)
        if total_shards == 0:
            return

        import os

        # Check if shard count is within recommended range
        if total_shards < self.min_shards:
            print(f"⚠️  Warning: Only {total_shards} shard(s) created, which is less than the recommended minimum of {self.min_shards}.")
            print(f"   Consider using a smaller --shard_bytes value to create more shards.")
            print(f"   Current: {self.shard_bytes / (1024*1024):.0f}MB per shard")
            print(f"   Suggested: {max(1, self._total_bytes_written // (self.min_shards * 1024 * 1024)):.0f}MB per shard")
        elif total_shards > self.max_shards:
            print(f"⚠️  Warning: {total_shards} shard(s) created, which exceeds the recommended maximum of {self.max_shards}.")
            print(f"   Consider using a larger --shard_bytes value to create fewer shards.")
            print(f"   Current: {self.shard_bytes / (1024*1024):.0f}MB per shard")
            print(f"   Suggested: {max(1, self._total_bytes_written // (self.max_shards * 1024 * 1024)):.0f}MB per shard")

        for idx, temp_path in enumerate(self._temp_paths):
            # TFDS format: {dataset_name}-{split}.tfrecord-{shard:05d}-of-{total:05d}
            final_name = f"{self.dataset_name}-{self.split}.tfrecord-{idx:05d}-of-{total_shards:05d}"
            if self.compress == "gzip":
                final_name += ".gz"
            final_path = self.out_dir / final_name

            # Rename temp file to final name
            os.rename(temp_path, str(final_path))
            self._final_paths.append(str(final_path))

        print(f"✅ Created {total_shards} shard(s) in TFDS format: {self.dataset_name}-{self.split}.tfrecord-*-of-{total_shards:05d}")
        print(f"   Total size: {self._total_bytes_written / (1024*1024):.2f} MB")
        print(f"   Average shard size: {self._total_bytes_written / total_shards / (1024*1024):.2f} MB")

    @property
    def paths(self) -> List[str]:
        """Return final TFDS-compliant paths (after close() is called)."""
        return list(self._final_paths) if self._final_paths else list(self._temp_paths)


# ---------------- ROS readers ----------------
def read_topic_series(bag: "rosbag.Bag", topic: str) -> Tuple[List[bytes], np.ndarray, List[str]]:
    """
    Read sensor_msgs/CompressedImage (RGB or depth). Return (bytes_list, ts_ns, fmt_list).

    Returns:
        data:      list[bytes]    原始压缩图像字节
        ts_ns:     np.ndarray     int64 纳秒级时间戳
        fmt_list:  list[str]      "jpeg" for RGB, "png" for depth

    改进点：
    1. 如果脚本里给的是 .../compressedDepth 但 bag 里实际是 .../compressed，
       会自动 fallback 到 .compressed。
    2. 根据 topic 名和 msg.format 自动区分 RGB vs 深度：
       - depth / Depth / 16UC / mono16 -> "png"
       - 其他当作 "jpeg"
    """
    data: List[bytes] = []
    ts_ns: List[int] = []
    fmt: List[str] = []

    # ---------- 自动 fallback: compressedDepth -> compressed ----------
    # 有些包里深度话题实际是 ".../compressed" 而不是 ".../compressedDepth"
    # 我们在这里尝试替换后再读，避免 0 帧的情况
    effective_topic = topic
    try:
        # bag.get_type_and_topic_info() 返回 (types, topics)
        # 其中 topics 是一个 {topic_name: info} 的字典
        available_topics = list(bag.get_type_and_topic_info()[1].keys())
        if effective_topic not in available_topics:
            if effective_topic.endswith("compressedDepth"):
                alt_topic = effective_topic.replace("compressedDepth", "compressed")
                if alt_topic in available_topics:
                    effective_topic = alt_topic
            elif effective_topic.endswith("compressed"):
                alt_topic = effective_topic.replace("compressed", "compressedDepth")
                if alt_topic in available_topics:
                    effective_topic = alt_topic
    except Exception as e:
        # 如果 rosbag 版本差异导致上面失败，不影响主流程
        print(f"  ⚠️ topic fallback check failed for {topic}: {e}")

    # ---------- 正式读取 ----------
    try:
        for top, msg, t in bag.read_messages(topics=[effective_topic]):
            b = getattr(msg, "data", None)
            if b is None:
                continue

            data.append(bytes(b))
            ts_ns.append(int(t.to_nsec()))

            # 自动判断图像属于 RGB 还是 Depth
            # 1) 根据 topic 名判断（含 depth 关键字）
            top_str = str(top)
            is_depth_by_name = ("depth" in top_str.lower()) or ("depth" in effective_topic.lower())

            # 2) 根据压缩图像的 format 字段判断（ROS CompressedImage.msg 里常带，比如 "16UC1", "mono16")
            msg_fmt = getattr(msg, "format", "")
            msg_fmt_lower = str(msg_fmt).lower()
            is_depth_by_fmt = (
                "16uc" in msg_fmt_lower or
                "mono16" in msg_fmt_lower or
                "depth" in msg_fmt_lower or
                "z16" in msg_fmt_lower
            )

            if is_depth_by_name or is_depth_by_fmt:
                fmt.append("png")   # 我们约定深度用 png 表示
            else:
                fmt.append("jpeg")  # RGB 视为 jpeg

    except Exception as e:
        print(f"  ⚠️ read_topic_series({topic}) failed: {e}")

    return data, np.array(ts_ns, np.int64), fmt


def read_joint_cmd(bag: "rosbag.Bag", topic: str = "/joint_cmd") -> Dict[str, np.ndarray]:
    """读取 /joint_cmd 作为 action (joint_q, joint_v, tau)"""
    q = []
    v = []
    tau = []
    tlist = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            q.append(list(getattr(msg, "joint_q", [])))
            v.append(list(getattr(msg, "joint_v", [])))
            tau.append(list(getattr(msg, "tau", [])))
            ts = None
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                try:
                    ts = int(msg.header.stamp.secs) * 1_000_000_000 + int(msg.header.stamp.nsecs)
                except Exception:
                    ts = None
            if ts is None:
                ts = int(t.to_nsec())
            tlist.append(ts)
    except Exception as e:
        print(f"  ⚠️ read_joint_cmd({topic}) failed: {e}")

    return {
        "q": np.array(q, np.float32) if q else np.zeros((0, 28), np.float32),
        "v": np.array(v, np.float32) if v else np.zeros((0, 28), np.float32),
        "tau": np.array(tau, np.float32) if tau else np.zeros((0, 28), np.float32),
        "ts": np.array(tlist, np.int64),
    }


def read_sensors_data_raw(bag: "rosbag.Bag", topic: str = "/sensors_data_raw") -> Dict[str, np.ndarray]:
    """读取 /sensors_data_raw 作为 state (observation) - joint_q, joint_v, joint_torque"""
    q = []
    v = []
    tau = []
    tlist = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            # sensors_data_raw 消息结构: joint_data 包含 joint_q, joint_v, joint_torque
            if hasattr(msg, "joint_data"):
                jd = msg.joint_data
                q.append(list(getattr(jd, "joint_q", [])))
                v.append(list(getattr(jd, "joint_v", [])))
                tau.append(list(getattr(jd, "joint_torque", [])))
            ts = None
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                try:
                    ts = int(msg.header.stamp.secs) * 1_000_000_000 + int(msg.header.stamp.nsecs)
                except Exception:
                    ts = None
            if ts is None:
                ts = int(t.to_nsec())
            tlist.append(ts)
    except Exception as e:
        print(f"  ⚠️ read_sensors_data_raw({topic}) failed: {e}")

    return {
        "q": np.array(q, np.float32) if q else np.zeros((0, 28), np.float32),
        "v": np.array(v, np.float32) if v else np.zeros((0, 28), np.float32),
        "tau": np.array(tau, np.float32) if tau else np.zeros((0, 28), np.float32),
        "ts": np.array(tlist, np.int64),
    }


# 夹爪命令/状态
def read_leju_claw_cmd(bag, topic="/leju_claw_command"):
    pos = []
    vel = []
    ts = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            p = list(getattr(msg, "position", []))
            v = list(getattr(msg, "velocity", [])) if hasattr(msg, "velocity") else []
            if len(p) == 0: p = [0.0, 0.0]
            if len(p) == 1: p = [p[0], 0.0]
            if len(v) == 0: v = [0.0, 0.0]
            if len(v) == 1: v = [v[0], 0.0]
            pos.append(p[:2]); vel.append(v[:2]); ts.append(int(t.to_nsec()))
    except Exception as e:
        print(f"  ⚠️ read_leju_claw_cmd failed: {e}")
    return (np.array(pos, np.float32), np.array(vel, np.float32), np.array(ts, np.int64))


def read_leju_claw_state(bag, topic="/leju_claw_state"):
    pos = []
    vel = []
    eff = []
    ts = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            p = list(getattr(msg, "position", []))
            v = list(getattr(msg, "velocity", []))
            e = list(getattr(msg, "effort", [])) if hasattr(msg, "effort") else []
            if len(p) == 0: p = [0.0, 0.0]
            if len(p) == 1: p = [p[0], 0.0]
            if len(v) == 0: v = [0.0, 0.0]
            if len(v) == 1: v = [v[0], 0.0]
            if len(e) == 0: e = [0.0, 0.0]
            if len(e) == 1: e = [e[0], 0.0]
            pos.append(p[:2]); vel.append(v[:2]); eff.append(e[:2]); ts.append(int(t.to_nsec()))
    except Exception as e:
        print(f"  ⚠️ read_leju_claw_state failed: {e}")
    return (np.array(pos, np.float32), np.array(vel, np.float32), np.array(eff, np.float32), np.array(ts, np.int64))


# 灵巧手命令/状态
def read_dexhand_cmd(bag, topic="/control_robot_hand_position"):
    """读取灵巧手指令 - 只有 position (12,)，没有 velocity 和 effort"""
    pos = []
    ts = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            L = list(getattr(msg, "left_hand_position", []))
            R = list(getattr(msg, "right_hand_position", []))
            L = (L + [0.0] * 6)[:6]; R = (R + [0.0] * 6)[:6]
            pos.append(L + R)
            ts.append(int(t.to_nsec()))
    except Exception as e:
        print(f"  ⚠️ read_dexhand_cmd failed: {e}")
    return (np.array(pos, np.float32), np.array(ts, np.int64))


def read_dexhand_state(bag, topic="/dexhand/state"):
    pos = []
    vel = []
    eff = []
    ts = []
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            p = list(getattr(msg, "position", []))
            v = list(getattr(msg, "velocity", []))
            e = list(getattr(msg, "effort", [])) if hasattr(msg, "effort") else []
            p = (p + [0.0] * 12)[:12]
            v = (v + [0.0] * 12)[:12]
            e = (e + [0.0] * 12)[:12]
            pos.append(p); vel.append(v); eff.append(e); ts.append(int(t.to_nsec()))
    except Exception as e:
        print(f"  ⚠️ read_dexhand_state failed: {e}")
    return (np.array(pos, np.float32), np.array(vel, np.float32), np.array(eff, np.float32), np.array(ts, np.int64))


def read_camera_info(bag: "rosbag.Bag", topic: str) -> Optional[dict]:
    """Read sensor_msgs/CameraInfo once; return dict or None."""
    try:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            return {
                "width": int(getattr(msg, "width", 0)),
                "height": int(getattr(msg, "height", 0)),
                "K": list(getattr(msg, "K", [])),
                "D": list(getattr(msg, "D", [])),
                "R": list(getattr(msg, "R", [])),
                "P": list(getattr(msg, "P", [])),
            }
    except Exception as e:
        print(f"  ⚠️ read_camera_info({topic}) failed: {e}")
    return None


# Note: main() function and _create_tfds_builder() have been removed.
# writer.py is now used only as a library module by rosbag_to_openx.py.
