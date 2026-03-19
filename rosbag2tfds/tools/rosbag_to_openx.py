#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level ROS bag → Open-X RLDS converter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation

# TFRecord helpers (used for TFDS direct writing)
from tensorflow.train import Example

from tools.sidecar_utils import (
    clip_window_cover_all_marks,
    load_sidecar,
    marks_to_step_ranges,
)
from tools.synchronization import build_alignment, build_alignments_batch
from tools.rosbag_reader import read_all_topics
from tools.urdf_utils import extract_camera_extrinsics, extract_joint_order, load_urdf
from tools.drake_fk_utils import create_fk_calculator
from tools.tf_reader import (
    read_tf_from_bag,
    get_tcp_pose_from_tf,
    get_camera_extrinsics_from_tf,
    find_urdf_in_mount_path,
    TFBuffer,
)
import writer as legacy_writer


try:
    rosbag = legacy_writer.rosbag  # Reuse the ROS import from writer.py
except AttributeError as exc:  # pragma: no cover - runtime guard
    raise ImportError("writer.py must expose rosbag import") from exc


CameraInfo = Tuple[List[bytes], np.ndarray, List[str]]


def compute_rotation_delta(quat_current: np.ndarray, quat_previous: np.ndarray) -> np.ndarray:
    """
    Compute rotation delta from two quaternions using optimized NumPy operations.

    This is faster than using scipy.spatial.transform.Rotation because it avoids
    object creation overhead and uses direct quaternion math.

    Args:
        quat_current: Current quaternion [x, y, z, w]
        quat_previous: Previous quaternion [x, y, z, w]

    Returns:
        Rotation delta as euler angles [roll, pitch, yaw] in radians
    """
    if quat_previous is None:
        return np.zeros(3, dtype=np.float32)

    try:
        # Normalize quaternions
        q_curr = quat_current / np.linalg.norm(quat_current)
        q_prev = quat_previous / np.linalg.norm(quat_previous)

        # Ensure quaternions are in the same hemisphere to avoid wrap-around discontinuities
        # q and -q represent the same rotation, but can cause jumps in euler angles
        if np.dot(q_curr, q_prev) < 0:
            q_curr = -q_curr

        # Compute relative quaternion: q_delta = q_current * q_previous.inverse()
        # q_prev.inverse = [-x, -y, -z, w] (for unit quaternion)
        q_prev_inv = np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]], dtype=np.float32)
        
        # Quaternion multiplication: q_curr * q_prev_inv
        # Input quaternions are [x, y, z, w] format
        # Standard quaternion multiplication formula:
        #   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        #   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        #   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        #   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        # Output: [x, y, z, w] to match input format
        x1, y1, z1, w1 = q_curr[0], q_curr[1], q_curr[2], q_curr[3]
        x2, y2, z2, w2 = q_prev_inv[0], q_prev_inv[1], q_prev_inv[2], q_prev_inv[3]

        q_delta = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        ], dtype=np.float32)
        
        # Normalize
        q_delta = q_delta / np.linalg.norm(q_delta)
        
        # Convert quaternion to Euler angles (xyz order)
        # Using direct formula (faster than scipy)
        x, y, z, w = q_delta
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw], dtype=np.float32)
    except Exception as e:
        # Fallback to scipy if NumPy calculation fails
        try:
            rot_current = Rotation.from_quat(quat_current)
            rot_previous = Rotation.from_quat(quat_previous)
            rot_delta = rot_current * rot_previous.inv()
            return rot_delta.as_euler('xyz').astype(np.float32)
        except Exception:
            print(f"Warning: Failed to compute rotation delta: {e}")
            return np.zeros(3, dtype=np.float32)


def ensure_quaternion_continuity(q_current: np.ndarray, q_prev: Optional[np.ndarray]) -> np.ndarray:
    """
    Ensure quaternion continuity by keeping quaternions in the same hemisphere.

    Since q and -q represent the same rotation, we flip the sign if the dot product
    with the previous quaternion is negative to avoid discontinuities.

    Args:
        q_current: Current quaternion [x, y, z, w]
        q_prev: Previous quaternion [x, y, z, w], or None for first frame

    Returns:
        Quaternion with consistent sign relative to q_prev
    """
    if q_prev is None:
        return q_current

    # Compute dot product
    dot = np.dot(q_current, q_prev)

    # If dot product is negative, flip the sign
    if dot < 0:
        return -q_current
    return q_current


@dataclass
class CameraStreamSpec:
    topic: str
    stream_kind: str  # "rgb" or "depth"
    camera: str       # "head", "left", "right", etc.
    role: str         # "primary" or "aux"
    aux_slot: Optional[int] = None

    def key_prefix(self) -> str:
        if self.role == "primary":
            return "observation/image_primary"
        assert self.aux_slot is not None
        return f"observation/image_aux_{self.aux_slot}"


def _parse_topics(topics_str: str) -> List[str]:
    return [t.strip() for t in topics_str.split(",") if t.strip()]


def _infer_camera_name(topic: str) -> str:
    lower = topic.lower()
    if "/cam_h/" in lower:
        return "head"
    if "/cam_l/" in lower:
        return "left"
    if "/cam_r/" in lower:
        return "right"
    return topic.split("/")[-2] if "/" in topic else topic


def _read_camera_streams(
    bag: "rosbag.Bag",
    specs: Sequence[CameraStreamSpec],
) -> Dict[str, Tuple[CameraStreamSpec, CameraInfo]]:
    out: Dict[str, Tuple[CameraStreamSpec, CameraInfo]] = {}
    # Map for quick topic adaptation based on available topics
    available_topics = set(bag.get_type_and_topic_info()[1].keys()) if bag else set()

    def _adapt_topic(topic: str) -> str:
        if topic in available_topics:
            return topic
        if topic.endswith("compressedDepth"):
            alt = topic.replace("compressedDepth", "compressed")
            if alt in available_topics:
                print(f"[camera adapt] {topic} -> {alt} (found in bag)")
                return alt
        if topic.endswith("compressed"):
            alt = topic.replace("compressed", "compressedDepth")
            if alt in available_topics:
                print(f"[camera adapt] {topic} -> {alt} (found in bag)")
                return alt
        return topic

    for spec in specs:
        topic = _adapt_topic(spec.topic)
        if topic != spec.topic:
            spec = CameraStreamSpec(
                topic=topic,
                stream_kind=spec.stream_kind,
                camera=spec.camera,
                role=spec.role,
                aux_slot=spec.aux_slot,
            )
        data, ts, fmt = legacy_writer.read_topic_series(bag, topic)
        out[spec.topic] = (spec, (data, ts, fmt))
        print(f"{spec.topic}: {len(ts)} frames ({spec.stream_kind}) [resolved to {topic}]")
    return out


def _align_streams(
    camera_streams: Dict[str, Tuple[CameraStreamSpec, CameraInfo]],
    main_topic: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    _, (_, ts_main, _) = camera_streams[main_topic]
    if len(ts_main) == 0:
        raise RuntimeError(f"No frames on main topic {main_topic}")
    idx_maps: Dict[str, np.ndarray] = {}
    for topic, (_, (_, ts, _)) in camera_streams.items():
        idx_maps[topic] = build_alignment(
            ts,
            ts_main,
            src_topic=topic,
            dst_topic=main_topic,
        ).indices
    return ts_main, idx_maps


def _resolve_camera_specs(args: argparse.Namespace) -> Tuple[List[CameraStreamSpec], int]:
    rgb_topics = _parse_topics(args.rgb_topics)
    depth_topics = _parse_topics(args.depth_topics)
    specs: List[CameraStreamSpec] = []
    for topic in rgb_topics:
        specs.append(
            CameraStreamSpec(
                topic=topic,
                stream_kind="rgb",
                camera=_infer_camera_name(topic),
                role="primary" if topic == args.main_rgb_topic else "aux",
            )
        )
    for topic in depth_topics:
        specs.append(
            CameraStreamSpec(
                topic=topic,
                stream_kind="depth",
                camera=_infer_camera_name(topic),
                role="aux",
            )
        )
    # Assign aux slots deterministically
    aux_slot = 1
    for spec in specs:
        if spec.role == "aux":
            if aux_slot > args.max_aux_streams:
                raise ValueError(
                    f"Requested {aux_slot} auxiliary camera streams, but --max_aux_streams is {args.max_aux_streams}."
                    " Increase --max_aux_streams to continue."
                )
            spec.aux_slot = aux_slot
            aux_slot += 1
    return specs, aux_slot - 1


def _camera_info_meta(bag: "rosbag.Bag", topics: Sequence[str]) -> Dict[str, dict]:
    info: Dict[str, dict] = {}
    for topic in topics:
        meta = legacy_writer.read_camera_info(bag, topic)
        if meta:
            info[_infer_camera_name(topic)] = meta
    return info


def _parse_camera_link_hints(raw: str) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    if not raw:
        return hints
    for item in raw.split(","):
        if ":" not in item:
            continue
        camera, link = item.split(":", 1)
        hints[camera.strip()] = link.strip()
    return hints


def _default_sidecar_path(bag_path: Path) -> Path:
    candidate = bag_path.with_suffix(".json")
    if candidate.exists():
        return candidate
    return Path("")


def _pick_instruction_text(mark: Dict) -> str:
    return (
        mark.get("detail_en")
        or mark.get("detail_zh")
        or mark.get("skill")
        or mark.get("detail")
        or ""
    )


def _build_step_instructions(marks: List[Dict], num_steps: int) -> List[str]:
    instructions = ["" for _ in range(num_steps)]
    for mark in marks or []:
        text = _pick_instruction_text(mark)
        if not text:
            continue
        start = max(0, int(mark.get("step_start", 0)))
        end = min(num_steps - 1, int(mark.get("step_end", start)))
        for idx in range(start, end + 1):
            instructions[idx] = text
    return instructions


def convert_bag(args: argparse.Namespace, bag_path: Path, *, tfds_writer=None, global_shard_info=None) -> Dict:
    out_root = Path(args.output_root)
    out_dir = out_root / bag_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    sidecar_path = Path(args.sidecar) if args.sidecar else _default_sidecar_path(bag_path)
    sidecar = load_sidecar(sidecar_path if sidecar_path.exists() else None)

    # Default joint names for 28-dim joint array (used when URDF is not available)
    # This matches the hardcoded structure in the robot's data format
    DEFAULT_JOINT_NAMES = [
        # Left leg joints (indices 0-5)
        "l_leg_roll", "l_leg_yaw", "l_leg_pitch", "l_knee", "l_foot_pitch", "l_foot_roll",
        # Right leg joints (indices 6-11)
        "r_leg_roll", "r_leg_yaw", "r_leg_pitch", "r_knee", "r_foot_pitch", "r_foot_roll",
        # Left arm joints (indices 12-18)
        "l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll",
        # Right arm joints (indices 19-25)
        "r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll",
        # Head joints (indices 26-27)
        "head_yaw", "head_pitch",
    ]

    urdf_meta: Dict = {"joint_names": DEFAULT_JOINT_NAMES.copy()}
    camera_extrinsics: Dict[str, Dict] = {}
    # Only load URDF if explicitly specified via --urdf argument
    # Debug: Print URDF argument value to help diagnose unexpected URDF loading
    if args.urdf:
        print(f"Debug: --urdf argument provided: '{args.urdf}'")
    else:
        print(f"Debug: --urdf argument not provided (value: {args.urdf})")

    if args.urdf:
        urdf_path = Path(args.urdf)
        # Priority: 1) Use specified path if exists, 2) Search in /cos/files if not found
        if not urdf_path.exists():
            # If specified path doesn't exist, try to find in mount path
            mount_urdf = find_urdf_in_mount_path(urdf_path.name, mount_path="/cos/files")
            if mount_urdf:
                urdf_path = Path(mount_urdf)
                print(f"Info: URDF not found at specified path, using from mount path: {urdf_path}")
            else:
                print(f"Warning: URDF file not found at specified path '{args.urdf}' and not found in /cos/files")

        if urdf_path.exists():
            try:
                print(f"✓ Loading URDF from: {urdf_path}")
                urdf_tree = load_urdf(urdf_path)
                urdf_joint_names = extract_joint_order(urdf_tree)
                # Only use URDF joint names if they match expected count (28)
                if len(urdf_joint_names) == 28:
                    urdf_meta["joint_names"] = urdf_joint_names
                elif urdf_joint_names:
                    print(f"⚠️  URDF has {len(urdf_joint_names)} joints, expected 28. Using default joint names.")
                camera_extrinsics = extract_camera_extrinsics(
                    urdf_tree, camera_link_hints=_parse_camera_link_hints(args.camera_link_hints)
                )
                if camera_extrinsics:
                    with open(metadata_dir / "camera_extrinsics.json", "w", encoding="utf-8") as f:
                        json.dump(camera_extrinsics, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to load URDF for camera extrinsics: {e}")
        else:
            print(f"Warning: URDF file not found: {args.urdf}. Camera extrinsics will not be extracted.")

    bag_path = bag_path.resolve()
    specs, num_aux_streams = _resolve_camera_specs(args)
    primary_topics = [spec.topic for spec in specs if spec.role == "primary"]
    if not primary_topics:
        raise ValueError("At least one RGB topic must be marked as primary.")
    main_topic = primary_topics[0]

    # Read TF transforms and all topics
    tf_buffer = None
    use_tf = False
    
    # Optimized: Read all topics in a single pass (5-10x faster)
    # Handle unindexed bag files by automatically reindexing
    try:
        bag = rosbag.Bag(str(bag_path), "r")
    except Exception as e:
        # Check if it's an unindexed bag error
        error_msg = str(e).lower()
        if "unindexed" in error_msg or "unindexed" in str(type(e).__name__).lower():
            print(f"⚠️  Bag file is unindexed: {bag_path.name}")
            print(f"   Attempting to reindex...")
            
            # Try to reindex the bag file
            try:
                # rosbag reindex creates a backup (.bag.orig) and reindexes in place
                result = subprocess.run(
                    ["rosbag", "reindex", str(bag_path)],
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    print(f"✅ Successfully reindexed bag file")
                    # Try to open again
                    bag = rosbag.Bag(str(bag_path), "r")
                else:
                    print(f"❌ Failed to reindex bag file:")
                    print(f"   stdout: {result.stdout}")
                    print(f"   stderr: {result.stderr}")
                    raise RuntimeError(f"Failed to reindex bag file: {bag_path}") from e
            except subprocess.TimeoutExpired:
                print(f"❌ Reindex operation timed out (exceeded 1 hour)")
                raise RuntimeError(f"Reindex timeout for bag file: {bag_path}") from e
            except FileNotFoundError:
                print(f"❌ 'rosbag' command not found. Please ensure ROS is properly installed.")
                raise RuntimeError(f"rosbag command not available") from e
            except Exception as reindex_error:
                print(f"❌ Error during reindex: {reindex_error}")
                raise RuntimeError(f"Failed to reindex bag file: {bag_path}") from reindex_error
        else:
            # Re-raise if it's a different error
            raise
    
    # Process the bag file
    with bag:
        # Get camera topics from specs
        camera_topic_list = [spec.topic for spec in specs]
        
        # Single-pass read of all topics
        all_data = read_all_topics(
            bag,
            camera_topics=camera_topic_list,
            joint_cmd_topic=args.joint_cmd_topic,
            sensors_data_raw_topic=args.sensors_data_raw_topic,
            camera_info_topics=_parse_topics(args.camera_info_topics),
        )
        
        # Read TF transforms (if available)
        tf_buffer, tf_timestamps = read_tf_from_bag(
            bag,
            tf_topic=args.tf_topic,
            tf_static_topic=getattr(args, "tf_static_topic", "/tf_static"),
        )

        # Debug: Print available frames in TF buffer
        if len(tf_timestamps) > 0:
            available_frames = tf_buffer.all_frames()
            print(f"Debug: TF buffer contains {len(available_frames)} unique frames")
            print(f"Debug: Looking for TCP frames: left='{args.tcp_frame_left}', right='{args.tcp_frame_right}'")
            print(f"Debug: Has left TCP frame: {tf_buffer.has_frame(args.tcp_frame_left)}")
            print(f"Debug: Has right TCP frame: {tf_buffer.has_frame(args.tcp_frame_right)}")
            
            # Show all frames for debugging (help user find correct TCP frame names)
            if len(available_frames) > 0:
                print(f"\n📋 All available frames in TF buffer ({len(available_frames)} total):")
                sorted_frames = sorted(available_frames)
                # Show all frames, but group by keywords
                end_effector_frames = [f for f in sorted_frames if 'end' in f.lower() or 'effector' in f.lower()]
                left_arm_frames = [f for f in sorted_frames if 'left' in f.lower() or 'l7' in f.lower() or ('l_' in f.lower() and 'arm' in f.lower())]
                right_arm_frames = [f for f in sorted_frames if 'right' in f.lower() or 'r7' in f.lower() or ('r_' in f.lower() and 'arm' in f.lower())]
                zarm_frames = [f for f in sorted_frames if 'zarm' in f.lower()]
                
                if end_effector_frames:
                    print(f"  🔧 End effector frames: {end_effector_frames}")
                if left_arm_frames:
                    print(f"  🤖 Left arm frames: {left_arm_frames}")
                if right_arm_frames:
                    print(f"  🤖 Right arm frames: {right_arm_frames}")
                if zarm_frames:
                    print(f"  ⚙️  Zarm frames: {zarm_frames[:10]}")  # Show first 10
                
                # Show all frames if not too many
                if len(sorted_frames) <= 50:
                    print(f"\n  📝 All frames: {sorted_frames}")
                else:
                    print(f"\n  📝 First 30 frames: {sorted_frames[:30]}")
                    print(f"  ... and {len(sorted_frames) - 30} more frames")
                
                # Suggest possible TCP frames
                print(f"\n💡 TCP Frame 说明:")
                print(f"  TCP = Tool Center Point（工具中心点），即机器人末端执行器的坐标系")
                print(f"  需要找到左臂和右臂的末端执行器 frame 名称")
                print(f"  如果找不到 '{args.tcp_frame_left}' 和 '{args.tcp_frame_right}'，")
                print(f"  请从上面的 frames 列表中选择正确的名称，并通过环境变量设置：")
                print(f"    export TCP_FRAME_LEFT='实际的左臂frame名称'")
                print(f"    export TCP_FRAME_RIGHT='实际的右臂frame名称'")
        
        # Try to find TCP frames (support both *_end_effector and *_link formats)
        tcp_frame_left_actual = args.tcp_frame_left
        tcp_frame_right_actual = args.tcp_frame_right
        
        print(f"\n🔍 TCP Frame 检测逻辑:")
        print(f"  用户传入的参数: left='{args.tcp_frame_left}', right='{args.tcp_frame_right}'")
        print(f"  初始值: tcp_frame_left_actual='{tcp_frame_left_actual}', tcp_frame_right_actual='{tcp_frame_right_actual}'")
        
        if len(tf_timestamps) > 0:
            # Auto-detect TCP frames if default names not found
            has_left = tf_buffer.has_frame(args.tcp_frame_left)
            has_right = tf_buffer.has_frame(args.tcp_frame_right)
            print(f"  TF 中是否存在: left={has_left}, right={has_right}")
            
            if not has_left:
                # Try alternative naming: zarm_l7_link, zarm_l7, etc.
                alternatives = [
                    args.tcp_frame_left.replace("_end_effector", "_link"),
                    args.tcp_frame_left.replace("end_effector", "link"),
                    "zarm_l7_link",
                    "zarm_l7",
                ]
                print(f"  ⚠️  左臂 frame '{args.tcp_frame_left}' 不存在，尝试自动检测...")
                for alt in alternatives:
                    if tf_buffer.has_frame(alt):
                        tcp_frame_left_actual = alt
                        print(f"  ✓ 找到替代左臂 frame: '{alt}' (替代 '{args.tcp_frame_left}')")
                        break
                else:
                    print(f"  ❌ 未找到可用的左臂 frame")
            else:
                print(f"  ✓ 左臂 frame '{args.tcp_frame_left}' 存在，直接使用")
            
            has_right = tf_buffer.has_frame(args.tcp_frame_right)
            if not has_right:
                # Try alternative naming: zarm_r7_link, zarm_r7, etc.
                alternatives = [
                    args.tcp_frame_right.replace("_end_effector", "_link"),
                    args.tcp_frame_right.replace("end_effector", "link"),
                    "zarm_r7_link",
                    "zarm_r7",
                ]
                print(f"  ⚠️  右臂 frame '{args.tcp_frame_right}' 不存在，尝试自动检测...")
                for alt in alternatives:
                    if tf_buffer.has_frame(alt):
                        tcp_frame_right_actual = alt
                        print(f"  ✓ 找到替代右臂 frame: '{alt}' (替代 '{args.tcp_frame_right}')")
                        break
                else:
                    print(f"  ❌ 未找到可用的右臂 frame")
            else:
                print(f"  ✓ 右臂 frame '{args.tcp_frame_right}' 存在，直接使用")
        
        print(f"  最终使用的 frame: left='{tcp_frame_left_actual}', right='{tcp_frame_right_actual}'")
        
        # Use TCP frames (with auto-detection fallback)
        use_tf = (len(tf_timestamps) > 0 and 
                  tf_buffer.has_frame(tcp_frame_left_actual) and 
                  tf_buffer.has_frame(tcp_frame_right_actual))
        if use_tf:
            print(f"✓ TF available: Will use TF for TCP pose calculation")
            print(f"  - Left TCP frame: {tcp_frame_left_actual}")
            print(f"  - Right TCP frame: {tcp_frame_right_actual}")
            print(f"  - Base frame: {args.base_frame}")
        else:
            if len(tf_timestamps) > 0:
                print(f"Info: TF transforms found but missing required TCP frames. Will use FK calculation.")
                print(f"  - Tried left TCP frame: {tcp_frame_left_actual}")
                print(f"  - Tried right TCP frame: {tcp_frame_right_actual}")
            else:
                print(f"Info: TF not available. Will use FK calculation.")
        
        # Extract camera streams (convert format to match existing code)
        # Note: all_data["camera_streams"] uses adapted topic names as keys
        # We need to map back to original spec.topic names
        camera_streams = {}
        for spec in specs:
            topic = spec.topic
            # Try exact match first
            if topic in all_data["camera_streams"]:
                data, ts, fmt = all_data["camera_streams"][topic]
                camera_streams[topic] = (spec, (data, ts, fmt))
            else:
                # Try to find adapted topic (e.g., compressedDepth -> compressed)
                found = False
                for adapted_topic, (data, ts, fmt) in all_data["camera_streams"].items():
                    # Match by topic suffix or camera name
                    if (adapted_topic.endswith(topic.split("/")[-1]) or 
                        topic.endswith(adapted_topic.split("/")[-1]) or
                        _infer_camera_name(adapted_topic) == spec.camera):
                        camera_streams[topic] = (spec, (data, ts, fmt))
                        found = True
                        break
                if not found:
                    # Create empty stream if not found
                    camera_streams[topic] = (spec, ([], np.array([], np.int64), []))
        
        ts_main, idx_maps = _align_streams(camera_streams, main_topic)
        
        # Extract other data
        jc = all_data["joint_cmd"]
        sdr = all_data["sensors_data_raw"]
        lc_pos, lc_vel, lc_ts = all_data["leju_claw_cmd"]
        ls_pos, ls_vel, ls_eff, ls_ts = all_data["leju_claw_state"]
        dx_pos, dx_ts = all_data["dexhand_cmd"]
        ds_pos, ds_vel, ds_eff, ds_ts = all_data["dexhand_state"]

        # VR TCP data (quaternion continuity handled per-frame in the loop, like TCP)
        vr_eef = all_data["vr_eef_pose"]
        vr_input = all_data["vr_input_pos"]

        # Convert camera_info format
        cam_info_map = {}
        for topic in _parse_topics(args.camera_info_topics):
            info = all_data["camera_info"].get(topic)
            if info:
                cam_info_map[_infer_camera_name(topic)] = info

    n_steps = len(ts_main)
    marks = marks_to_step_ranges(sidecar, n_steps)
    step_instructions_full = _build_step_instructions(marks, n_steps)
    global_instruction = (sidecar or {}).get("globalInstruction")
    if global_instruction is None:
        global_instruction = "NULL"
    global_instruction_variants = (sidecar or {}).get("globalInstructionVariants", [])
    # Extract individual variants (up to 3)
    global_instruction_1 = global_instruction_variants[0] if len(global_instruction_variants) > 0 else ""
    global_instruction_2 = global_instruction_variants[1] if len(global_instruction_variants) > 1 else ""
    global_instruction_3 = global_instruction_variants[2] if len(global_instruction_variants) > 2 else ""

    clip_window = None
    if args.clip_to_marks:
        clip_window = clip_window_cover_all_marks(sidecar, n_steps)

    if clip_window:
        valid_indices = list(range(clip_window[0], clip_window[1] + 1))
    else:
        valid_indices = list(range(n_steps))

    step_instruction_for_loop = [step_instructions_full[i] for i in valid_indices]

    episode_id = bag_path.stem
    camera_intrinsics_json = json.dumps(cam_info_map or {}, ensure_ascii=False)
    # Initialize camera_extrinsics_json from URDF (will be updated from TF if available)
    camera_extrinsics_json = json.dumps(camera_extrinsics or {}, ensure_ascii=False)
    # If tfds_writer is provided, skip RLDS writer and write TFDS examples directly.
    if tfds_writer is None:
        writer = legacy_writer.ShardedWriter(
            out_dir,
            dataset_name=args.dataset_name,
            split=args.split,
            shard_bytes=args.shard_bytes,
            compress=args.compress,
            min_shards=args.min_shards,
            max_shards=args.max_shards,
        )
        use_tfds_direct = False
    else:
        writer = tfds_writer  # Must expose .write(example_bytes) and .close()
        use_tfds_direct = True

    def _row(arr: Optional[np.ndarray], idx: Optional[np.ndarray], i_step: int, dim: int) -> np.ndarray:
        """
        Optimized row extraction using NumPy indexing.
        Returns numpy array instead of list for better performance.
        """
        if arr is None or idx is None or len(arr) == 0 or len(idx) == 0:
            return np.zeros(dim, dtype=np.float32)
        j = int(idx[i_step])
        if j < 0 or j >= len(arr):
            return np.zeros(dim, dtype=np.float32)
        row = arr[j].astype(np.float32)
        if len(row) < dim:
            # Pad with zeros
            padded = np.zeros(dim, dtype=np.float32)
            padded[:len(row)] = row
            return padded
        elif len(row) > dim:
            # Truncate
            return row[:dim]
        return row

    # Optimized: Batch align all streams at once (1.2x faster)
    src_ts_dict = {}
    if len(jc["ts"]) > 0:
        src_ts_dict[args.joint_cmd_topic] = jc["ts"]
    if len(sdr["ts"]) > 0:
        src_ts_dict[args.sensors_data_raw_topic] = sdr["ts"]
    if len(lc_ts) > 0:
        src_ts_dict["leju_claw_cmd"] = lc_ts
    if len(ls_ts) > 0:
        src_ts_dict["leju_claw_state"] = ls_ts
    if len(dx_ts) > 0:
        src_ts_dict["dexhand_cmd"] = dx_ts
    if len(ds_ts) > 0:
        src_ts_dict["dexhand_state"] = ds_ts
    if len(vr_eef["ts"]) > 0:
        src_ts_dict["vr_eef_pose"] = vr_eef["ts"]
    if len(vr_input["ts"]) > 0:
        src_ts_dict["vr_input_pos"] = vr_input["ts"]

    batch_alignments = build_alignments_batch(src_ts_dict, ts_main, dst_topic=main_topic)

    idx_joint_action = batch_alignments.get(args.joint_cmd_topic) if len(jc["ts"]) else None
    idx_joint_state = batch_alignments.get(args.sensors_data_raw_topic) if len(sdr["ts"]) else None
    idx_lc = batch_alignments.get("leju_claw_cmd") if len(lc_ts) else None
    idx_ls = batch_alignments.get("leju_claw_state") if len(ls_ts) else None
    idx_dx = batch_alignments.get("dexhand_cmd") if len(dx_ts) else None
    idx_ds = batch_alignments.get("dexhand_state") if len(ds_ts) else None
    idx_vr_eef = batch_alignments.get("vr_eef_pose") if len(vr_eef["ts"]) else None
    idx_vr_input = batch_alignments.get("vr_input_pos") if len(vr_input["ts"]) else None

    eef_type = args.eef
    if eef_type == "auto":
        eef_type = "leju_claw" if max(len(lc_ts), len(ls_ts)) >= max(len(dx_ts), len(ds_ts)) else "dexhand"
    eef_dim = 2 if eef_type == "leju_claw" else 12

    # Initialize FK calculator for TCP pose computation (fallback if TF not available)
    fk_calculator = None
    if not use_tf:
        # Only initialize FK if TF is not available
        if args.urdf:
            urdf_path = Path(args.urdf)
            # Priority: 1) Use specified path if exists, 2) Search in /cos/files if not found
            if not urdf_path.exists():
                mount_urdf = find_urdf_in_mount_path(urdf_path.name, mount_path="/cos/files")
                if mount_urdf:
                    urdf_path = Path(mount_urdf)
                    print(f"Info: URDF not found at specified path, using from mount path: {urdf_path}")
            
            if urdf_path.exists():
                try:
                    fk_calculator = create_fk_calculator(urdf_path, base_link="base_link")
                    if fk_calculator is None:
                        print("Warning: FK calculator initialization failed. TCP pose fields will be set to zeros.")
                    else:
                        print(f"✓ FK calculator initialized successfully from {urdf_path}")
                except Exception as e:
                    print(f"Warning: Failed to create FK calculator: {e}. TCP pose fields will be set to zeros.")
            else:
                print(f"Warning: URDF file not found: {args.urdf}. TCP pose fields will be set to zeros.")
        else:
            print("Info: No URDF path provided. TCP pose fields will be set to zeros.")

    # Pre-compute common strings to avoid repeated encoding
    episode_id_str = legacy_writer._str(episode_id)
    bag_path_str = legacy_writer._str(str(bag_path))
    global_instruction_str = legacy_writer._str(global_instruction)
    global_instruction_1_str = legacy_writer._str(global_instruction_1)
    global_instruction_2_str = legacy_writer._str(global_instruction_2)
    global_instruction_3_str = legacy_writer._str(global_instruction_3)
    camera_intrinsics_json_str = legacy_writer._str(camera_intrinsics_json)
    # Initialize camera_extrinsics_json_str from URDF (will be updated from TF if available in first frame)
    camera_extrinsics_json_str = legacy_writer._str(camera_extrinsics_json)
    camera_extrinsics_from_tf = None
    # Track final camera extrinsics (from TF or URDF) for metadata
    final_camera_extrinsics = camera_extrinsics
    # Track if we've successfully extracted extrinsics (to avoid trying every step)
    final_camera_extrinsics_extracted = False

    # Track previous quaternions for continuity (avoid sign flipping)
    prev_tcp_quat_left: Optional[np.ndarray] = None
    prev_tcp_quat_right: Optional[np.ndarray] = None
    # VR quaternion continuity tracking (same approach as TCP)
    prev_vr_quat_left: Optional[np.ndarray] = None
    prev_vr_quat_right: Optional[np.ndarray] = None
    prev_vr_input_quat_left: Optional[np.ndarray] = None
    prev_vr_input_quat_right: Optional[np.ndarray] = None

    total_steps = 0
    for new_idx, step_idx in enumerate(valid_indices):
        ts_ns = int(ts_main[step_idx])
        is_last_step = (new_idx == len(valid_indices) - 1)
        
        # Optimized: Pre-allocate dict with known size
        feat = {
            "episode_id": episode_id_str,
            "bag_path": bag_path_str,
            "step_index": legacy_writer._int64(new_idx),
            "timestamp": legacy_writer._int64(ts_ns),
            "is_first": legacy_writer._int64(1 if new_idx == 0 else 0),
            "is_last": legacy_writer._int64(1 if is_last_step else 0),
            "reward": legacy_writer._float_list([0.0]),
            "discount": legacy_writer._float_list([1.0]),
        }

        feat["episode_metadata/camera_intrinsics_json"] = camera_intrinsics_json_str
        
        # Get camera extrinsics dynamically from TF if available (each step)
        # Priority: 1) TF (dynamic), 2) URDF (static fallback)
        # Note: Camera extrinsics TF lookup is independent of TCP frame detection (use_tf)
        # We only need tf_buffer to be available and have transforms
        camera_extrinsics_current = None
        if tf_buffer and len(tf_timestamps) > 0:
            # Extract camera frame names from camera_link_hints
            camera_link_hints = _parse_camera_link_hints(args.camera_link_hints)
            camera_extrinsics_from_tf = {}
            for camera_name, camera_frame in camera_link_hints.items():
                try:
                    extrinsics = get_camera_extrinsics_from_tf(
                        tf_buffer, args.base_frame, camera_frame, ts_ns
                    )
                    if extrinsics is not None:
                        camera_extrinsics_from_tf[camera_name] = extrinsics
                except Exception as e:
                    # Fallback to URDF for this camera if TF fails
                    if camera_name in camera_extrinsics:
                        camera_extrinsics_from_tf[camera_name] = camera_extrinsics[camera_name]
                    elif new_idx < 3:  # Only print warning for first few steps
                        print(f"Warning: Failed to get extrinsics for {camera_name} ({camera_frame}) from TF: {e}")
            
            if camera_extrinsics_from_tf:
                camera_extrinsics_current = camera_extrinsics_from_tf
                # Update final camera extrinsics for metadata (use first successful TF values)
                # Try multiple steps until we successfully extract extrinsics (TF may not be available at first step)
                if not final_camera_extrinsics_extracted:
                    print(f"✓ Extracted camera extrinsics from TF for {len(camera_extrinsics_from_tf)} cameras (dynamic) at step {new_idx}")
                    final_camera_extrinsics = camera_extrinsics_from_tf.copy()
                    final_camera_extrinsics_extracted = True
        else:
            # Use URDF extrinsics as fallback (static)
            if camera_extrinsics:
                camera_extrinsics_current = camera_extrinsics
                if new_idx == 0:
                    print(f"✓ Using camera extrinsics from URDF (static fallback)")
        
        # Use current step's extrinsics, or fallback to URDF, or empty JSON
        if camera_extrinsics_current:
            camera_extrinsics_json = json.dumps(camera_extrinsics_current, ensure_ascii=False)
            camera_extrinsics_json_str = legacy_writer._str(camera_extrinsics_json)
        else:
            # Keep the initialized value (URDF or empty)
            pass  # camera_extrinsics_json_str already initialized
        
        feat["episode_metadata/camera_extrinsics_json"] = camera_extrinsics_json_str
        feat["observation/camera_extrinsics_json"] = camera_extrinsics_json_str
        # Natural language instruction (episode-level, same for all steps, stored in each step's observation)
        feat["observation/natural_language_instruction"] = global_instruction_str
        # Natural language instruction variants (episode-level, individual string fields)
        feat["observation/natural_language_instruction_1"] = global_instruction_1_str
        feat["observation/natural_language_instruction_2"] = global_instruction_2_str
        feat["observation/natural_language_instruction_3"] = global_instruction_3_str
        # Subtask language instruction (step-level, dynamic)
        step_lang = step_instruction_for_loop[new_idx] if new_idx < len(step_instruction_for_loop) else ""
        feat["observation/subtask_language_instruction"] = legacy_writer._str(step_lang)

        # Primary + aux images
        for topic, (spec, (data, _, fmt)) in camera_streams.items():
            key_prefix = spec.key_prefix()
            idx = idx_maps[topic]
            frame_idx = int(idx[step_idx]) if len(idx) else None
            if frame_idx is None or frame_idx >= len(data):
                continue
            feat[f"{key_prefix}/encoded"] = legacy_writer._bytes(data[frame_idx])
            feat[f"{key_prefix}/format"] = legacy_writer._str(fmt[frame_idx])
            feat[f"{key_prefix}/camera"] = legacy_writer._str(spec.camera)
            feat[f"{key_prefix}/topic"] = legacy_writer._str(spec.topic)
            feat[f"{key_prefix}/stream_type"] = legacy_writer._str(spec.stream_kind)

        # State / observation (optimized: _row now returns numpy array)
        joint_position = _row(sdr["q"], idx_joint_state, step_idx, 28)
        feat["observation/state/joint_position"] = legacy_writer._float_list(joint_position.tolist())
        feat["observation/state/joint_velocity"] = legacy_writer._float_list(_row(sdr["v"], idx_joint_state, step_idx, 28).tolist())
        feat["observation/state/joint_torque"] = legacy_writer._float_list(_row(sdr["tau"], idx_joint_state, step_idx, 28).tolist())
        
        # Compute TCP pose: Priority 1) TF, 2) FK, 3) zeros
        tcp_pos_left = np.zeros(3, dtype=np.float32)
        tcp_pos_right = np.zeros(3, dtype=np.float32)
        tcp_quat_left = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Identity quaternion
        tcp_quat_right = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Identity quaternion
        
        if use_tf:
            # Priority 1: Use TF if available
            try:
                tcp_left = get_tcp_pose_from_tf(
                    tf_buffer, args.base_frame, tcp_frame_left_actual, ts_ns
                )
                if tcp_left is not None:
                    tcp_pos_left, tcp_quat_left = tcp_left
                elif new_idx < 3:  # Only print warning for first few steps
                    print(f"Warning: Failed to get left TCP pose from TF at step {new_idx} (frame: {tcp_frame_left_actual}, base: {args.base_frame})")
                
                tcp_right = get_tcp_pose_from_tf(
                    tf_buffer, args.base_frame, tcp_frame_right_actual, ts_ns
                )
                if tcp_right is not None:
                    tcp_pos_right, tcp_quat_right = tcp_right
                elif new_idx < 3:  # Only print warning for first few steps
                    print(f"Warning: Failed to get right TCP pose from TF at step {new_idx} (frame: {tcp_frame_right_actual}, base: {args.base_frame})")
            except Exception as e:
                if new_idx < 3:  # Only print error for first few steps
                    print(f"Warning: Failed to get TCP pose from TF at step {new_idx}: {e}")
                # Fallback to FK if TF fails
                if fk_calculator is not None:
                    try:
                        tcp_pos_left, tcp_quat_left = fk_calculator.compute_tcp_pose_left(joint_position)
                        tcp_pos_right, tcp_quat_right = fk_calculator.compute_tcp_pose_right(joint_position)
                        if new_idx < 3:
                            print(f"Info: Using FK fallback for TCP pose at step {new_idx}")
                    except Exception as fk_e:
                        if new_idx < 3:
                            print(f"Warning: FK fallback also failed: {fk_e}")
        elif fk_calculator is not None:
            # Priority 2: Use FK if TF not available
            try:
                # joint_position is already a numpy array from optimized _row()
                # Compute left arm TCP pose
                tcp_pos_left, tcp_quat_left = fk_calculator.compute_tcp_pose_left(joint_position)
                # Compute right arm TCP pose
                tcp_pos_right, tcp_quat_right = fk_calculator.compute_tcp_pose_right(joint_position)
            except Exception as e:
                print(f"Warning: Failed to compute TCP pose at step {step_idx}: {e}")
        # Priority 3: If both fail, use zeros (already set above)

        # Ensure quaternion continuity (keep in same hemisphere as previous)
        tcp_quat_left = ensure_quaternion_continuity(tcp_quat_left, prev_tcp_quat_left)
        tcp_quat_right = ensure_quaternion_continuity(tcp_quat_right, prev_tcp_quat_right)
        prev_tcp_quat_left = tcp_quat_left.copy()
        prev_tcp_quat_right = tcp_quat_right.copy()

        # Add TCP pose to observation
        feat["observation/state/tcp_position_left"] = legacy_writer._float_list(list(tcp_pos_left))
        feat["observation/state/tcp_position_right"] = legacy_writer._float_list(list(tcp_pos_right))
        feat["observation/state/tcp_orientation_left"] = legacy_writer._float_list(list(tcp_quat_left))
        feat["observation/state/tcp_orientation_right"] = legacy_writer._float_list(list(tcp_quat_right))

        # VR TCP pose (observation): 14 floats (left_pos[3] + left_quat[4] + right_pos[3] + right_quat[4])
        # Apply quaternion continuity per-frame (same as TCP)
        vr_tcp_pose_data = np.zeros(14, dtype=np.float32)
        if idx_vr_eef is not None and len(vr_eef["left_pos"]) > 0:
            j = int(idx_vr_eef[step_idx]) if step_idx < len(idx_vr_eef) else -1
            if 0 <= j < len(vr_eef["left_pos"]):
                vr_tcp_pose_data[0:3] = vr_eef["left_pos"][j]
                vr_left_quat = np.array(vr_eef["left_quat"][j], dtype=np.float32)
                vr_tcp_pose_data[7:10] = vr_eef["right_pos"][j]
                vr_right_quat = np.array(vr_eef["right_quat"][j], dtype=np.float32)
                # Ensure quaternion continuity
                vr_left_quat = ensure_quaternion_continuity(vr_left_quat, prev_vr_quat_left)
                vr_right_quat = ensure_quaternion_continuity(vr_right_quat, prev_vr_quat_right)
                prev_vr_quat_left = vr_left_quat.copy()
                prev_vr_quat_right = vr_right_quat.copy()
                vr_tcp_pose_data[3:7] = vr_left_quat
                vr_tcp_pose_data[10:14] = vr_right_quat
        feat["observation/state/vr_tcp_pose"] = legacy_writer._float_list(vr_tcp_pose_data.tolist())

        # VR TCP input pose (action): 14 floats (left[7] + right[7])
        # Apply quaternion continuity per-frame
        vr_tcp_input_pose_data = _row(vr_input["data"], idx_vr_input, step_idx, 14)
        if np.any(vr_tcp_input_pose_data != 0):
            vr_input_left_quat = vr_tcp_input_pose_data[3:7].copy()
            vr_input_right_quat = vr_tcp_input_pose_data[10:14].copy()
            vr_input_left_quat = ensure_quaternion_continuity(vr_input_left_quat, prev_vr_input_quat_left)
            vr_input_right_quat = ensure_quaternion_continuity(vr_input_right_quat, prev_vr_input_quat_right)
            prev_vr_input_quat_left = vr_input_left_quat.copy()
            prev_vr_input_quat_right = vr_input_right_quat.copy()
            vr_tcp_input_pose_data[3:7] = vr_input_left_quat
            vr_tcp_input_pose_data[10:14] = vr_input_right_quat
        feat["action/agent/vr_tcp_input_pose"] = legacy_writer._float_list(vr_tcp_input_pose_data.tolist())

        # Actions from joint_cmd (optimized)
        feat["action/agent/joint_position"] = legacy_writer._float_list(_row(jc["q"], idx_joint_action, step_idx, 28).tolist())
        feat["action/agent/joint_velocity"] = legacy_writer._float_list(_row(jc["v"], idx_joint_action, step_idx, 28).tolist())
        feat["action/agent/joint_torque"] = legacy_writer._float_list(_row(jc["tau"], idx_joint_action, step_idx, 28).tolist())

        # Get current EEF state for Open-X format computation (optimized)
        if eef_type == "leju_claw":
            eef_pos_raw = _row(ls_pos, idx_ls, step_idx, 2)
            eef_pos_current = np.zeros(12, dtype=np.float32)
            eef_pos_current[:2] = eef_pos_raw
            eef_pos_raw_list = eef_pos_raw.tolist()
            
            lc_pos_row = _row(lc_pos, idx_lc, step_idx, 2)
            lc_vel_row = _row(lc_vel, idx_lc, step_idx, 2)
            ls_vel_row = _row(ls_vel, idx_ls, step_idx, 2)
            ls_eff_row = _row(ls_eff, idx_ls, step_idx, 2)
            
            feat["action/agent/eef_position"] = legacy_writer._float_list(lc_pos_row.tolist() + [0.0] * 10)
            feat["action/agent/eef_velocity"] = legacy_writer._float_list(lc_vel_row.tolist() + [0.0] * 10)
            feat["observation/state/eef_position"] = legacy_writer._float_list(eef_pos_raw_list + [0.0] * 10)
            feat["observation/state/eef_velocity"] = legacy_writer._float_list(ls_vel_row.tolist() + [0.0] * 10)
            feat["observation/state/eef_effort"] = legacy_writer._float_list(ls_eff_row.tolist() + [0.0] * 10)
            
            # For leju_claw: position[0] is gripper state (0-100), normalize to [-1, 1]
            if len(eef_pos_raw) > 0:
                gripper_raw = float(eef_pos_raw[0])
                gripper_closedness = np.clip((gripper_raw / 100.0) * 2.0 - 1.0, -1.0, 1.0)
            else:
                gripper_closedness = 0.0
        else:
            eef_pos_current = _row(ds_pos, idx_ds, step_idx, 12)
            dx_pos_row = _row(dx_pos, idx_dx, step_idx, 12)
            ds_vel_row = _row(ds_vel, idx_ds, step_idx, 12)
            ds_eff_row = _row(ds_eff, idx_ds, step_idx, 12)
            
            feat["action/agent/eef_position"] = legacy_writer._float_list(dx_pos_row.tolist())
            feat["action/agent/eef_velocity"] = legacy_writer._float_list([0.0] * 12)
            feat["observation/state/eef_position"] = legacy_writer._float_list(eef_pos_current.tolist())
            feat["observation/state/eef_velocity"] = legacy_writer._float_list(ds_vel_row.tolist())
            feat["observation/state/eef_effort"] = legacy_writer._float_list(ds_eff_row.tolist())
            
            # For dexhand: first joint angle represents gripper state, normalize to [-1, 1]
            if len(eef_pos_current) > 0:
                gripper_raw = float(eef_pos_current[0])
                gripper_closedness = np.clip(gripper_raw / np.pi, -1.0, 1.0)
            else:
                gripper_closedness = 0.0
        
        # Compute Open-X format actions based on TCP pose
        # Customer requirement: a_t = s_(t+1) - s_t (look-ahead, not backward)
        # For last step, use zeros since there's no next step

        if not is_last_step:
            # Get next step's TCP pose from TF
            next_step_idx = valid_indices[new_idx + 1]
            next_ts_ns = int(ts_main[next_step_idx])

            # Initialize next TCP poses
            next_tcp_pos_left = np.zeros(3, dtype=np.float32)
            next_tcp_pos_right = np.zeros(3, dtype=np.float32)
            next_tcp_quat_left = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            next_tcp_quat_right = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if use_tf:
                try:
                    next_tcp_left = get_tcp_pose_from_tf(
                        tf_buffer, args.base_frame, tcp_frame_left_actual, next_ts_ns
                    )
                    if next_tcp_left is not None:
                        next_tcp_pos_left, next_tcp_quat_left = next_tcp_left
                except Exception:
                    pass  # Keep zeros

                try:
                    next_tcp_right = get_tcp_pose_from_tf(
                        tf_buffer, args.base_frame, tcp_frame_right_actual, next_ts_ns
                    )
                    if next_tcp_right is not None:
                        next_tcp_pos_right, next_tcp_quat_right = next_tcp_right
                except Exception:
                    pass  # Keep zeros
            elif fk_calculator is not None:
                # Use FK if TF not available
                next_joint_position = _row(sdr["q"], idx_joint_state, next_step_idx, 28)
                try:
                    next_fk_left = fk_calculator.compute_tcp_pose(next_joint_position, "left")
                    if next_fk_left is not None:
                        next_tcp_pos_left, next_tcp_quat_left = next_fk_left
                except Exception:
                    pass
                try:
                    next_fk_right = fk_calculator.compute_tcp_pose(next_joint_position, "right")
                    if next_fk_right is not None:
                        next_tcp_pos_right, next_tcp_quat_right = next_fk_right
                except Exception:
                    pass

            # Ensure next quaternions are continuous with current
            next_tcp_quat_left = ensure_quaternion_continuity(next_tcp_quat_left, tcp_quat_left)
            next_tcp_quat_right = ensure_quaternion_continuity(next_tcp_quat_right, tcp_quat_right)

            # world_vector = next_tcp - current_tcp
            world_vector_left = (next_tcp_pos_left - tcp_pos_left).astype(np.float32)
            world_vector_right = (next_tcp_pos_right - tcp_pos_right).astype(np.float32)

            # rotation_delta = R_(t+1) * R_t^(-1)
            rotation_delta_left = compute_rotation_delta(next_tcp_quat_left, tcp_quat_left)
            rotation_delta_right = compute_rotation_delta(next_tcp_quat_right, tcp_quat_right)
        else:
            # Last step: no next step, use zeros
            world_vector_left = np.zeros(3, dtype=np.float32)
            world_vector_right = np.zeros(3, dtype=np.float32)
            rotation_delta_left = np.zeros(3, dtype=np.float32)
            rotation_delta_right = np.zeros(3, dtype=np.float32)

        # Main action fields (use left arm for compatibility with single-arm models)
        world_vector = world_vector_left
        rotation_delta = rotation_delta_left

        # gripper_closedness_action: placeholder 0.0 for dexterous hands
        gripper_closedness_action = np.array([0.0], dtype=np.float32)

        # terminate_episode: 1.0 if last step, else 0.0
        terminate_episode = np.array(1.0 if is_last_step else 0.0, dtype=np.float32)

        # Write Open-X format actions (main fields for dataset import/training)
        feat["action/world_vector"] = legacy_writer._float_list(list(world_vector))
        feat["action/rotation_delta"] = legacy_writer._float_list(list(rotation_delta))
        feat["action/gripper_closedness_action"] = legacy_writer._float_list(list(gripper_closedness_action))
        feat["action/terminate_episode"] = legacy_writer._float_list([terminate_episode])

        # Write optional action fields (left/right arm details for user access)
        feat["action/world_vector_left"] = legacy_writer._float_list(list(world_vector_left))
        feat["action/world_vector_right"] = legacy_writer._float_list(list(world_vector_right))
        feat["action/rotation_delta_left"] = legacy_writer._float_list(list(rotation_delta_left))
        feat["action/rotation_delta_right"] = legacy_writer._float_list(list(rotation_delta_right))

        example = tf.train.Example(features=tf.train.Features(feature=feat))
        if use_tfds_direct:
            writer.write(example.SerializeToString())
        else:
            writer.write(example)
        total_steps += 1
        if new_idx % 200 == 0:
            print(f"  wrote {new_idx}/{len(valid_indices)} steps")

    if not use_tfds_direct:
        writer.close()

    if clip_window and marks:
        start, end = clip_window
        clipped_marks = []
        for mark in marks:
            a = max(start, mark["step_start"])
            b = min(end, mark["step_end"])
            if b < a:
                continue
            new_mark = dict(mark)
            new_mark["step_start"] = a - start
            new_mark["step_end"] = b - start
            clipped_marks.append(new_mark)
        marks = clipped_marks

    meta = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "bag_path": str(bag_path),
        "episode_id": episode_id,
        "timeline": args.timeline,
        "main_rgb_topic": args.main_rgb_topic,
        "num_steps": total_steps,
        "rgb_topics": _parse_topics(args.rgb_topics),
        "depth_topics": _parse_topics(args.depth_topics),
        "num_aux_streams": num_aux_streams,
        "joint_cmd_topic": args.joint_cmd_topic,
        "sensors_data_raw_topic": args.sensors_data_raw_topic,
        "eef_type": eef_type,
        "eef_dim": eef_dim,
        "compress": args.compress,
        "shard_bytes": int(args.shard_bytes),
        "num_shards": len(writer.paths),
        "shards": writer.paths,
        "sidecar_meta_path": str(sidecar_path) if sidecar_path and sidecar_path.exists() else None,
        "sidecar_meta": sidecar or None,
        "marks_by_step": marks,
        "global_language_instruction": global_instruction,
        "global_language_instruction_1": global_instruction_1,
        "global_language_instruction_2": global_instruction_2,
        "global_language_instruction_3": global_instruction_3,
        "camera_info": cam_info_map,
        "camera_extrinsics": final_camera_extrinsics or None,
        "joint_names": urdf_meta.get("joint_names"),
    }

    if clip_window:
        meta["clip_window"] = {
            "start_step": int(clip_window[0]),
            "end_step": int(clip_window[1]),
            "original_steps": int(n_steps),
        }

    # Aggregate per-episode summary for easier inspection
    meta["episodes"] = [
        {
            "episode_id": episode_id,
            "bag_path": str(bag_path),
            "num_steps": total_steps,
            "shards": writer.paths if hasattr(writer, "paths") else [],
        }
    ]

    with open(out_dir / "rlds_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    processing_log = {
        "bag": str(bag_path),
        "sidecar": str(sidecar_path) if sidecar_path else None,
        "urdf": args.urdf,
        "command": " ".join(os.sys.argv),
        "total_steps": total_steps,
        "clip_window": meta.get("clip_window"),
        "timeline": args.timeline,
    }
    with open(metadata_dir / "processing_log.json", "w", encoding="utf-8") as f:
        json.dump(processing_log, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {bag_path.name} -> {out_dir}")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert rosbag recordings into Open-X RLDS datasets.")
    ap.add_argument("--bag", help="Path to a single .bag file.")
    ap.add_argument("--bag_dir", default="rosbag", help="Directory to scan for .bag files if --bag is omitted.")
    ap.add_argument("--output_root", default="output/openx", help="Root folder for converted datasets.")
    ap.add_argument("--dataset_name", default="delivery_openx", help="Dataset name used in TFDS shards.")
    ap.add_argument("--split", default="train", help="TFDS split name.")
    ap.add_argument("--timeline", choices=["camera"], default="camera", help="Timeline selection (camera only for now).")
    ap.add_argument("--rgb_topics", default="/cam_h/color/image_raw/compressed,/cam_l/color/image_raw/compressed,/cam_r/color/image_raw/compressed")
    ap.add_argument("--depth_topics", default="/cam_h/depth/image_raw/compressed,/cam_l/depth/image_rect_raw/compressed,/cam_r/depth/image_rect_raw/compressed")
    ap.add_argument("--main_rgb_topic", default="/cam_h/color/image_raw/compressed")
    ap.add_argument("--camera_info_topics", default="/cam_h/color/camera_info,/cam_l/color/camera_info,/cam_r/color/camera_info")
    ap.add_argument("--max_aux_streams", type=int, default=8, help="Maximum auxiliary image streams to encode.")
    ap.add_argument("--joint_cmd_topic", default="/joint_cmd")
    ap.add_argument("--sensors_data_raw_topic", default="/sensors_data_raw")
    ap.add_argument("--sidecar", help="Optional explicit sidecar JSON path.")
    ap.add_argument("--urdf", help="URDF file for camera extrinsics / joint names. Will also search in /cos/files/ if not found.")
    ap.add_argument("--camera_link_hints", default="head:camera,left:l_hand_camera,right:r_hand_camera",
                    help="Comma-separated camera:link mappings for TF/URDF extrinsics. Uses /tf_static frame names.")
    ap.add_argument("--eef", choices=["auto", "leju_claw", "dexhand"], default="auto")
    ap.add_argument("--clip_to_marks", action="store_true", help="Clip episode range based on sidecar marks.")
    # TF-related arguments (can be set via environment variables)
    ap.add_argument("--tf_topic", default=os.getenv("TF_TOPIC", "/tf"), help="TF topic name (default: /tf, or TF_TOPIC env var)")
    ap.add_argument("--tf_static_topic", default=os.getenv("TF_STATIC_TOPIC", "/tf_static"), help="TF static topic name (default: /tf_static, or TF_STATIC_TOPIC env var)")
    ap.add_argument("--base_frame", default=os.getenv("BASE_FRAME", "base_link"), help="Base frame name (default: base_link, or BASE_FRAME env var)")
    ap.add_argument("--tcp_frame_left", default=os.getenv("TCP_FRAME_LEFT", "zarm_l7_end_effector"), 
                    help="Left arm TCP frame name (default: zarm_l7_end_effector, or TCP_FRAME_LEFT env var)")
    ap.add_argument("--tcp_frame_right", default=os.getenv("TCP_FRAME_RIGHT", "zarm_r7_end_effector"),
                    help="Right arm TCP frame name (default: zarm_r7_end_effector, or TCP_FRAME_RIGHT env var)")
    ap.add_argument("--shard_bytes", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--compress", choices=["none", "gzip"], default="none")
    ap.add_argument("--min_shards", type=int, default=1)
    ap.add_argument("--max_shards", type=int, default=100)
    return ap


def discover_bags(args: argparse.Namespace) -> List[Path]:
    if args.bag:
        return [Path(args.bag)]
    bag_dir = Path(args.bag_dir)
    return sorted(bag_dir.glob("*.bag"))


def main(argv: Optional[Sequence[str]] = None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    bag_paths = discover_bags(args)
    if not bag_paths:
        raise FileNotFoundError("No bag files found. Specify --bag or --bag_dir with .bag files.")
    metas = []
    for bag_path in bag_paths:
        meta = convert_bag(args, bag_path)
        metas.append(meta)
    summary_path = Path(args.output_root) / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Summary written to {summary_path}")


if __name__ == "__main__":
    main()
