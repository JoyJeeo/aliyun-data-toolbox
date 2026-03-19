#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-pass conversion: multiple rosbags -> final TFDS dataset (one bag -> one shard).

- 扫描 input_root 下的子目录，每个子目录包含 .bag 和 sidecar .json。
- 每个 bag 解析/对齐后，直接写 TFDS 分片（默认一包一分片），命名遵循 TFDS 规范。
- 最终输出目录：<output_dir>/delivery_openx/1.0.0/{dataset_info,features,metadata}.json + train/*.tfrecord-xxxxx-of-YYYYYY
- 完全离线：禁用 TFDS GCS/下载，不依赖 TFDS Builder 的二次解码。

注意：字段/形状/dtype 依据 leju/tools/tfds_builder_delivery_openx.py 的 _info 定义。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# ============================================================================
# GCS/Google 访问监控系统
# ============================================================================

def _log_gcs_access(func_name: str, *args, **kwargs):
    """记录所有GCS/Google远程访问尝试"""
    import traceback
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # 获取调用栈（跳过当前函数和log函数本身）
    stack = traceback.extract_stack()
    caller_info = []
    for frame in stack[-5:-1]:  # 只显示最近4层调用栈
        caller_info.append(f"  {frame.filename}:{frame.lineno} in {frame.name}")
    
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"⚠️  [GCS ACCESS DETECTED] {timestamp}", file=sys.stderr)
    print(f"   Function: {func_name}", file=sys.stderr)
    if args:
        print(f"   Args: {args}", file=sys.stderr)
    if kwargs:
        print(f"   Kwargs: {kwargs}", file=sys.stderr)
    print(f"   Call stack:", file=sys.stderr)
    print("\n".join(caller_info), file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)
    sys.stderr.flush()

def _wrap_with_logging(original_func: Callable, func_name: str) -> Callable:
    """包装函数以记录所有调用"""
    def wrapped(*args, **kwargs):
        _log_gcs_access(func_name, *args, **kwargs)
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            print(f"   ❌ Exception in {func_name}: {e}", file=sys.stderr)
            raise
    return wrapped

# Force offline TFDS/GCS and silence cloud auth before importing TF/TFDS
os.environ.setdefault("TFDS_DISABLE_DOWNLOAD", "1")
os.environ.setdefault("TFDS_DISABLE_GCS", "1")
os.environ.setdefault("TFDS_GCS_DISABLED", "1")
os.environ.setdefault("TFDS_DATA_DIR", str(Path("~/.tfds").expanduser()))
# 防止 TF 初始化时尝试访问 GCE 元数据 / GCS
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_GCE_CHECK", "1")
os.environ.setdefault("GOOGLE_CLOUD_DISABLE_AUTH", "1")
os.environ.setdefault("GOOGLE_AUTH_DISABLE_GCE_CHECK", "1")
os.environ.setdefault("GCE_METADATA_HOST", "0.0.0.0")
os.environ.setdefault("GCE_METADATA_IP", "0.0.0.0")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
# 尝试禁用 TF 新版 GCS 插件以避免凭证探测
os.environ.setdefault("TF_USE_LEGACY_GCS_IMPLEMENTATION", "1")
os.environ.setdefault("TF_GCS_DISABLE_AUTH", "1")

def _disable_google_auth_http():
    """Monkey-patch google-auth to skip any metadata/GCE probing."""
    try:
        import google.auth.compute_engine._metadata as _metadata  # type: ignore
        
        # 保存原始函数以便监控
        original_ping = _metadata.ping
        original_get = _metadata.get
        
        def logged_ping(*args, **kwargs):
            _log_gcs_access("google.auth.compute_engine._metadata.ping", *args, **kwargs)
            return False
        
        def logged_get(*args, **kwargs):
            _log_gcs_access("google.auth.compute_engine._metadata.get", *args, **kwargs)
            return None
        
        _metadata.ping = logged_ping
        _metadata.get = logged_get
        _metadata._GCE_METADATA_ROOT = "http://0.0.0.0"
        print("✅ Google Auth metadata access monitoring enabled", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Failed to setup google-auth monitoring: {e}", file=sys.stderr)

_disable_google_auth_http()

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import naming

# Monitor TensorFlow GCS file system access
try:
    # Monitor tf.io.gfile operations on GCS paths
    original_gfile_exists = tf.io.gfile.exists
    original_gfile_listdir = tf.io.gfile.listdir
    original_gfile_glob = tf.io.gfile.glob
    original_gfile_stat = tf.io.gfile.stat
    original_gfile_open = tf.io.gfile.GFile.__init__
    
    def logged_gfile_exists(path):
        if isinstance(path, str) and path.startswith('gs://'):
            _log_gcs_access("tf.io.gfile.exists", path=path)
            return False
        return original_gfile_exists(path)
    
    def logged_gfile_listdir(path):
        if isinstance(path, str) and path.startswith('gs://'):
            _log_gcs_access("tf.io.gfile.listdir", path=path)
            return []
        return original_gfile_listdir(path)
    
    def logged_gfile_glob(pattern):
        if isinstance(pattern, str) and pattern.startswith('gs://'):
            _log_gcs_access("tf.io.gfile.glob", pattern=pattern)
            return []
        return original_gfile_glob(pattern)
    
    def logged_gfile_stat(path):
        if isinstance(path, str) and path.startswith('gs://'):
            _log_gcs_access("tf.io.gfile.stat", path=path)
            raise FileNotFoundError(f"GCS path blocked: {path}")
        return original_gfile_stat(path)
    
    def logged_gfile_open(self, name, mode='r'):
        if isinstance(name, str) and name.startswith('gs://'):
            _log_gcs_access("tf.io.gfile.GFile.__init__", path=name, mode=mode)
            raise FileNotFoundError(f"GCS path blocked: {name}")
        return original_gfile_open(self, name, mode)
    
    tf.io.gfile.exists = logged_gfile_exists
    tf.io.gfile.listdir = logged_gfile_listdir
    tf.io.gfile.glob = logged_gfile_glob
    tf.io.gfile.stat = logged_gfile_stat
    tf.io.gfile.GFile.__init__ = logged_gfile_open
    print("✅ TensorFlow GCS file system monitoring enabled", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Failed to setup TensorFlow GCS monitoring: {e}", file=sys.stderr)

# CRITICAL: Disable GCS access at module level BEFORE any TFDS operations
# This ensures GCS is disabled even if DeliveryOpenxBuilder is not used
# Also monitor all GCS access attempts
try:
    from tensorflow_datasets.core.utils import gcs_utils
    # Force-disable GCS access at module level
    if hasattr(gcs_utils, "gcs_disabled"):
        gcs_utils.gcs_disabled = True
    if hasattr(gcs_utils, "is_gcs_disabled"):
        original_is_gcs_disabled = gcs_utils.is_gcs_disabled
        def logged_is_gcs_disabled():
            result = original_is_gcs_disabled() if callable(original_is_gcs_disabled) else True
            if not result:
                _log_gcs_access("gcs_utils.is_gcs_disabled")
            return True  # Always return True (disabled)
        gcs_utils.is_gcs_disabled = logged_is_gcs_disabled
    if hasattr(gcs_utils, "_is_gcs_disabled"):
        gcs_utils._is_gcs_disabled = True
    if hasattr(gcs_utils, "is_dataset_on_gcs"):
        original_is_dataset_on_gcs = gcs_utils.is_dataset_on_gcs
        def logged_is_dataset_on_gcs(*args, **kwargs):
            _log_gcs_access("gcs_utils.is_dataset_on_gcs", *args, **kwargs)
            return False  # Always return False (not on GCS)
        gcs_utils.is_dataset_on_gcs = logged_is_dataset_on_gcs
    # CRITICAL: Also disable gcs_dataset_info_files to prevent initialize_from_bucket() from accessing GCS
    if hasattr(gcs_utils, "gcs_dataset_info_files"):
        original_gcs_dataset_info_files = gcs_utils.gcs_dataset_info_files
        def logged_gcs_dataset_info_files(*args, **kwargs):
            _log_gcs_access("gcs_utils.gcs_dataset_info_files", *args, **kwargs)
            return None  # Return None to indicate no files
        gcs_utils.gcs_dataset_info_files = logged_gcs_dataset_info_files
    if hasattr(gcs_utils, "gcs_dataset_info_path"):
        original_gcs_dataset_info_path = gcs_utils.gcs_dataset_info_path
        def logged_gcs_dataset_info_path(*args, **kwargs):
            _log_gcs_access("gcs_utils.gcs_dataset_info_path", *args, **kwargs)
            return None  # Return None to indicate no path
        gcs_utils.gcs_dataset_info_path = logged_gcs_dataset_info_path
    # Disable exists() check for GCS paths
    if hasattr(gcs_utils, "exists"):
        original_exists = gcs_utils.exists
        def logged_exists(path):
            # If path is a GCS path (starts with gs://), log and return False immediately
            path_str = str(path) if hasattr(path, '__str__') else path
            if isinstance(path_str, str) and path_str.startswith('gs://'):
                _log_gcs_access("gcs_utils.exists", path=path_str)
                return False
            # For non-GCS paths, use original behavior (but still safe)
            try:
                return original_exists(path)
            except Exception as e:
                _log_gcs_access("gcs_utils.exists (exception)", path=path_str, error=str(e))
                return False
        gcs_utils.exists = logged_exists
    print("✅ TFDS GCS access monitoring enabled", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Failed to setup TFDS GCS monitoring: {e}", file=sys.stderr)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sidecar_utils import (
    clip_window_cover_all_marks,
    load_sidecar,
    marks_to_step_ranges,
)
from tools.synchronization import build_alignments_batch
from tools.rosbag_reader import read_all_topics
from tools.urdf_utils import extract_camera_extrinsics, extract_joint_order, load_urdf
from tools.drake_fk_utils import create_fk_calculator
from tools.tf_reader import read_tf_from_bag, get_tcp_pose_from_tf, get_camera_extrinsics_from_tf, find_urdf_in_mount_path
from tools.tfds_builder_delivery_openx import (
    DeliveryOpenxBuilder,
    _infer_png_bit_depth,
)
from tools.rosbag_to_openx import compute_rotation_delta, _build_step_instructions

try:
    import rosbag  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("rosbag module is required") from exc

# Feature schema helper: simplified from tfds_builder_delivery_openx.py
TFDS_VERSION = "1.0.0"
DATASET_NAME = "delivery_openx"


_DUMMY_CACHE: Dict[Tuple[int, str], bytes] = {}


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


def _align_streams(
    camera_streams: Dict[str, Tuple[CameraStreamSpec, Tuple[List[bytes], np.ndarray, List[str]]]],
    main_topic: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Align all camera streams to main topic timeline using nearest-neighbor matching."""
    from tools.synchronization import nearest_indices

    _, (_, ts_main, _) = camera_streams[main_topic]
    ts_maps = {}
    for topic, (_, (_, ts, _)) in camera_streams.items():
        if len(ts) == 0:
            ts_maps[topic] = np.full(len(ts_main), -1, dtype=np.int32)
            continue
        # Use nearest_indices for proper nearest-neighbor alignment
        idx = nearest_indices(ts, ts_main)
        ts_maps[topic] = idx
    return ts_main, ts_maps


def _report_camera_alignment(
    camera_streams: Dict[str, Tuple[CameraStreamSpec, Tuple[List[bytes], np.ndarray, List[str]]]],
    idx_maps: Dict[str, np.ndarray],
    ts_main: np.ndarray,
):
    """Report alignment statistics: RGB↔Depth, RGB↔RGB, Depth↔Depth misalignment."""

    # Separate RGB and Depth streams
    rgb_streams: Dict[str, Tuple[CameraStreamSpec, np.ndarray]] = {}
    depth_streams: Dict[str, Tuple[CameraStreamSpec, np.ndarray]] = {}

    for topic, (spec, (_, ts, _)) in camera_streams.items():
        if spec.stream_kind == "rgb":
            rgb_streams[topic] = (spec, ts)
        elif spec.stream_kind == "depth":
            depth_streams[topic] = (spec, ts)

    def _compute_offsets_ms(ts1: np.ndarray, ts2: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, ts_main: np.ndarray) -> Tuple[float, float, float]:
        """Compute alignment offsets between two streams (in ms)."""
        if len(ts1) == 0 or len(ts2) == 0 or len(ts_main) == 0:
            return 0.0, 0.0, 0.0

        # Get aligned timestamps for each stream
        aligned_ts1 = ts1[np.clip(idx1, 0, len(ts1) - 1)]
        aligned_ts2 = ts2[np.clip(idx2, 0, len(ts2) - 1)]

        # Compute offset between the two streams (in ns)
        offsets_ns = aligned_ts1.astype(np.float64) - aligned_ts2.astype(np.float64)

        median_ms = float(np.median(offsets_ns)) / 1e6
        max_ms = float(np.max(np.abs(offsets_ns))) / 1e6
        mean_ms = float(np.mean(np.abs(offsets_ns))) / 1e6

        return median_ms, max_ms, mean_ms

    print("\n" + "=" * 70)
    print("📊 Camera Stream Alignment Report")
    print("=" * 70)

    # 1. RGB ↔ Depth misalignment (same camera)
    print("\n🔵 RGB ↔ Depth Misalignment (same camera):")
    rgb_depth_pairs = []
    for rgb_topic, (rgb_spec, rgb_ts) in rgb_streams.items():
        for depth_topic, (depth_spec, depth_ts) in depth_streams.items():
            if rgb_spec.camera == depth_spec.camera:
                rgb_depth_pairs.append((rgb_topic, depth_topic, rgb_spec.camera, rgb_ts, depth_ts))

    if rgb_depth_pairs:
        for rgb_topic, depth_topic, camera, rgb_ts, depth_ts in rgb_depth_pairs:
            rgb_idx = idx_maps.get(rgb_topic, np.array([]))
            depth_idx = idx_maps.get(depth_topic, np.array([]))
            if len(rgb_idx) > 0 and len(depth_idx) > 0:
                median_ms, max_ms, mean_ms = _compute_offsets_ms(rgb_ts, depth_ts, rgb_idx, depth_idx, ts_main)
                status = "✅" if max_ms < 50 else ("⚠️" if max_ms < 100 else "❌")
                print(f"   {status} [{camera}] median={median_ms:+.2f}ms, max={max_ms:.2f}ms, mean={mean_ms:.2f}ms")
                print(f"      RGB:   {rgb_topic} ({len(rgb_ts)} frames)")
                print(f"      Depth: {depth_topic} ({len(depth_ts)} frames)")
    else:
        print("   (no RGB-Depth pairs found for same camera)")

    # 2. RGB ↔ RGB misalignment (cross-camera)
    print("\n🟢 RGB ↔ RGB Misalignment (cross-camera):")
    rgb_topics = list(rgb_streams.keys())
    if len(rgb_topics) >= 2:
        for i, topic1 in enumerate(rgb_topics):
            for topic2 in rgb_topics[i+1:]:
                spec1, ts1 = rgb_streams[topic1]
                spec2, ts2 = rgb_streams[topic2]
                idx1 = idx_maps.get(topic1, np.array([]))
                idx2 = idx_maps.get(topic2, np.array([]))
                if len(idx1) > 0 and len(idx2) > 0:
                    median_ms, max_ms, mean_ms = _compute_offsets_ms(ts1, ts2, idx1, idx2, ts_main)
                    status = "✅" if max_ms < 50 else ("⚠️" if max_ms < 100 else "❌")
                    print(f"   {status} [{spec1.camera}] ↔ [{spec2.camera}]: median={median_ms:+.2f}ms, max={max_ms:.2f}ms, mean={mean_ms:.2f}ms")
    else:
        print("   (only one RGB stream, no cross-camera comparison)")

    # 3. Depth ↔ Depth misalignment (cross-camera)
    print("\n🟠 Depth ↔ Depth Misalignment (cross-camera):")
    depth_topics = list(depth_streams.keys())
    if len(depth_topics) >= 2:
        for i, topic1 in enumerate(depth_topics):
            for topic2 in depth_topics[i+1:]:
                spec1, ts1 = depth_streams[topic1]
                spec2, ts2 = depth_streams[topic2]
                idx1 = idx_maps.get(topic1, np.array([]))
                idx2 = idx_maps.get(topic2, np.array([]))
                if len(idx1) > 0 and len(idx2) > 0:
                    median_ms, max_ms, mean_ms = _compute_offsets_ms(ts1, ts2, idx1, idx2, ts_main)
                    status = "✅" if max_ms < 50 else ("⚠️" if max_ms < 100 else "❌")
                    print(f"   {status} [{spec1.camera}] ↔ [{spec2.camera}]: median={median_ms:+.2f}ms, max={max_ms:.2f}ms, mean={mean_ms:.2f}ms")
    else:
        print("   (fewer than 2 depth streams, no cross-camera comparison)")

    # 4. Summary: alignment to main timeline
    print("\n📐 Alignment to Main Timeline:")
    for topic, (spec, ts) in list(rgb_streams.items()) + list(depth_streams.items()):
        idx = idx_maps.get(topic, np.array([]))
        if len(ts) > 0 and len(idx) > 0 and len(ts_main) > 0:
            aligned_ts = ts[np.clip(idx, 0, len(ts) - 1)]
            offsets_ns = aligned_ts.astype(np.float64) - ts_main.astype(np.float64)
            median_ms = float(np.median(offsets_ns)) / 1e6
            max_ms = float(np.max(np.abs(offsets_ns))) / 1e6
            status = "✅" if max_ms < 50 else ("⚠️" if max_ms < 100 else "❌")
            print(f"   {status} [{spec.camera}/{spec.stream_kind}] → main: median={median_ms:+.2f}ms, max={max_ms:.2f}ms ({len(ts)} frames)")

    print("=" * 70 + "\n")


def _build_camera_specs(args) -> Tuple[CameraStreamSpec, List[CameraStreamSpec]]:
    rgb_topics = _parse_topics(args.rgb_topics)
    depth_topics = _parse_topics(args.depth_topics)
    main_topic = args.main_rgb_topic
    specs: List[CameraStreamSpec] = []
    main_spec = None
    for topic in rgb_topics:
        spec = CameraStreamSpec(
            topic=topic,
            stream_kind="rgb",
            camera=_infer_camera_name(topic),
            role="primary" if topic == main_topic else "aux",
            aux_slot=None,
        )
        if spec.role == "primary":
            main_spec = spec
        specs.append(spec)
    aux_slot = 1
    for spec in specs:
        if spec.role == "aux":
            spec.aux_slot = aux_slot
            aux_slot += 1
    for topic in depth_topics:
        specs.append(
            CameraStreamSpec(
                topic=topic,
                stream_kind="depth",
                camera=_infer_camera_name(topic),
                role="aux",
                aux_slot=aux_slot,
            )
        )
        aux_slot += 1
    if main_spec is None:
        main_spec = CameraStreamSpec(topic=main_topic, stream_kind="rgb", camera="head", role="primary")
    aux_specs = [s for s in specs if s.role == "aux"]
    return main_spec, aux_specs


def _ensure_permissions(root: Path):
    """Ensure generated files/dirs are writable even if created by root."""
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        try:
            dpath.chmod(0o777)
        except Exception:
            pass
        for name in dirnames:
            try:
                (dpath / name).chmod(0o777)
            except Exception:
                pass
        for name in filenames:
            try:
                (dpath / name).chmod(0o666)
            except Exception:
                pass


def _dummy_image(channels: int, dtype: tf.dtypes.DType, encoding_format: str = "png") -> bytes:
    """Return a cached 1x1 image with the requested channel/dtype/format."""
    key = (channels, dtype.name, encoding_format)
    if key not in _DUMMY_CACHE:
        arr = tf.zeros((1, 1, channels), dtype=dtype)
        if encoding_format == "jpeg":
            # JPEG only supports uint8, cast if needed.
            if dtype != tf.uint8:
                arr = tf.cast(arr, tf.uint8)
            _DUMMY_CACHE[key] = tf.image.encode_jpeg(arr).numpy()
        else:
            _DUMMY_CACHE[key] = tf.image.encode_png(arr).numpy()
    return _DUMMY_CACHE[key]


def _bit_depth(encoded: bytes, fmt: str, stream_kind: str) -> int:
    if not encoded:
        return 0
    fmt_lower = (fmt or "").lower()
    if "png" in fmt_lower or stream_kind == "depth":
        return int(_infer_png_bit_depth(encoded))
    return 8


def _row(arr: Optional[np.ndarray], idx: Optional[np.ndarray], step_idx: int, dim: int) -> np.ndarray:
    if arr is None or idx is None or len(arr) == 0 or len(idx) == 0:
        return np.zeros(dim, dtype=np.float32)
    j = int(idx[step_idx])
    if j < 0 or j >= len(arr):
        return np.zeros(dim, dtype=np.float32)
    row = np.asarray(arr[j], dtype=np.float32)
    if row.shape[0] < dim:
        padded = np.zeros(dim, dtype=np.float32)
        padded[: row.shape[0]] = row
        return padded
    return row[:dim]


def _get_vr_tcp_pose(
    vr_eef: Dict,
    idx_vr_eef: Optional[np.ndarray],
    step_idx: int,
    prev_vr_quat_left: Optional[np.ndarray] = None,
    prev_vr_quat_right: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get VR TCP pose as 14-dim array with quaternion continuity.

    Returns:
        Tuple of (result[14], left_quat[4], right_quat[4]) for tracking previous quaternions.
        When data is invalid, returns zeros but preserves the previous quaternion tracking.
    """
    result = np.zeros(14, dtype=np.float32)
    # When data is invalid, preserve the previous tracking quaternions (don't reset)
    default_left = prev_vr_quat_left if prev_vr_quat_left is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    default_right = prev_vr_quat_right if prev_vr_quat_right is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    if idx_vr_eef is None or len(vr_eef.get("left_pos", [])) == 0:
        return result, default_left, default_right
    if step_idx >= len(idx_vr_eef):
        return result, default_left, default_right
    j = int(idx_vr_eef[step_idx])
    if j < 0 or j >= len(vr_eef["left_pos"]):
        return result, default_left, default_right

    result[0:3] = vr_eef["left_pos"][j]
    left_quat = np.array(vr_eef["left_quat"][j], dtype=np.float32)
    result[7:10] = vr_eef["right_pos"][j]
    right_quat = np.array(vr_eef["right_quat"][j], dtype=np.float32)

    # Ensure quaternion continuity (same as TCP handling)
    left_quat = ensure_quaternion_continuity(left_quat, prev_vr_quat_left)
    right_quat = ensure_quaternion_continuity(right_quat, prev_vr_quat_right)

    result[3:7] = left_quat
    result[10:14] = right_quat

    return result, left_quat, right_quat


def _get_vr_input_pose(
    vr_input: Dict,
    idx_vr_input: Optional[np.ndarray],
    step_idx: int,
    prev_input_quat_left: Optional[np.ndarray] = None,
    prev_input_quat_right: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get VR input pose as 14-dim array with quaternion continuity.

    Returns:
        Tuple of (result[14], left_quat[4], right_quat[4]) for tracking previous quaternions.
        When data is invalid, returns zeros but preserves the previous quaternion tracking.
    """
    result = np.zeros(14, dtype=np.float32)
    # When data is invalid, preserve the previous tracking quaternions (don't reset)
    default_left = prev_input_quat_left if prev_input_quat_left is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    default_right = prev_input_quat_right if prev_input_quat_right is not None else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    if idx_vr_input is None or len(vr_input.get("data", [])) == 0:
        return result, default_left, default_right
    if step_idx >= len(idx_vr_input):
        return result, default_left, default_right
    j = int(idx_vr_input[step_idx])
    if j < 0 or j >= len(vr_input["data"]):
        return result, default_left, default_right

    data = np.array(vr_input["data"][j], dtype=np.float32)
    result[:] = data

    left_quat = data[3:7].copy()
    right_quat = data[10:14].copy()

    # Ensure quaternion continuity (same as TCP handling)
    left_quat = ensure_quaternion_continuity(left_quat, prev_input_quat_left)
    right_quat = ensure_quaternion_continuity(right_quat, prev_input_quat_right)

    result[3:7] = left_quat
    result[10:14] = right_quat

    return result, left_quat, right_quat


def _adapt_camera_streams(
    all_data: Dict,
    specs: List[CameraStreamSpec],
) -> Dict[str, Tuple[CameraStreamSpec, Tuple[List[bytes], np.ndarray, List[str]]]]:
    """Map requested specs to available topics (compressed <-> compressedDepth fallback)."""
    available_topics = set(all_data.get("camera_streams", {}).keys())

    def _adapt(topic: str) -> str:
        if topic in available_topics:
            return topic
        if topic.endswith("compressedDepth"):
            alt = topic.replace("compressedDepth", "compressed")
            if alt in available_topics:
                return alt
        if topic.endswith("compressed"):
            alt = topic.replace("compressed", "compressedDepth")
            if alt in available_topics:
                return alt
        return topic

    camera_streams: Dict[str, Tuple[CameraStreamSpec, Tuple[List[bytes], np.ndarray, List[str]]]] = {}
    for spec in specs:
        topic = _adapt(spec.topic)
        data_tuple = all_data["camera_streams"].get(topic, ([], np.array([], np.int64), []))
        camera_streams[spec.topic] = (spec, data_tuple)
    return camera_streams


def _auto_detect_tcp_frames(tf_buffer) -> Tuple[str, str]:
    """Heuristically pick TCP frame names from TF buffer (best-effort)."""
    if tf_buffer is None:
        return "", ""
    frames = tf_buffer.all_frames()
    left = next((f for f in frames if "zarm_l7" in f and "end_effector" in f), "")
    if not left:
        left = next((f for f in frames if "zarm_l7" in f and "link" in f), "")
    right = next((f for f in frames if "zarm_r7" in f and "end_effector" in f), "")
    if not right:
        right = next((f for f in frames if "zarm_r7" in f and "link" in f), "")
    return left, right


def ensure_quaternion_continuity(q_current: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    """
    Ensure quaternion continuity by keeping quaternions in the same hemisphere.

    Since q and -q represent the same rotation, we flip the sign if the dot product
    with the previous quaternion is negative to avoid discontinuities.

    Args:
        q_current: Current quaternion [x, y, z, w]
        q_prev: Previous quaternion [x, y, z, w]

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


def _strip_png_padding(data: bytes) -> Tuple[bytes, int]:
    sig = b"\x89PNG\r\n\x1a\n"
    if not data:
        return data, 0
    idx = data.find(sig)
    if idx > 0:
        return data[idx:], idx
    return data, 0


def _reencode_image(spec: CameraStreamSpec, encoded: bytes, fmt: str) -> Tuple[bytes, str]:
    """Decode + re-encode to canonical PNG/JPEG so TF decode_image never fails."""
    if encoded is None or len(encoded) == 0:
        dtype = tf.uint16 if spec.stream_kind == "depth" else tf.uint8
        default_fmt = "png" if spec.stream_kind == "depth" else "jpeg"
        return _dummy_image(1 if spec.stream_kind == "depth" else 3, dtype, default_fmt), default_fmt
    fmt_lower = (fmt or "").lower()
    try:
        if spec.stream_kind == "depth" or "depth" in fmt_lower or "png" in fmt_lower:
            data, _ = _strip_png_padding(encoded)
            dtype = tf.uint16 if spec.stream_kind == "depth" else tf.uint8
            decoded = tf.io.decode_png(data, dtype=dtype, channels=0)
            arr = decoded
            if len(arr.shape) == 2:
                arr = tf.expand_dims(arr, -1)
            if len(arr.shape) == 3 and arr.shape[-1] > 1:
                arr = arr[..., :1]
            encoded_new = tf.image.encode_png(arr)
            return encoded_new.numpy(), "png"
        else:
            # treat as RGB
            decoded = tf.image.decode_image(encoded, channels=3, expand_animations=False)
            encoded_new = tf.image.encode_png(decoded)  # png for safety
            return encoded_new.numpy(), "png"
    except Exception:
        dtype = tf.uint16 if spec.stream_kind == "depth" else tf.uint8
        default_fmt = "png" if spec.stream_kind == "depth" else "jpeg"
        return _dummy_image(1 if spec.stream_kind == "depth" else 3, dtype, default_fmt), default_fmt



def _episode_from_bag(
    args,
    bag_dir: Path,
    main_spec: CameraStreamSpec,
    aux_specs: List[CameraStreamSpec],
) -> Tuple[List[Dict], Dict, Dict]:
    bag_files = list(bag_dir.glob("*.bag"))
    if not bag_files:
        raise ValueError(f"No .bag file found in directory: {bag_dir}")
    bag_path = bag_files[0]
    sidecars = list(bag_dir.glob("*.json"))
    sidecar_path = sidecars[0] if sidecars else None
    sidecar = load_sidecar(sidecar_path) if sidecar_path else {}

    bag = rosbag.Bag(str(bag_path), "r")
    tf_buffer, tf_timestamps = read_tf_from_bag(
        bag,
        getattr(args, "tf_topic", "/tf"),
        getattr(args, "tf_static_topic", "/tf_static"),
    )
    all_data = read_all_topics(
        bag,
        camera_topics=[main_spec.topic] + [s.topic for s in aux_specs],
        joint_cmd_topic=args.joint_cmd_topic,
        sensors_data_raw_topic=args.sensors_data_raw_topic,
        camera_info_topics=_parse_topics(args.camera_info_topics),
    )
    bag.close()

    camera_streams = _adapt_camera_streams(all_data, [main_spec] + aux_specs)
    ts_main, idx_maps = _align_streams(camera_streams, main_spec.topic)
    n_steps = len(ts_main)
    if n_steps == 0:
        raise ValueError(f"No frames found for {bag_path}")

    # Report camera alignment statistics
    _report_camera_alignment(camera_streams, idx_maps, ts_main)

    jc = all_data["joint_cmd"]
    sdr = all_data["sensors_data_raw"]
    lc_pos, lc_vel, lc_ts = all_data["leju_claw_cmd"]
    ls_pos, ls_vel, ls_eff, ls_ts = all_data["leju_claw_state"]
    dx_pos, dx_ts = all_data["dexhand_cmd"]
    ds_pos, ds_vel, ds_eff, ds_ts = all_data["dexhand_state"]

    # VR TCP data (quaternion continuity handled per-frame in the loop, like TCP)
    vr_eef = all_data["vr_eef_pose"]
    vr_input = all_data["vr_input_pos"]

    cam_info_map = {}
    for topic in _parse_topics(args.camera_info_topics):
        info = all_data["camera_info"].get(topic)
        if info:
            cam_info_map[_infer_camera_name(topic)] = info

    marks = marks_to_step_ranges(sidecar, n_steps)
    step_instructions_full = _build_step_instructions(marks, n_steps)
    global_instruction = sidecar.get("globalInstruction", "NULL")
    global_instruction_variants = sidecar.get("globalInstructionVariants", [])
    # Extract individual variants (up to 3)
    global_instruction_1 = global_instruction_variants[0] if len(global_instruction_variants) > 0 else ""
    global_instruction_2 = global_instruction_variants[1] if len(global_instruction_variants) > 1 else ""
    global_instruction_3 = global_instruction_variants[2] if len(global_instruction_variants) > 2 else ""
    clip_window = clip_window_cover_all_marks(sidecar, n_steps) if args.clip_to_marks else None
    if clip_window:
        valid_indices = list(range(clip_window[0], clip_window[1] + 1))
    else:
        valid_indices = list(range(n_steps))
    step_instruction_for_loop = [step_instructions_full[i] for i in valid_indices]

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
    batch_alignments = build_alignments_batch(src_ts_dict, ts_main, dst_topic=main_spec.topic)
    idx_joint_action = batch_alignments.get(args.joint_cmd_topic)
    idx_joint_state = batch_alignments.get(args.sensors_data_raw_topic)
    idx_lc = batch_alignments.get("leju_claw_cmd")
    idx_ls = batch_alignments.get("leju_claw_state")
    idx_dx = batch_alignments.get("dexhand_cmd")
    idx_ds = batch_alignments.get("dexhand_state")
    idx_vr_eef = batch_alignments.get("vr_eef_pose") if len(vr_eef["ts"]) else None
    idx_vr_input = batch_alignments.get("vr_input_pos") if len(vr_input["ts"]) else None

    eef_type = args.eef
    if eef_type == "auto":
        eef_type = "leju_claw" if max(len(lc_ts), len(ls_ts)) >= max(len(dx_ts), len(ds_ts)) else "dexhand"
    eef_dim = 2 if eef_type == "leju_claw" else 12

    camera_extrinsics = {}
    # Default joint names for 28-dim joint array (used when URDF is not available)
    # This matches the joint names in biped_s49.urdf
    DEFAULT_JOINT_NAMES = [
        # Left leg joints (indices 0-5)
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        # Right leg joints (indices 6-11)
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        # Left arm joints (indices 12-18)
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        # Right arm joints (indices 19-25)
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
        # Head joints (indices 26-27)
        "zhead_1_joint", "zhead_2_joint",
    ]
    joint_names: List[str] = DEFAULT_JOINT_NAMES.copy()

    # Parse camera_link_hints once (used for both URDF and TF)
    # Supports both JSON format: {"head":"camera","left":"l_hand_camera"}
    # and comma-separated format: head:camera,left:l_hand_camera
    camera_link_hints = args.camera_link_hints
    if isinstance(camera_link_hints, str):
        # Try JSON first
        try:
            camera_link_hints = json.loads(camera_link_hints)
        except Exception:
            # Fallback to comma-separated format
            hints = {}
            for item in camera_link_hints.split(","):
                if ":" in item:
                    camera, link = item.split(":", 1)
                    hints[camera.strip()] = link.strip()
            camera_link_hints = hints
    if not isinstance(camera_link_hints, dict):
        camera_link_hints = {}

    if args.urdf:
        urdf_path = Path(args.urdf)
        if not urdf_path.exists():
            mount_urdf = find_urdf_in_mount_path(urdf_path.name, mount_path="/cos/files")
            if mount_urdf:
                urdf_path = Path(mount_urdf)
        if urdf_path.exists():
            urdf_tree = load_urdf(urdf_path)
            urdf_joint_names = extract_joint_order(urdf_tree)
            # Only use URDF joint names if they match expected count (28)
            if len(urdf_joint_names) == 28:
                joint_names = urdf_joint_names
            elif urdf_joint_names:
                print(f"⚠️  URDF has {len(urdf_joint_names)} joints, expected 28. Using default joint names.")
            camera_extrinsics = extract_camera_extrinsics(
                urdf_tree,
                camera_link_hints=camera_link_hints or {},
            )

    # Print joint names for verification
    print(f"\n📋 Joint Names ({len(joint_names)} joints):")
    for i, name in enumerate(joint_names):
        print(f"   [{i:2d}] {name}")
    print()

    fk_calc = None
    if args.urdf:
        urdf_path = Path(args.urdf)
        if not urdf_path.exists():
            mount_urdf = find_urdf_in_mount_path(urdf_path.name, mount_path="/cos/files")
            if mount_urdf:
                urdf_path = Path(mount_urdf)
        if urdf_path.exists():
            fk_calc = create_fk_calculator(urdf_path, args.base_frame)

    # TCP frame selection: use user args first; if missing in TF, try auto-detect fallback
    tcp_frame_left = args.tcp_frame_left
    tcp_frame_right = args.tcp_frame_right
    has_left = tf_buffer.has_frame(tcp_frame_left) if tf_buffer else False
    has_right = tf_buffer.has_frame(tcp_frame_right) if tf_buffer else False
    if tf_buffer and (not has_left or not has_right):
        auto_left, auto_right = _auto_detect_tcp_frames(tf_buffer)
        if not has_left and auto_left:
            print(f"⚠️  Left TCP frame '{tcp_frame_left}' not found, fallback to '{auto_left}'")
            tcp_frame_left = auto_left
            has_left = True
        if not has_right and auto_right:
            print(f"⚠️  Right TCP frame '{tcp_frame_right}' not found, fallback to '{auto_right}'")
            tcp_frame_right = auto_right
            has_right = True
    use_tf = (len(tf_timestamps) > 0 and has_left and has_right)

    steps: List[Dict] = []
    prev_frames: Dict[str, Tuple[bytes, str]] = {spec.topic: None for spec in [main_spec] + aux_specs}
    # Track final camera extrinsics (from TF or URDF) for metadata
    final_camera_extrinsics = camera_extrinsics
    final_camera_extrinsics_extracted = False
    # Cache bit_depth results to avoid repeated PNG header parsing
    # Key: (encoded_bytes, fmt, stream_kind) - all parameters that affect bit_depth calculation
    bit_depth_cache: Dict[Tuple[bytes, str, str], int] = {}

    def _tcp_from_tf(ts_ns: int, joint_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tcp_pos_left = np.zeros(3, dtype=np.float32)
        tcp_pos_right = np.zeros(3, dtype=np.float32)
        tcp_quat_left = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        tcp_quat_right = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if use_tf:
            try:
                left = get_tcp_pose_from_tf(tf_buffer, args.base_frame, tcp_frame_left, ts_ns)
                right = get_tcp_pose_from_tf(tf_buffer, args.base_frame, tcp_frame_right, ts_ns)
                if left is not None:
                    tcp_pos_left, tcp_quat_left = left
                if right is not None:
                    tcp_pos_right, tcp_quat_right = right
            except Exception:
                pass
        elif fk_calc is not None:
            try:
                tcp_pos_left, tcp_quat_left = fk_calc.compute_tcp_pose_left(joint_position, tcp_link=args.tcp_frame_left)
                tcp_pos_right, tcp_quat_right = fk_calc.compute_tcp_pose_right(joint_position, tcp_link=args.tcp_frame_right)
            except Exception:
                pass
        return tcp_pos_left, tcp_quat_left, tcp_pos_right, tcp_quat_right

    def _encode_image(spec: CameraStreamSpec, step_idx: int) -> Dict:
        data_list, _, fmt_list = camera_streams.get(spec.topic, (spec, ([], np.array([], np.int64), [])))[1]
        j = idx_maps.get(spec.topic, np.array([], np.int64))
        frame_idx = int(j[step_idx]) if len(j) else -1
        encoded: bytes
        fmt: str
        if 0 <= frame_idx < len(data_list):
            encoded = data_list[frame_idx]
            fmt = fmt_list[frame_idx] if frame_idx < len(fmt_list) else ("png" if spec.stream_kind == "depth" else "jpeg")
            fmt_lower = (fmt or "").lower()
            # Depth/compressedDepth often has padding before PNG signature; strip it but keep original encoding.
            if ("png" in fmt_lower) or ("compresseddepth" in fmt_lower) or (spec.stream_kind == "depth"):
                encoded, _ = _strip_png_padding(encoded)
            if encoded is not None and len(encoded) > 0:
                prev_frames[spec.topic] = (encoded, fmt)
        else:
            if prev_frames[spec.topic] is not None:
                encoded, fmt = prev_frames[spec.topic]
            else:
                dtype = tf.uint16 if spec.stream_kind == "depth" else tf.uint8
                fmt = "png" if spec.stream_kind == "depth" else "jpeg"
                encoded = _dummy_image(1 if spec.stream_kind == "depth" else 3, dtype, fmt)
        
        # Cache bit_depth to avoid repeated PNG header parsing for the same image
        # Include stream_kind in cache key since it affects the calculation
        cache_key = (encoded, fmt, spec.stream_kind)
        if cache_key not in bit_depth_cache:
            bit_depth_cache[cache_key] = _bit_depth(encoded, fmt, spec.stream_kind)
        bit_depth = bit_depth_cache[cache_key]
        
        return {
            "image": encoded,
            "bit_depth": bit_depth,
            "format": fmt,
            "camera": spec.camera,
            "topic": spec.topic,
            "stream_type": spec.stream_kind,
        }

    # Track previous quaternions for continuity (avoid sign flipping)
    prev_tcp_quat_left: Optional[np.ndarray] = None
    prev_tcp_quat_right: Optional[np.ndarray] = None
    # VR quaternion continuity tracking (same approach as TCP)
    prev_vr_quat_left: Optional[np.ndarray] = None
    prev_vr_quat_right: Optional[np.ndarray] = None
    prev_vr_input_quat_left: Optional[np.ndarray] = None
    prev_vr_input_quat_right: Optional[np.ndarray] = None

    for new_idx, step_idx in enumerate(valid_indices):
        ts_ns = int(ts_main[step_idx])
        joint_position = _row(sdr["q"], idx_joint_state, step_idx, 28)

        tcp_pos_left, tcp_quat_left, tcp_pos_right, tcp_quat_right = _tcp_from_tf(ts_ns, joint_position)

        # Ensure quaternion continuity (keep in same hemisphere as previous)
        tcp_quat_left = ensure_quaternion_continuity(tcp_quat_left, prev_tcp_quat_left)
        tcp_quat_right = ensure_quaternion_continuity(tcp_quat_right, prev_tcp_quat_right)
        prev_tcp_quat_left = tcp_quat_left.copy()
        prev_tcp_quat_right = tcp_quat_right.copy()

        # VR TCP pose with quaternion continuity (same approach as TCP)
        vr_tcp_pose, prev_vr_quat_left, prev_vr_quat_right = _get_vr_tcp_pose(
            vr_eef, idx_vr_eef, step_idx, prev_vr_quat_left, prev_vr_quat_right
        )
        # VR input pose with quaternion continuity
        vr_input_pose, prev_vr_input_quat_left, prev_vr_input_quat_right = _get_vr_input_pose(
            vr_input, idx_vr_input, step_idx, prev_vr_input_quat_left, prev_vr_input_quat_right
        )

        # Calculate world_vector as next_tcp - current_tcp (customer requirement)
        # Check if there's a next step
        is_last = new_idx == len(valid_indices) - 1
        if not is_last:
            # Get next step's TCP position
            next_step_idx = valid_indices[new_idx + 1]
            next_ts_ns = int(ts_main[next_step_idx])
            next_joint_position = _row(sdr["q"], idx_joint_state, next_step_idx, 28)
            next_tcp_pos_left, next_tcp_quat_left, next_tcp_pos_right, next_tcp_quat_right = _tcp_from_tf(next_ts_ns, next_joint_position)

            # Ensure next quaternions are also continuous with current
            next_tcp_quat_left = ensure_quaternion_continuity(next_tcp_quat_left, tcp_quat_left)
            next_tcp_quat_right = ensure_quaternion_continuity(next_tcp_quat_right, tcp_quat_right)

            # world_vector = next_tcp - current_tcp
            world_vector_left = (next_tcp_pos_left - tcp_pos_left).astype(np.float32)
            world_vector_right = (next_tcp_pos_right - tcp_pos_right).astype(np.float32)
            rotation_delta_left = compute_rotation_delta(next_tcp_quat_left, tcp_quat_left)
            rotation_delta_right = compute_rotation_delta(next_tcp_quat_right, tcp_quat_right)
        else:
            # Last step: no next step, use zero vector
            world_vector_left = np.zeros(3, dtype=np.float32)
            world_vector_right = np.zeros(3, dtype=np.float32)
            rotation_delta_left = np.zeros(3, dtype=np.float32)
            rotation_delta_right = np.zeros(3, dtype=np.float32)

        world_vector = world_vector_left
        rotation_delta = rotation_delta_left

        if eef_type == "leju_claw":
            eef_pos_raw = _row(ls_pos, idx_ls, step_idx, 2)
            eef_pos_current = np.zeros(12, dtype=np.float32)
            eef_pos_current[:2] = eef_pos_raw
            lc_pos_row = _row(lc_pos, idx_lc, step_idx, 2)
            lc_vel_row = _row(lc_vel, idx_lc, step_idx, 2)
            ls_vel_row = _row(ls_vel, idx_ls, step_idx, 2)
            ls_eff_row = _row(ls_eff, idx_ls, step_idx, 2)
            action_eef_position = np.concatenate([lc_pos_row, np.zeros(10, dtype=np.float32)])
            action_eef_velocity = np.concatenate([lc_vel_row, np.zeros(10, dtype=np.float32)])
            obs_eef_vel = np.concatenate([ls_vel_row, np.zeros(10, dtype=np.float32)])
            obs_eef_eff = np.concatenate([ls_eff_row, np.zeros(10, dtype=np.float32)])
        else:
            eef_pos_current = _row(ds_pos, idx_ds, step_idx, 12)
            action_eef_position = _row(dx_pos, idx_dx, step_idx, 12)
            action_eef_velocity = np.zeros(12, dtype=np.float32)
            obs_eef_vel = _row(ds_vel, idx_ds, step_idx, 12)
            obs_eef_eff = _row(ds_eff, idx_ds, step_idx, 12)
        
        # Get camera extrinsics dynamically from TF if available (each step)
        # Priority: 1) TF (dynamic), 2) URDF (static fallback)
        # Note: Camera extrinsics TF lookup is independent of TCP frame detection (use_tf)
        # We only need tf_buffer to be available and have transforms
        camera_extrinsics_current = None
        if tf_buffer and len(tf_timestamps) > 0 and camera_link_hints:
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
                    final_camera_extrinsics = camera_extrinsics_from_tf.copy()
                    final_camera_extrinsics_extracted = True
        else:
            # Use URDF extrinsics as fallback (static)
            if camera_extrinsics:
                camera_extrinsics_current = camera_extrinsics
        
        # Use current step's extrinsics, or fallback to URDF, or empty dict
        if camera_extrinsics_current:
            camera_extrinsics_json_str = json.dumps(camera_extrinsics_current, ensure_ascii=False)
        else:
            # Fallback to URDF or empty
            camera_extrinsics_json_str = json.dumps(camera_extrinsics, ensure_ascii=False)
        
        obs = {
            "image_primary": _encode_image(main_spec, step_idx),
            "state": {
                "joint_position": joint_position,
                "joint_velocity": _row(sdr["v"], idx_joint_state, step_idx, 28),
                "joint_torque": _row(sdr["tau"], idx_joint_state, step_idx, 28),
                "eef_position": eef_pos_current,
                "eef_velocity": obs_eef_vel,
                "eef_effort": obs_eef_eff,
                "tcp_position_left": tcp_pos_left.astype(np.float32),
                "tcp_position_right": tcp_pos_right.astype(np.float32),
                "tcp_orientation_left": tcp_quat_left.astype(np.float32),
                "tcp_orientation_right": tcp_quat_right.astype(np.float32),
                "vr_tcp_pose": vr_tcp_pose,
            },
            "timestamp": ts_ns,
            "natural_language_instruction": global_instruction,
            "natural_language_instruction_1": global_instruction_1,
            "natural_language_instruction_2": global_instruction_2,
            "natural_language_instruction_3": global_instruction_3,
            "subtask_language_instruction": step_instruction_for_loop[new_idx] if new_idx < len(step_instruction_for_loop) else "",
            "camera_extrinsics_json": camera_extrinsics_json_str,
        }
        for spec in aux_specs:
            obs[f"image_aux_{spec.aux_slot}"] = _encode_image(spec, step_idx)

        action = {
            "agent": {
                "joint_position": _row(jc["q"], idx_joint_action, step_idx, 28),
                "joint_velocity": _row(jc["v"], idx_joint_action, step_idx, 28),
                "joint_torque": _row(jc["tau"], idx_joint_action, step_idx, 28),
                "eef_position": action_eef_position,
                "eef_velocity": action_eef_velocity,
                "vr_tcp_input_pose": vr_input_pose,
            },
            "world_vector": world_vector,
            "rotation_delta": rotation_delta,
            "world_vector_left": world_vector_left,
            "world_vector_right": world_vector_right,
            "rotation_delta_left": rotation_delta_left,
            "rotation_delta_right": rotation_delta_right,
            "gripper_closedness_action": np.array([0.0], dtype=np.float32),
            "terminate_episode": bool(is_last),
        }
        steps.append(
            {
                "observation": obs,
                "action": action,
                "reward": 0.0,
                "discount": 1.0,
                "is_first": bool(new_idx == 0),
                "is_last": bool(is_last),
                "is_terminal": bool(is_last),
            }
        )

    episode_metadata = {
        "episode_id": bag_path.stem,
        "bag_path": str(bag_path),
        "eef_type": eef_type,
        "eef_dim": eef_dim,
        "timeline": args.timeline,
        "num_steps": len(steps),
        "camera_info_json": json.dumps(cam_info_map, ensure_ascii=False),
        "camera_intrinsics_json": json.dumps(cam_info_map, ensure_ascii=False),
        "sidecar_json": json.dumps(sidecar, ensure_ascii=False),
        "camera_extrinsics_json": json.dumps(final_camera_extrinsics, ensure_ascii=False),
        "joint_names_json": json.dumps(joint_names, ensure_ascii=False),
    }

    meta_update = {
        "camera_info": cam_info_map,
        "camera_info_by_episode": {bag_path.stem: cam_info_map},
        "marks_by_step": marks,
        "sidecar_meta": sidecar,
        "camera_extrinsics": final_camera_extrinsics,  # Use final value (from TF or URDF)
        "camera_extrinsics_by_episode": {bag_path.stem: final_camera_extrinsics},  # Use final value (from TF or URDF)
        "joint_names": joint_names,
        "rgb_topics": _parse_topics(args.rgb_topics),
        "depth_topics": _parse_topics(args.depth_topics),
        "eef_type": eef_type,
        "eef_dim": eef_dim,
    }

    return steps, episode_metadata, meta_update


def write_dataset_info(
    tfds_dir: Path,
    split: str,
    shard_paths: List[Path],
    base_meta: Dict,
) -> None:
    # CRITICAL: Create a minimal dataset_info.json BEFORE creating DeliveryOpenxBuilder
    # This prevents TFDS from calling initialize_from_bucket() which would try to access GCS
    # TFDS checks: if dataset_info.json exists -> read from local, else -> call initialize_from_bucket()
    dataset_info_path = tfds_dir / "dataset_info.json"
    if not dataset_info_path.exists():
        # Create minimal valid dataset_info.json to prevent GCS access
        minimal_info = {
            "name": DATASET_NAME,
            "version": TFDS_VERSION,
            "description": "Temporary dataset_info.json to prevent GCS access",
            "features": {},  # Will be populated by builder._info()
        }
        dataset_info_path.write_text(json.dumps(minimal_info, indent=2), encoding="utf-8")
        print(f"✅ Created minimal dataset_info.json to prevent GCS access", file=sys.stderr)
    
    builder = DeliveryOpenxBuilder(data_dir=str(tfds_dir))
    builder._meta = base_meta
    info = builder._info()

    shard_lengths = [1 for _ in shard_paths]  # one episode per shard (TFDS expects example count)
    num_bytes = sum(p.stat().st_size for p in shard_paths)
    template = naming.ShardedFileTemplate(
        data_dir=str(tfds_dir),
        template=f"{split}/{{DATASET}}-{{SPLIT}}.{{FILEFORMAT}}-{{SHARD_X_OF_Y}}",
        dataset_name=DATASET_NAME,
        split=split,
        filetype_suffix="tfrecord",
    )
    split_info = tfds.core.SplitInfo(
        name=split,
        shard_lengths=shard_lengths,
        num_bytes=num_bytes,
        filename_template=template,
    )
    info.set_splits(tfds.core.SplitDict([split_info]))
    info.set_file_format(tfds.core.FileFormat.TFRECORD)
    # Attach merged per-episode metadata for inspection
    meta_dict = info.metadata
    meta_dict["camera_info"] = base_meta.get("camera_info", {})
    meta_dict["camera_extrinsics"] = base_meta.get("camera_extrinsics", {})
    meta_dict["camera_info_by_episode"] = base_meta.get("camera_info_by_episode", {})
    meta_dict["camera_extrinsics_by_episode"] = base_meta.get("camera_extrinsics_by_episode", {})
    meta_dict["marks_by_episode"] = base_meta.get("marks_by_episode", {})
    meta_dict["sidecar_by_episode"] = base_meta.get("sidecar_by_episode", {})
    meta_dict["num_steps_by_episode"] = base_meta.get("num_steps_by_episode", {})
    info.write_to_directory(str(tfds_dir))
    # Pretty-print metadata.json for readability
    meta_path = tfds_dir / "metadata.json"
    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def process_all(args):
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)
    split = args.split

    # 排除隐藏目录（如 .cos）
    bag_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir() and not p.name.startswith('.')]
    if not bag_dirs:
        raise FileNotFoundError(f"No bag subdirectories found in {input_root}")

    tfds_dir = output_root / DATASET_NAME / TFDS_VERSION
    train_dir = tfds_dir / split
    train_dir.mkdir(parents=True, exist_ok=True)

    main_spec, aux_specs = _build_camera_specs(args)
    base_meta = {
        "dataset_name": DATASET_NAME,
        "split": split,
        "main_rgb_topic": main_spec.topic,
        "rgb_topics": _parse_topics(args.rgb_topics),
        "depth_topics": _parse_topics(args.depth_topics),
        "num_aux_streams": len(aux_specs),
        "timeline": args.timeline,
        "camera_info_by_episode": {},
        "camera_extrinsics_by_episode": {},
        "marks_by_episode": {},
        "sidecar_by_episode": {},
        "num_steps_by_episode": {},
        "episodes": [],
    }

    features = None
    info = None
    shard_paths: List[Path] = []
    for idx, bag_dir in enumerate(bag_dirs):
        print(f"\n=== Processing {bag_dir.name} ({idx + 1}/{len(bag_dirs)}) ===")
        steps, episode_meta, meta_update = _episode_from_bag(args, bag_dir, main_spec, aux_specs)
        for key, val in meta_update.items():
            if key.endswith("_by_episode"):
                merged = base_meta.get(key, {})
                merged.update(val)
                base_meta[key] = merged
            elif key not in base_meta or not base_meta[key]:
                base_meta[key] = val
        base_meta.setdefault("eef_type", episode_meta["eef_type"])
        base_meta.setdefault("eef_dim", episode_meta["eef_dim"])
        # Keep a union camera_info/camera_extrinsics for backward-compatible metadata
        if "camera_info" in meta_update and meta_update["camera_info"]:
            merged_info = base_meta.get("camera_info", {})
            merged_info.update(meta_update["camera_info"])
            base_meta["camera_info"] = merged_info
        if "camera_extrinsics" in meta_update and meta_update["camera_extrinsics"]:
            merged_ext = base_meta.get("camera_extrinsics", {})
            merged_ext.update(meta_update["camera_extrinsics"])
            base_meta["camera_extrinsics"] = merged_ext
        # Track per-episode step counts and sidecar/marks
        base_meta["marks_by_episode"][episode_meta["episode_id"]] = meta_update.get("marks_by_step", [])
        base_meta["sidecar_by_episode"][episode_meta["episode_id"]] = meta_update.get("sidecar_meta", {})
        base_meta["num_steps_by_episode"][episode_meta["episode_id"]] = episode_meta.get("num_steps", 0)
        if features is None or info is None:
            builder = DeliveryOpenxBuilder(data_dir=str(tfds_dir))
            builder._meta = base_meta
            info = builder._info()
            features = info.features

        example_dict = {"steps": steps, "episode_metadata": episode_meta}
        shard_name = f"{DATASET_NAME}-{split}.tfrecord-{idx:05d}-of-{len(bag_dirs):05d}"
        shard_path = train_dir / shard_name
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            writer.write(features.serialize_example(example_dict))
        shard_paths.append(shard_path)
        base_meta["episodes"].append({
            "episode_id": episode_meta["episode_id"],
            "bag_path": episode_meta.get("bag_path", ""),
            "num_steps": episode_meta.get("num_steps", 0),
            "shards": [str(shard_path)],
        })

    base_meta["shards"] = [str(p) for p in shard_paths]
    write_dataset_info(tfds_dir, split, shard_paths, base_meta)
    _ensure_permissions(output_root)
    print(f"✅ TFDS dataset ready at {tfds_dir}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description="One-pass rosbag -> TFDS (one bag -> one shard)")
    ap.add_argument("--input_root", type=Path, required=True, help="Root dir containing subdirs with .bag + sidecar .json")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output root for TFDS dataset")
    ap.add_argument("--split", default="train", help="TFDS split name")
    # Pass-through args for parsing
    ap.add_argument("--main_rgb_topic", default="/cam_h/color/image_raw/compressed")
    ap.add_argument("--rgb_topics", default="/cam_h/color/image_raw/compressed")
    ap.add_argument("--depth_topics", default="")
    ap.add_argument("--camera_info_topics", default="/cam_h/color/camera_info")
    ap.add_argument("--joint_cmd_topic", default="/joint_cmd")
    ap.add_argument("--sensors_data_raw_topic", default="/sensors_data_raw")
    ap.add_argument("--timeline", default="camera")
    ap.add_argument("--clip_to_marks", action="store_true")
    ap.add_argument("--eef", default="auto")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--base_frame", default="base_link")
    ap.add_argument("--tcp_frame_left", default="zarm_l7_link")
    ap.add_argument("--tcp_frame_right", default="zarm_r7_link")
    ap.add_argument("--tf_topic", default="/tf")
    ap.add_argument("--tf_static_topic", default="/tf_static")
    ap.add_argument("--camera_link_hints", default=None, help="JSON dict mapping camera name to link for URDF extrinsics")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    import time

    # Print GCS monitoring status
    print("\n" + "="*80, file=sys.stderr)
    print("🔍 GCS/Google Remote Access Monitoring ENABLED", file=sys.stderr)
    print("   All attempts to access GCS, Google Cloud, or Google Auth will be logged", file=sys.stderr)
    print("   Look for '⚠️  [GCS ACCESS DETECTED]' messages in stderr", file=sys.stderr)
    print("="*80 + "\n", file=sys.stderr)
    sys.stderr.flush()

    start_time = time.time()
    args = build_arg_parser().parse_args(argv)
    # Parse camera_link_hints: supports both JSON and comma-separated formats
    # This is done here to avoid duplicate parsing in process_all
    if isinstance(args.camera_link_hints, str) and args.camera_link_hints:
        try:
            args.camera_link_hints = json.loads(args.camera_link_hints)
        except Exception:
            # Fallback to comma-separated format: head:camera,left:l_hand_camera
            hints = {}
            for item in args.camera_link_hints.split(","):
                if ":" in item:
                    camera, link = item.split(":", 1)
                    hints[camera.strip()] = link.strip()
            args.camera_link_hints = hints if hints else None
    try:
        process_all(args)
        elapsed = time.time() - start_time
        print(f"⏱  Total conversion time: {elapsed:.1f} seconds")
        return 0
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
