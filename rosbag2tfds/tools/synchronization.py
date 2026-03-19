"""Timestamp alignment utilities for ROS bag → RLDS conversion."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Optional

import numpy as np


def nearest_indices(src_ts: np.ndarray, dst_ts: np.ndarray) -> np.ndarray:
    """Return indices of ``src_ts`` that are closest to each ``dst_ts`` entry."""
    if len(src_ts) == 0 or len(dst_ts) == 0:
        return np.zeros((0,), dtype=np.int32)
    idx = np.searchsorted(src_ts, dst_ts, side="left")
    idx = np.clip(idx, 0, len(src_ts) - 1)
    left = np.abs(src_ts[idx] - dst_ts)
    right_idx = np.minimum(idx + 1, len(src_ts) - 1)
    right = np.abs(src_ts[right_idx] - dst_ts)
    use_right = right < left
    idx[use_right] = right_idx[use_right]
    return idx.astype(np.int32)


@dataclasses.dataclass
class StreamAlignment:
    """Holds alignment indices and statistics for later diagnostics."""

    src_topic: str
    dst_topic: str
    indices: np.ndarray
    median_offset_ns: float
    max_offset_ns: float


def build_alignment(
    src_ts: np.ndarray,
    dst_ts: np.ndarray,
    *,
    src_topic: str,
    dst_topic: str,
) -> StreamAlignment:
    """Create an alignment object with timing statistics."""
    idx = nearest_indices(src_ts, dst_ts)
    if len(src_ts) == 0 or len(dst_ts) == 0 or len(idx) == 0:
        med = 0.0
        mx = 0.0
    else:
        offsets = src_ts[idx] - dst_ts
        med = float(np.median(offsets)) if len(offsets) else 0.0
        mx = float(np.max(np.abs(offsets))) if len(offsets) else 0.0
    return StreamAlignment(
        src_topic=src_topic,
        dst_topic=dst_topic,
        indices=idx,
        median_offset_ns=med,
        max_offset_ns=mx,
    )


def build_alignments_batch(
    src_ts_dict: Dict[str, np.ndarray],
    dst_ts: np.ndarray,
    *,
    dst_topic: str = "main_timeline",
) -> Dict[str, np.ndarray]:
    """
    Batch align multiple source timestamps to destination timeline.
    
    This is more efficient than calling build_alignment() multiple times
    because it reuses the sorted destination array.
    
    Args:
        src_ts_dict: Dictionary mapping topic names to source timestamps
        dst_ts: Destination timeline timestamps
        dst_topic: Name of destination topic (for logging)
    
    Returns:
        Dictionary mapping topic names to alignment indices
    """
    if len(dst_ts) == 0:
        return {topic: np.zeros((0,), dtype=np.int32) for topic in src_ts_dict}
    
    # Sort destination once (if not already sorted)
    if not np.all(dst_ts[:-1] <= dst_ts[1:]):
        dst_ts_sorted = np.sort(dst_ts)
        dst_idx_map = np.argsort(dst_ts)
    else:
        dst_ts_sorted = dst_ts
        dst_idx_map = None
    
    result = {}
    for topic, src_ts in src_ts_dict.items():
        if len(src_ts) == 0:
            result[topic] = np.zeros((len(dst_ts),), dtype=np.int32)
        else:
            idx = nearest_indices(src_ts, dst_ts_sorted)
            if dst_idx_map is not None:
                # Remap indices if we sorted dst_ts
                result[topic] = dst_idx_map[idx]
            else:
                result[topic] = idx
    
    return result
