"""Utilities for extracting kinematic info from URDF files."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _parse_xyz(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.strip().split()
    if len(parts) != 3:
        raise ValueError(f"invalid xyz string: {text}")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _parse_rpy(text: Optional[str]) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.strip().split()
    if len(parts) != 3:
        raise ValueError(f"invalid rpy string: {text}")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return rot


def load_urdf(path: Path) -> ET.ElementTree:
    """Load URDF into an ElementTree."""
    tree = ET.parse(path)
    return tree


def extract_joint_order(
    tree: ET.ElementTree,
    *,
    include_fixed: bool = False,
    whitelist: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return joint names in file order."""
    allowed_types = {"revolute", "continuous", "prismatic"}
    if include_fixed:
        allowed_types.add("fixed")
    whitelist_set = set(whitelist or [])
    joint_names: List[str] = []
    root = tree.getroot()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        if whitelist_set and name not in whitelist_set:
            continue
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type not in allowed_types:
            continue
        joint_names.append(name)
    return joint_names


def extract_camera_extrinsics(
    tree: ET.ElementTree,
    *,
    camera_link_hints: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """Derive camera extrinsics from link origins."""
    root = tree.getroot()
    camera_map: Dict[str, Dict] = {}
    hints = camera_link_hints or {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        child_link = child.attrib.get("link")
        parent_link = parent.attrib.get("link")
        if not child_link or not parent_link:
            continue
        camera_name = None
        for hint_name, link_name in hints.items():
            if child_link == link_name:
                camera_name = hint_name
                break
        if not camera_name and ("camera" in (child_link.lower())):
            camera_name = child_link
        if not camera_name:
            continue
        origin = joint.find("origin")
        xyz = _parse_xyz(origin.attrib.get("xyz")) if origin is not None else (0.0, 0.0, 0.0)
        rpy = _parse_rpy(origin.attrib.get("rpy")) if origin is not None else (0.0, 0.0, 0.0)
        rot = _rpy_to_matrix(rpy)
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = rot
        tf[:3, 3] = np.array(xyz)
        camera_map[camera_name] = {
            "parent_link": parent_link,
            "child_link": child_link,
            "xyz": list(xyz),
            "rpy": list(rpy),
            "transform_matrix": tf.tolist(),
        }
    return camera_map
