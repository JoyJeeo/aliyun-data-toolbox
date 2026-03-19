"""TF (Transform) reading utilities for extracting TCP poses from rosbag."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import os

from scipy.spatial.transform import Rotation


class TFBuffer:
    """
    Buffer for storing and querying TF transforms from rosbag.

    Supports:
    - Both /tf (dynamic) and /tf_static (static, published once)
    - BFS path finding through TF graph
    - Forward and reverse edge traversal with matrix inversion
    - Multiple time matching modes for static vs dynamic TF
    """

    def __init__(self):
        """Initialize empty TF buffer."""
        # Edge storage: (parent, child) -> {timestamp_ns: TransformStamped}
        self.edges: Dict[Tuple[str, str], Dict[int, any]] = {}
        # Forward adjacency: parent -> set(children)
        self.adj: Dict[str, set] = {}
        # Reverse adjacency: child -> set(parents)
        self.rev_adj: Dict[str, set] = {}
        # All timestamps
        self.all_timestamps: List[int] = []

    def add_transform(self, transform, timestamp_ns: int):
        """
        Add a transform to the buffer.

        Args:
            transform: TransformStamped message
            timestamp_ns: Timestamp in nanoseconds
        """
        parent = transform.header.frame_id
        child = transform.child_frame_id
        key = (parent, child)

        if key not in self.edges:
            self.edges[key] = {}
        self.edges[key][timestamp_ns] = transform

        # Update adjacency
        if parent not in self.adj:
            self.adj[parent] = set()
        self.adj[parent].add(child)

        if child not in self.rev_adj:
            self.rev_adj[child] = set()
        self.rev_adj[child].add(parent)

        self.all_timestamps.append(timestamp_ns)

    def _nearest_transform(self, key: Tuple[str, str], timestamp_ns: int, tolerance_ns: int, mode: str = "nearest_within"):
        """
        Find transform for edge at given timestamp.

        Args:
            key: (parent, child) edge key
            timestamp_ns: Target timestamp
            tolerance_ns: Tolerance for nearest_within mode
            mode: One of:
                - "nearest_within": Only within tolerance (for dynamic TF)
                - "latest_before": Latest before timestamp, fallback to earliest (for static TF)
                - "nearest_any": Any nearest without tolerance limit

        Returns:
            TransformStamped message or None
        """
        if key not in self.edges:
            return None
        frame_transforms = self.edges[key]
        if not frame_transforms:
            return None

        chosen_ts = None

        if mode == "latest_before":
            # For static TF: find <= timestamp_ns, fallback to earliest
            candidates = [ts for ts in frame_transforms.keys() if ts <= timestamp_ns]
            if candidates:
                chosen_ts = max(candidates)
            else:
                # Static TF may have timestamp 0 or very early
                chosen_ts = min(frame_transforms.keys())
        elif mode == "nearest_any":
            # Find absolute nearest without tolerance
            min_diff = float('inf')
            for ts in frame_transforms.keys():
                diff = abs(ts - timestamp_ns)
                if diff < min_diff:
                    min_diff = diff
                    chosen_ts = ts
        else:  # nearest_within
            # Only within tolerance (for dynamic TF)
            min_diff = float('inf')
            for ts in frame_transforms.keys():
                diff = abs(ts - timestamp_ns)
                if diff < min_diff and diff <= tolerance_ns:
                    min_diff = diff
                    chosen_ts = ts

        if chosen_ts is None:
            return None
        return frame_transforms[chosen_ts]

    @staticmethod
    def _tf_to_matrix(transform) -> np.ndarray:
        """Convert TransformStamped to 4x4 homogeneous matrix."""
        t = transform.transform.translation
        r = transform.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat = np.array([r.x, r.y, r.z, r.w], dtype=np.float64)
        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    @staticmethod
    def _invert_matrix(T: np.ndarray) -> np.ndarray:
        """Invert a 4x4 homogeneous transform matrix."""
        R = T[:3, :3]
        p = T[:3, 3]
        T_inv = np.eye(4, dtype=np.float64)
        R_inv = R.T
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = -R_inv @ p
        return T_inv

    def _find_path(self, start: str, goal: str) -> Optional[List[Tuple[str, str, bool]]]:
        """
        Find path from start to goal in TF graph using BFS.

        Args:
            start: Starting frame
            goal: Goal frame

        Returns:
            List of (u, v, forward) tuples:
            - forward=True: edge exists as (u->v)
            - forward=False: edge exists as (v->u), need to invert
            Returns None if no path found.
        """
        q = deque()
        q.append(start)
        prev = {start: None}
        edge_dir = {}

        while q:
            u = q.popleft()
            if u == goal:
                break

            # Forward neighbors (u is parent)
            for v in self.adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    edge_dir[(u, v)] = True
                    q.append(v)

            # Reverse neighbors (u is child, walk backward)
            for v in self.rev_adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    edge_dir[(v, u)] = False  # Walking from u to v using (v->u) edge inverted
                    q.append(v)

        if goal not in prev:
            return None

        # Backtrack path
        path_nodes = []
        cur = goal
        while cur is not None:
            path_nodes.append(cur)
            cur = prev[cur]
        path_nodes.reverse()  # start -> goal

        # Convert to edge list with direction
        path_edges = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            forward = edge_dir.get((u, v))
            if forward is None:
                forward = False
            path_edges.append((u, v, forward))

        return path_edges

    def get_transform(
        self,
        target_frame: str,
        source_frame: str,
        timestamp_ns: int,
        tolerance_ns: int = 100000000,  # 100ms tolerance
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get transform from source_frame to target_frame at given timestamp.
        Supports chain traversal through multiple frames and inverse transforms.

        Args:
            target_frame: Target frame (e.g., "base_link")
            source_frame: Source frame (e.g., "camera")
            timestamp_ns: Timestamp in nanoseconds
            tolerance_ns: Maximum time difference for dynamic TF lookup

        Returns:
            Tuple of (position, quaternion) or None if not found
            - position: [x, y, z] in meters
            - quaternion: [x, y, z, w] (ROS format)
        """
        if source_frame == target_frame:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Find path: target_frame -> source_frame
        path = self._find_path(target_frame, source_frame)
        if not path:
            return None

        # Multiply transforms along path
        T_total = np.eye(4, dtype=np.float64)

        for u, v, forward in path:
            if forward:
                key = (u, v)
            else:
                key = (v, u)

            # Try multiple modes: nearest_within -> latest_before -> nearest_any
            # This handles both dynamic TF (high frequency) and static TF (published once)
            msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_within")
            if msg is None:
                msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="latest_before")
            if msg is None:
                msg = self._nearest_transform(key, timestamp_ns, tolerance_ns, mode="nearest_any")

            if msg is None:
                return None

            T = self._tf_to_matrix(msg)
            if not forward:
                T = self._invert_matrix(T)

            T_total = T_total @ T

        # Extract position and quaternion
        pos = T_total[:3, 3]
        rot = Rotation.from_matrix(T_total[:3, :3]).as_quat()  # [x, y, z, w]

        return pos.astype(np.float32), rot.astype(np.float32)

    def has_frame(self, frame_id: str) -> bool:
        """Check if frame exists in buffer (as parent or child)."""
        return frame_id in self.adj or frame_id in self.rev_adj

    def get_timestamps(self) -> np.ndarray:
        """Get all unique timestamps in buffer."""
        if not self.all_timestamps:
            return np.array([], dtype=np.int64)
        return np.unique(np.array(self.all_timestamps, dtype=np.int64))

    def all_frames(self) -> List[str]:
        """Get all frame names in buffer."""
        return sorted(set(list(self.adj.keys()) + list(self.rev_adj.keys())))


def read_tf_from_bag(
    bag: "rosbag.Bag",
    tf_topic: str = "/tf",
    tf_static_topic: str = "/tf_static",
) -> Tuple[TFBuffer, np.ndarray]:
    """
    Read TF transforms from rosbag (supports both /tf and /tf_static).

    Args:
        bag: Open rosbag.Bag instance
        tf_topic: TF topic name (default: "/tf")
        tf_static_topic: TF static topic name (default: "/tf_static")

    Returns:
        Tuple of (TFBuffer, timestamps_array)
        - TFBuffer: Buffer containing all transforms
        - timestamps: Array of timestamps in nanoseconds
    """
    tf_buffer = TFBuffer()
    timestamps = []

    # Check which TF topics exist
    available_tf_topics = []
    try:
        topic_info = bag.get_type_and_topic_info()[1]
        if tf_topic in topic_info:
            available_tf_topics.append(tf_topic)
        if tf_static_topic in topic_info:
            available_tf_topics.append(tf_static_topic)

        if not available_tf_topics:
            print(f"Info: TF topics '{tf_topic}' and '{tf_static_topic}' not found in bag. Will use FK calculation.")
            return tf_buffer, np.array([], dtype=np.int64)
    except Exception:
        # If we can't check, try both topics anyway
        available_tf_topics = [tf_topic, tf_static_topic]

    try:
        transform_count = 0
        tf_count = 0
        tf_static_count = 0

        for topic, msg, t in bag.read_messages(topics=available_tf_topics):
            if hasattr(msg, 'transforms'):
                # TFMessage contains list of TransformStamped
                for transform in msg.transforms:
                    # Use message timestamp (transform.header.stamp) instead of bag write time (t)
                    # This ensures proper alignment with other sensor data that also uses message timestamps
                    timestamp_ns = int(transform.header.stamp.to_nsec())
                    tf_buffer.add_transform(transform, timestamp_ns)
                    timestamps.append(timestamp_ns)
                    transform_count += 1

                    if topic == tf_topic:
                        tf_count += 1
                    elif topic == tf_static_topic:
                        tf_static_count += 1

        if transform_count > 0:
            print(f"✓ Read {transform_count} TF transforms (tf: {tf_count}, tf_static: {tf_static_count})")
        else:
            print(f"Info: No TF transforms found in {available_tf_topics}. Will use FK calculation.")
    except Exception as e:
        print(f"Warning: Failed to read TF from bag: {e}. Will use FK calculation.")

    return tf_buffer, np.array(timestamps, dtype=np.int64) if timestamps else np.array([], dtype=np.int64)


def get_tcp_pose_from_tf(
    tf_buffer: TFBuffer,
    base_frame: str,
    tcp_frame: str,
    timestamp_ns: int,
    tolerance_ns: int = 100000000,  # 100ms
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get TCP pose from TF buffer.
    
    Args:
        tf_buffer: TFBuffer instance
        base_frame: Base frame name (e.g., "base_link")
        tcp_frame: TCP frame name (e.g., "zarm_l7_end_effector")
        timestamp_ns: Timestamp in nanoseconds
        tolerance_ns: Maximum time difference for lookup
    
    Returns:
        Tuple of (position, quaternion) or None if not found
        - position: [x, y, z] in meters
        - quaternion: [x, y, z, w] (ROS format)
    """
    if not tf_buffer.has_frame(tcp_frame):
        return None
    
    return tf_buffer.get_transform(base_frame, tcp_frame, timestamp_ns, tolerance_ns)


def get_camera_extrinsics_from_tf(
    tf_buffer: TFBuffer,
    base_frame: str,
    camera_frame: str,
    timestamp_ns: int,
    tolerance_ns: int = 100000000,  # 100ms
) -> Optional[Dict]:
    """
    Get camera extrinsics (pose relative to base_frame) from TF buffer.
    
    Args:
        tf_buffer: TFBuffer instance
        base_frame: Base frame name (e.g., "base_link")
        camera_frame: Camera frame name (e.g., "camera_head_optical_link")
        timestamp_ns: Timestamp in nanoseconds
        tolerance_ns: Maximum time difference for lookup
    
    Returns:
        Dictionary with extrinsics info or None if not found:
        {
            "parent_link": base_frame,
            "child_link": camera_frame,
            "xyz": [x, y, z],
            "rpy": [roll, pitch, yaw],
            "transform_matrix": 4x4 matrix (list of lists)
        }
    """
    pose = tf_buffer.get_transform(base_frame, camera_frame, timestamp_ns, tolerance_ns)
    if pose is None:
        return None
    
    pos, quat = pose
    
    # Convert quaternion to RPY
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(quat)  # [x, y, z, w]
    rpy = rot.as_euler('xyz')
    
    # Build 4x4 transform matrix
    rot_matrix = rot.as_matrix()
    # Convert all numpy types to Python native types for JSON serialization
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


def find_urdf_in_mount_path(urdf_name: str = "biped_s49.urdf", mount_path: str = "/cos/files") -> Optional[str]:
    """
    Find URDF file in mount path.
    
    Args:
        urdf_name: URDF filename (default: "biped_s49.urdf")
        mount_path: Mount path to search (default: "/cos/files")
    
    Returns:
        Full path to URDF file, or None if not found
    """
    # Try direct path first
    direct_path = os.path.join(mount_path, urdf_name)
    if os.path.exists(direct_path):
        return direct_path
    
    # Try common subdirectories
    search_paths = [
        os.path.join(mount_path, "urdf", urdf_name),
        os.path.join(mount_path, "urdf", "biped_s49", "urdf", urdf_name),
        os.path.join(mount_path, "rosbag", "biped_s49", "urdf", urdf_name),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None

