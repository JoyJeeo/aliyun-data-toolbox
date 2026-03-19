"""Drake Forward Kinematics utilities for computing TCP poses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.systems.framework import Context
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False


class DrakeFKCalculator:
    """Forward Kinematics calculator using Drake library."""
    
    def __init__(self, urdf_path: Path, base_link: str = "base_link"):
        """
        Initialize FK calculator with URDF file.
        
        Args:
            urdf_path: Path to URDF file
            base_link: Name of base link (default: "base_link")
        
        Raises:
            ImportError: If Drake is not available
            FileNotFoundError: If URDF file doesn't exist
        """
        if not DRAKE_AVAILABLE:
            raise ImportError(
                "Drake library is not available. Please install it with: pip install drake"
            )
        
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        
        self.urdf_path = urdf_path
        self.base_link = base_link
        
        # Initialize MultibodyPlant
        self.plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self.plant)
        # Drake 1.26.0+ uses AddModels instead of AddModelFromFile
        try:
            # Try new API first (Drake 1.26.0+)
            parser.AddModels(str(urdf_path))
        except (AttributeError, TypeError):
            # Fallback to old API (older Drake versions)
            parser.AddModelFromFile(str(urdf_path))
        self.plant.Finalize()
        
        # Get base frame
        try:
            self.base_frame = self.plant.GetFrameByName(base_link)
        except Exception as e:
            raise ValueError(f"Base link '{base_link}' not found in URDF: {e}")
        
        # Get TCP frames (will be set when needed)
        self.tcp_frame_left = None
        self.tcp_frame_right = None
        
        # Create context
        self.context = self.plant.CreateDefaultContext()
        
        # Cache joint indices
        self._left_arm_joint_indices = None
        self._right_arm_joint_indices = None
    
    def _get_joint_indices(self, joint_names: list[str]) -> list[int]:
        """
        Get Drake plant position indices from joint names.
        Returns list of position_start() indices for each joint.
        """
        indices = []
        for name in joint_names:
            try:
                joint = self.plant.GetJointByName(name)
                pos_start = joint.position_start()
                num_pos = joint.num_positions()
                # Add all position indices for this joint
                for i in range(num_pos):
                    indices.append(pos_start + i)
            except Exception as e:
                raise ValueError(f"Joint '{name}' not found in URDF: {e}")
        return indices
    
    def _get_tcp_frame(self, tcp_link_name: str):
        """Get TCP frame by link name."""
        try:
            return self.plant.GetFrameByName(tcp_link_name)
        except Exception as e:
            raise ValueError(f"TCP link '{tcp_link_name}' not found in URDF: {e}")
    
    def compute_tcp_pose_left(
        self,
        joint_positions: np.ndarray,
        tcp_link: str = "zarm_l7_end_effector",
        left_arm_joint_names: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute left arm TCP pose from joint positions.
        
        Args:
            joint_positions: Full joint position array (28 dims)
            tcp_link: Name of left arm TCP link (default: "zarm_l7_end_effector")
            left_arm_joint_names: List of left arm joint names (default: auto-detect)
                                Expected: ["zarm_l1_joint", ..., "zarm_l7_joint"]
        
        Returns:
            Tuple of (position, quaternion):
            - position: [x, y, z] in meters relative to base_link
            - quaternion: [x, y, z, w] relative to base_link
        """
        if left_arm_joint_names is None:
            # Default left arm joint names (indices 12-18 in 28-dim array)
            left_arm_joint_names = [
                "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint",
                "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint"
            ]
        
        # Get joint indices if not cached
        if self._left_arm_joint_indices is None:
            self._left_arm_joint_indices = self._get_joint_indices(left_arm_joint_names)
        
        # Extract left arm joint positions from our 28-dim array
        # Left arm joints are at indices 12-18 in the 28-dim array
        left_arm_array_indices = list(range(12, 19))  # zarm_l1 to zarm_l7
        if len(joint_positions) < 19:
            raise ValueError(
                f"Joint positions array too short: {len(joint_positions)}, "
                f"expected at least 19 for left arm joints"
            )
        
        left_arm_values = [joint_positions[i] for i in left_arm_array_indices]
        
        # Get TCP frame if not cached
        if self.tcp_frame_left is None:
            self.tcp_frame_left = self._get_tcp_frame(tcp_link)
        
        # Set all joint positions (Drake requires full state)
        all_positions = self.plant.GetPositions(self.context)
        # Set left arm joint positions
        for i, drake_idx in enumerate(self._left_arm_joint_indices):
            if i < len(left_arm_values):
                all_positions[drake_idx] = left_arm_values[i]
        self.plant.SetPositions(self.context, all_positions)
        
        # Compute TCP pose relative to base_link
        tcp_pose = self.tcp_frame_left.CalcPoseInWorld(self.context)
        
        # Extract position and quaternion
        position = tcp_pose.translation()  # [x, y, z]
        quaternion = tcp_pose.rotation().ToQuaternion()  # Drake returns [w, x, y, z]
        
        # Convert to ROS format [x, y, z, w]
        quat_ros = np.array([quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()])
        
        return position.astype(np.float32), quat_ros.astype(np.float32)
    
    def compute_tcp_pose_right(
        self,
        joint_positions: np.ndarray,
        tcp_link: str = "zarm_r7_end_effector",
        right_arm_joint_names: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute right arm TCP pose from joint positions.
        
        Args:
            joint_positions: Full joint position array (28 dims)
            tcp_link: Name of right arm TCP link (default: "zarm_r7_end_effector")
            right_arm_joint_names: List of right arm joint names (default: auto-detect)
                                 Expected: ["zarm_r1_joint", ..., "zarm_r7_joint"]
        
        Returns:
            Tuple of (position, quaternion):
            - position: [x, y, z] in meters relative to base_link
            - quaternion: [x, y, z, w] relative to base_link
        """
        if right_arm_joint_names is None:
            # Default right arm joint names (indices 19-25 in 28-dim array)
            right_arm_joint_names = [
                "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint",
                "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint"
            ]
        
        # Get joint indices if not cached
        if self._right_arm_joint_indices is None:
            self._right_arm_joint_indices = self._get_joint_indices(right_arm_joint_names)
        
        # Extract right arm joint positions from our 28-dim array
        # Right arm joints are at indices 19-25 in the 28-dim array
        right_arm_array_indices = list(range(19, 26))  # zarm_r1 to zarm_r7
        if len(joint_positions) < 26:
            raise ValueError(
                f"Joint positions array too short: {len(joint_positions)}, "
                f"expected at least 26 for right arm joints"
            )
        
        right_arm_values = [joint_positions[i] for i in right_arm_array_indices]
        
        # Get TCP frame if not cached
        if self.tcp_frame_right is None:
            self.tcp_frame_right = self._get_tcp_frame(tcp_link)
        
        # Set all joint positions (Drake requires full state)
        all_positions = self.plant.GetPositions(self.context)
        # Set right arm joint positions
        for i, drake_idx in enumerate(self._right_arm_joint_indices):
            if i < len(right_arm_values):
                all_positions[drake_idx] = right_arm_values[i]
        self.plant.SetPositions(self.context, all_positions)
        
        # Compute TCP pose relative to base_link
        tcp_pose = self.tcp_frame_right.CalcPoseInWorld(self.context)
        
        # Extract position and quaternion
        position = tcp_pose.translation()  # [x, y, z]
        quaternion = tcp_pose.rotation().ToQuaternion()  # Drake returns [w, x, y, z]
        
        # Convert to ROS format [x, y, z, w]
        quat_ros = np.array([quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()])
        
        return position.astype(np.float32), quat_ros.astype(np.float32)


def create_fk_calculator(urdf_path: Path, base_link: str = "base_link") -> Optional[DrakeFKCalculator]:
    """
    Create FK calculator instance.
    
    Args:
        urdf_path: Path to URDF file
        base_link: Name of base link
    
    Returns:
        DrakeFKCalculator instance, or None if Drake is not available
    """
    if not DRAKE_AVAILABLE:
        print("Error: Drake library is not available. Please install it with: pip install drake")
        print("       Or use the pre-built Drake binaries from: https://drake.mit.edu/installation.html")
        return None
    
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        return None
    
    try:
        return DrakeFKCalculator(urdf_path, base_link)
    except ImportError as e:
        print(f"Error: Drake import failed: {e}")
        print("       Please ensure Drake is properly installed.")
        return None
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error: URDF parsing or configuration issue: {e}")
        print(f"       Check that the URDF file is valid and contains the required links/joints.")
        return None
    except Exception as e:
        print(f"Error: Failed to create FK calculator: {type(e).__name__}: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return None

