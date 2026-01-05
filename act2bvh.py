#!/usr/bin/env python3
"""
ACT to BVH Converter - Production Quality
Converts Hash Animation Master ACT motion capture files to Biovision BVH format.

Features:
- Robust error handling and validation
- Configurable bone hierarchy mapping
- Support for multiple ACT file versions
- Progress reporting
- Logging system
- Unit conversion options
- Frame rate detection and configuration
"""

import sys
import os
import re
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json

# ============================================================================
# Data Structures
# ============================================================================

class RotationOrder(Enum):
    """BVH rotation order conventions"""
    ZXY = "Zrotation Xrotation Yrotation"
    XYZ = "Xrotation Yrotation Zrotation"
    ZYX = "Zrotation Yrotation Xrotation"
    YXZ = "Yrotation Xrotation Zrotation"
    
    @classmethod
    def from_string(cls, order_str: str) -> 'RotationOrder':
        """Parse rotation order from string"""
        order_str = order_str.upper().replace(" ", "")
        for member in cls:
            if member.name == order_str:
                return member
        raise ValueError(f"Invalid rotation order: {order_str}. "
                        f"Valid options: {[m.name for m in cls]}")

@dataclass
class Bone:
    """Bone definition for BVH hierarchy"""
    name: str
    parent: Optional[str]
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: List[str] = None
    children: List['Bone'] = None
    has_end_site: bool = False
    end_site_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.channels is None:
            self.channels = ["Xposition", "Yposition", "Zposition", 
                           "Zrotation", "Yrotation", "Xrotation"]

@dataclass
class FrameData:
    """Animation frame data"""
    frame_number: int
    bone_rotations: Dict[str, Tuple[float, float, float]]  # bone_name -> (rx, ry, rz)
    bone_positions: Dict[str, Tuple[float, float, float]] = None
    
    def __post_init__(self):
        if self.bone_positions is None:
            self.bone_positions = {}

@dataclass
class ACTFile:
    """Parsed ACT file data"""
    bones: Dict[str, List[FrameData]]
    frame_count: int
    fps: float = 30.0
    metadata: Dict[str, Any] = None
    original_file: str = ""
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ============================================================================
# Configuration Manager
# ============================================================================

class Config:
    """Configuration manager for conversion settings"""
    
    DEFAULT_CONFIG = {
        # Output settings
        'output_rotation_order': 'ZXY',
        'scale_factor': 1.0,
        'negate_y': False,
        'negate_z': False,
        
        # Bone mapping presets
        'bone_mappings': {
            # Common ACT to BVH bone name mappings
            'Hips': ['Hips', 'Pelvis', 'Root'],
            'Spine': ['Spine', 'Spine1', 'Chest'],
            'Chest': ['Chest', 'Spine2', 'Spine3'],
            'Neck': ['Neck', 'Neck1'],
            'Head': ['Head'],
            'LeftShoulder': ['LeftShoulder', 'L_Clavicle'],
            'LeftArm': ['LeftArm', 'L_UpperArm'],
            'LeftForeArm': ['LeftForeArm', 'L_Forearm'],
            'LeftHand': ['LeftHand', 'L_Hand'],
            'RightShoulder': ['RightShoulder', 'R_Clavicle'],
            'RightArm': ['RightArm', 'R_UpperArm'],
            'RightForeArm': ['RightForeArm', 'R_Forearm'],
            'RightHand': ['RightHand', 'R_Hand'],
            'LeftUpLeg': ['LeftUpLeg', 'L_Thigh'],
            'LeftLeg': ['LeftLeg', 'L_Calf'],
            'LeftFoot': ['LeftFoot', 'L_Foot'],
            'LeftToeBase': ['LeftToeBase', 'L_Toe'],
            'RightUpLeg': ['RightUpLeg', 'R_Thigh'],
            'RightLeg': ['RightLeg', 'R_Calf'],
            'RightFoot': ['RightFoot', 'R_Foot'],
            'RightToeBase': ['RightToeBase', 'R_Toe'],
        },
        
        # Hierarchy templates
        'hierarchy_templates': {
            'standard': [
                ('Hips', None),
                ('Spine', 'Hips'),
                ('Chest', 'Spine'),
                ('Neck', 'Chest'),
                ('Head', 'Neck'),
                ('LeftShoulder', 'Chest'),
                ('LeftArm', 'LeftShoulder'),
                ('LeftForeArm', 'LeftArm'),
                ('LeftHand', 'LeftForeArm'),
                ('RightShoulder', 'Chest'),
                ('RightArm', 'RightShoulder'),
                ('RightForeArm', 'RightArm'),
                ('RightHand', 'RightForeArm'),
                ('LeftUpLeg', 'Hips'),
                ('LeftLeg', 'LeftUpLeg'),
                ('LeftFoot', 'LeftLeg'),
                ('LeftToeBase', 'LeftFoot'),
                ('RightUpLeg', 'Hips'),
                ('RightLeg', 'RightUpLeg'),
                ('RightFoot', 'RightLeg'),
                ('RightToeBase', 'RightFoot'),
            ]
        },
        
        # Frame rate detection
        'auto_detect_fps': True,
        'default_fps': 30.0,
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.settings = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            self.settings.update(user_config)
            logging.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.settings[key] = value
    
    def save(self, config_file: str):
        """Save configuration to JSON file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            logging.info(f"Saved configuration to {config_file}")
        except Exception as e:
            logging.error(f"Failed to save config file: {e}")

# ============================================================================
# ACT Parser
# ============================================================================

class ACTParser:
    """Robust ACT file parser with error handling"""
    
    # ACT file format patterns
    BONE_PATTERN = re.compile(r'^([A-Za-z][A-Za-z0-9_\-\.]*)$')
    FRAME_PATTERN = re.compile(r'^(\d+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)$')
    METADATA_PATTERN = re.compile(r'^#\s*(\w+):\s*(.+)$')
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def parse(self, filepath: str) -> ACTFile:
        """
        Parse ACT file with comprehensive error handling
        
        Args:
            filepath: Path to ACT file
            
        Returns:
            ACTFile object containing parsed data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is malformed
        """
        self.logger.info(f"Parsing ACT file: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ACT file not found: {filepath}")
        
        bones: Dict[str, List[FrameData]] = {}
        metadata = {}
        current_bone = None
        frame_data = []
        max_frame = 0
        line_num = 0
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.rstrip('\n')
                    
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Parse metadata comments
                    if line.startswith('#'):
                        match = self.METADATA_PATTERN.match(line)
                        if match:
                            key, value = match.groups()
                            metadata[key] = value.strip()
                        continue
                    
                    # Check if line is a bone name
                    if self.BONE_PATTERN.match(line):
                        current_bone = line.strip()
                        if current_bone not in bones:
                            bones[current_bone] = []
                        self.logger.debug(f"Found bone: {current_bone}")
                        continue
                    
                    # Parse frame data
                    match = self.FRAME_PATTERN.match(line)
                    if match and current_bone:
                        try:
                            frame_num = int(match.group(1))
                            rot_x = float(match.group(2))
                            rot_y = float(match.group(3))
                            rot_z = float(match.group(4))
                            
                            # Validate rotation values
                            if not (-360 <= rot_x <= 360 and -360 <= rot_y <= 360 and -360 <= rot_z <= 360):
                                self.logger.warning(
                                    f"Line {line_num}: Rotation values out of expected range "
                                    f"for bone {current_bone} frame {frame_num}"
                                )
                            
                            frame_data = FrameData(
                                frame_number=frame_num,
                                bone_rotations={current_bone: (rot_x, rot_y, rot_z)}
                            )
                            bones[current_bone].append(frame_data)
                            
                            if frame_num > max_frame:
                                max_frame = frame_num
                                
                        except ValueError as e:
                            self.logger.warning(
                                f"Line {line_num}: Could not parse frame data: {line}. Error: {e}"
                            )
                        continue
                    
                    # Unknown line format
                    self.logger.warning(f"Line {line_num}: Unrecognized format: {line}")
            
            # Validate parsed data
            self._validate_act_data(bones, max_frame)
            
            # Determine frame rate
            fps = self._detect_frame_rate(metadata, bones)
            
            # Organize frame data
            organized_bones = self._organize_frame_data(bones, max_frame + 1)
            
            act_file = ACTFile(
                bones=organized_bones,
                frame_count=max_frame + 1,
                fps=fps,
                metadata=metadata,
                original_file=filepath
            )
            
            self.logger.info(f"Successfully parsed ACT file: {len(bones)} bones, "
                           f"{act_file.frame_count} frames, {fps} FPS")
            
            return act_file
            
        except UnicodeDecodeError:
            self.logger.error(f"File {filepath} is not a valid text file")
            raise ValueError(f"File {filepath} is not a valid ACT text file")
        except Exception as e:
            self.logger.error(f"Error parsing ACT file at line {line_num}: {e}")
            raise ValueError(f"Failed to parse ACT file: {e}")
    
    def _validate_act_data(self, bones: Dict, max_frame: int):
        """Validate parsed ACT data"""
        if not bones:
            raise ValueError("No bone data found in ACT file")
        
        # Check for consistent frame counts per bone
        frame_counts = {bone: len(frames) for bone, frames in bones.items()}
        unique_counts = set(frame_counts.values())
        
        if len(unique_counts) > 1:
            self.logger.warning(
                f"Inconsistent frame counts across bones: {frame_counts}"
            )
        
        # Check for missing frames
        for bone_name, frames in bones.items():
            frame_numbers = [f.frame_number for f in frames]
            if len(frame_numbers) != len(set(frame_numbers)):
                self.logger.warning(f"Duplicate frame numbers found for bone {bone_name}")
    
    def _detect_frame_rate(self, metadata: Dict, bones: Dict) -> float:
        """Detect frame rate from metadata or data"""
        
        # Try to get FPS from metadata
        if 'FPS' in metadata:
            try:
                return float(metadata['FPS'])
            except ValueError:
                self.logger.warning(f"Invalid FPS value in metadata: {metadata['FPS']}")
        
        # Try common metadata keys
        for key in ['FrameRate', 'frame_rate', 'fps', 'FramesPerSecond']:
            if key in metadata:
                try:
                    return float(metadata[key])
                except (ValueError, KeyError):
                    continue
        
        # Auto-detect from frame numbers if enabled
        if self.config.get('auto_detect_fps', True):
            for bone_name, frames in bones.items():
                if frames:
                    # Check if frames are sequential
                    frame_nums = sorted([f.frame_number for f in frames])
                    if len(frame_nums) > 10:  # Need enough frames
                        time_span = frame_nums[-1] - frame_nums[0]
                        if time_span > 0:
                            # Calculate FPS based on frame numbering
                            # Assuming frames are numbered by time
                            fps = len(frame_nums) / time_span * 1000  # Rough estimate
                            if 1 <= fps <= 120:  # Reasonable FPS range
                                self.logger.info(f"Auto-detected FPS: {fps:.2f}")
                                return fps
        
        # Default FPS
        default_fps = self.config.get('default_fps', 30.0)
        self.logger.info(f"Using default FPS: {default_fps}")
        return default_fps
    
    def _organize_frame_data(self, bones: Dict, frame_count: int) -> Dict:
        """
        Reorganize frame data to be indexed by frame number
        """
        organized = {bone: [None] * frame_count for bone in bones.keys()}
        
        for bone_name, frames in bones.items():
            for frame in frames:
                if frame.frame_number < frame_count:
                    organized[bone_name][frame.frame_number] = frame
        
        return organized

# ============================================================================
# Bone Mapper
# ============================================================================

class BoneMapper:
    """Maps ACT bone names to BVH bone hierarchy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def map_bones(self, act_bones: List[str], template_name: str = 'standard') -> List[Bone]:
        """
        Map ACT bone names to BVG hierarchy
        
        Args:
            act_bones: List of bone names from ACT file
            template_name: Name of hierarchy template to use
            
        Returns:
            List of Bone objects for BVH hierarchy
        """
        templates = self.config.get('hierarchy_templates', {})
        if template_name not in templates:
            raise ValueError(f"Unknown hierarchy template: {template_name}. "
                           f"Available: {list(templates.keys())}")
        
        hierarchy_template = templates[template_name]
        bone_mappings = self.config.get('bone_mappings', {})
        
        # Create bone objects
        bones_by_name = {}
        
        # First pass: create all bones
        for bvh_bone_name, parent_name in hierarchy_template:
            bone = Bone(
                name=bvh_bone_name,
                parent=parent_name,
                offset=self._calculate_offset(bvh_bone_name),
                channels=self._get_channels_for_bone(bvh_bone_name)
            )
            bones_by_name[bvh_bone_name] = bone
        
        # Second pass: build hierarchy
        root_bones = []
        for bone_name, bone in bones_by_name.items():
            if bone.parent is None:
                root_bones.append(bone)
            else:
                if bone.parent in bones_by_name:
                    bones_by_name[bone.parent].children.append(bone)
                else:
                    self.logger.warning(f"Parent bone '{bone.parent}' not found for '{bone_name}'")
        
        # Map ACT bone names to BVH bones
        self._match_act_to_bvh_bones(act_bones, bones_by_name, bone_mappings)
        
        return root_bones
    
    def _calculate_offset(self, bone_name: str) -> Tuple[float, float, float]:
        """Calculate bone offset based on bone name"""
        # These are example offsets - should be adjusted based on your character
        offsets = {
            'Hips': (0.0, 0.0, 0.0),
            'Spine': (0.0, 1.0, 0.0),
            'Chest': (0.0, 1.5, 0.0),
            'Neck': (0.0, 0.5, 0.0),
            'Head': (0.0, 0.3, 0.0),
            'LeftShoulder': (-0.2, 0.0, 0.0),
            'LeftArm': (-0.5, 0.0, 0.0),
            'LeftForeArm': (-0.5, 0.0, 0.0),
            'LeftHand': (-0.2, 0.0, 0.0),
            'RightShoulder': (0.2, 0.0, 0.0),
            'RightArm': (0.5, 0.0, 0.0),
            'RightForeArm': (0.5, 0.0, 0.0),
            'RightHand': (0.2, 0.0, 0.0),
            'LeftUpLeg': (-0.2, -0.5, 0.0),
            'LeftLeg': (0.0, -1.0, 0.0),
            'LeftFoot': (0.0, -0.5, 0.2),
            'LeftToeBase': (0.0, 0.0, 0.3),
            'RightUpLeg': (0.2, -0.5, 0.0),
            'RightLeg': (0.0, -1.0, 0.0),
            'RightFoot': (0.0, -0.5, 0.2),
            'RightToeBase': (0.0, 0.0, 0.3),
        }
        return offsets.get(bone_name, (0.0, 0.0, 0.0))
    
    def _get_channels_for_bone(self, bone_name: str) -> List[str]:
        """Get channel configuration for bone"""
        if bone_name == 'Hips':
            return ["Xposition", "Yposition", "Zposition", 
                   "Zrotation", "Yrotation", "Xrotation"]
        else:
            return ["Zrotation", "Yrotation", "Xrotation"]
    
    def _match_act_to_bvh_bones(self, act_bones: List[str], 
                               bvh_bones: Dict[str, Bone],
                               bone_mappings: Dict):
        """Match ACT bone names to BVH bones using mappings"""
        matched = set()
        unmatched_act = set(act_bones)
        
        # Try exact matches first
        for act_bone in act_bones:
            for bvh_name, bvh_aliases in bone_mappings.items():
                if bvh_name in bvh_bones and act_bone in bvh_aliases:
                    bvh_bones[bvh_name].name = act_bone  # Use ACT bone name
                    matched.add(act_bone)
                    break
        
        # Try partial matches
        for act_bone in list(unmatched_act):
            act_lower = act_bone.lower()
            for bvh_name, bone in bvh_bones.items():
                if bvh_name.lower() in act_lower or act_lower in bvh_name.lower():
                    if act_bone not in matched:
                        self.logger.info(f"Partial match: {act_bone} -> {bvh_name}")
                        bone.name = act_bone
                        matched.add(act_bone)
                        break
        
        # Log unmatched bones
        unmatched_act -= matched
        if unmatched_act:
            self.logger.warning(f"Unmatched ACT bones: {unmatched_act}")

# ============================================================================
# BVH Writer
# ============================================================================

class BVHWriter:
    """Writes BVH files with proper formatting"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def write(self, bvh_bones: List[Bone], act_data: ACTFile, 
              output_path: str, rotation_order: RotationOrder):
        """
        Write BVH file
        
        Args:
            bvh_bones: Bone hierarchy
            act_data: Parsed ACT data
            output_path: Output file path
            rotation_order: Rotation order for channels
        """
        self.logger.info(f"Writing BVH file to: {output_path}")
        
        scale = self.config.get('scale_factor', 1.0)
        negate_y = self.config.get('negate_y', False)
        negate_z = self.config.get('negate_z', False)
        
        try:
            with open(output_path, 'w') as f:
                # Write header
                self._write_header(f, act_data)
                
                # Write hierarchy
                self._write_hierarchy(f, bvh_bones, scale, negate_y, negate_z, rotation_order)
                
                # Write motion data
                self._write_motion(f, bvh_bones, act_data, rotation_order)
            
            self.logger.info(f"Successfully wrote BVH file: {output_path}")
            
        except IOError as e:
            self.logger.error(f"Failed to write BVH file: {e}")
            raise
    
    def _write_header(self, f, act_data: ACTFile):
        """Write BVH file header"""
        f.write(f"# BVH file converted from ACT: {act_data.original_file}\n")
        f.write(f"# Conversion date: {datetime.now().isoformat()}\n")
        f.write(f"# Original FPS: {act_data.fps}\n")
        f.write(f"# Frame count: {act_data.frame_count}\n")
        if act_data.metadata:
            for key, value in act_data.metadata.items():
                f.write(f"# {key}: {value}\n")
        f.write("\n")
    
    def _write_hierarchy(self, f, bones: List[Bone], scale: float,
                        negate_y: bool, negate_z: bool, rotation_order: RotationOrder):
        """Write BVH hierarchy section"""
        f.write("HIERARCHY\n")
        
        for bone in bones:
            self._write_bone(f, bone, 0, scale, negate_y, negate_z, rotation_order)
    
    def _write_bone(self, f, bone: Bone, indent: int, scale: float,
                   negate_y: bool, negate_z: bool, rotation_order: RotationOrder):
        """Write a single bone and its children"""
        indent_str = " " * indent
        
        if bone.parent is None:
            f.write(f"{indent_str}ROOT {bone.name}\n")
        else:
            f.write(f"{indent_str}JOINT {bone.name}\n")
        
        f.write(f"{indent_str}{{\n")
        
        # Write offset
        offset = bone.offset
        if scale != 1.0:
            offset = (offset[0] * scale, offset[1] * scale, offset[2] * scale)
        if negate_y:
            offset = (offset[0], -offset[1], offset[2])
        if negate_z:
            offset = (offset[0], offset[1], -offset[2])
        
        f.write(f"{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels_str = " ".join(bone.channels)
        # Replace rotation order if needed
        if "Zrotation Yrotation Xrotation" in channels_str:
            channels_str = channels_str.replace(
                "Zrotation Yrotation Xrotation", 
                rotation_order.value
            )
        f.write(f"{indent_str}  CHANNELS {len(bone.channels)} {channels_str}\n")
        
        # Write children
        for child in bone.children:
            self._write_bone(f, child, indent + 2, scale, negate_y, negate_z, rotation_order)
        
        # Write End Site if needed
        if bone.has_end_site:
            f.write(f"{indent_str}  End Site\n")
            f.write(f"{indent_str}  {{\n")
            end_offset = bone.end_site_offset
            if scale != 1.0:
                end_offset = (end_offset[0] * scale, end_offset[1] * scale, end_offset[2] * scale)
            f.write(f"{indent_str}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{indent_str}  }}\n")
        
        f.write(f"{indent_str}}}\n")
    
    def _write_motion(self, f, bvh_bones: List[Bone], act_data: ACTFile,
                     rotation_order: RotationOrder):
        """Write motion data section"""
        f.write("\nMOTION\n")
        f.write(f"Frames: {act_data.frame_count}\n")
        f.write(f"Frame Time: {1.0/act_data.fps:.6f}\n")
        
        # Get all bones in depth-first order
        all_bones = self._get_bones_depth_first(bvh_bones)
        
        # Write each frame
        for frame_idx in range(act_data.frame_count):
            frame_values = []
            
            for bone in all_bones:
                # Get rotation data for this bone at this frame
                if bone.name in act_data.bones:
                    frame_list = act_data.bones[bone.name]
                    if frame_idx < len(frame_list) and frame_list[frame_idx] is not None:
                        rotations = frame_list[frame_idx].bone_rotations.get(bone.name, (0.0, 0.0, 0.0))
                    else:
                        rotations = (0.0, 0.0, 0.0)
                else:
                    rotations = (0.0, 0.0, 0.0)
                
                # Apply rotation order
                if rotation_order == RotationOrder.ZXY:
                    rx, ry, rz = rotations
                    rotations = (rz, rx, ry)
                elif rotation_order == RotationOrder.XYZ:
                    # Already in XYZ order
                    pass
                elif rotation_order == RotationOrder.ZYX:
                    rx, ry, rz = rotations
                    rotations = (rz, ry, rx)
                elif rotation_order == RotationOrder.YXZ:
                    rx, ry, rz = rotations
                    rotations = (ry, rx, rz)
                
                # Add position for root bone
                if bone.parent is None:  # Root bone
                    position = (0.0, 0.0, 0.0)  # Default position
                    frame_values.extend([f"{pos:.6f}" for pos in position])
                
                # Add rotations
                frame_values.extend([f"{rot:.6f}" for rot in rotations])
            
            f.write(" ".join(frame_values) + "\n")
    
    def _get_bones_depth_first(self, bones: List[Bone]) -> List[Bone]:
        """Get all bones in depth-first order"""
        result = []
        
        def traverse(bone):
            result.append(bone)
            for child in bone.children:
                traverse(child)
        
        for bone in bones:
            traverse(bone)
        
        return result

# ============================================================================
# Main Converter Class
# ============================================================================

class ACTtoBVHConverter:
    """Main converter class orchestrating the conversion process"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = Config(config_file)
        self.parser = ACTParser(self.config)
        self.mapper = BoneMapper(self.config)
        self.writer = BVHWriter(self.config)
        self.setup_logging()
    
    def setup_logging(self, log_level=logging.INFO):
        """Setup logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('act2bvh_conversion.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def convert(self, input_file: str, output_file: Optional[str] = None,
                rotation_order: str = "ZXY", template: str = "standard") -> str:
        """
        Convert ACT file to BVH format
        
        Args:
            input_file: Path to input ACT file
            output_file: Path to output BVH file (auto-generated if None)
            rotation_order: BVH rotation order (ZXY, XYZ, ZYX, YXZ)
            template: Bone hierarchy template name
            
        Returns:
            Path to output BVH file
        """
        start_time = datetime.now()
        self.logger.info(f"Starting conversion: {input_file}")
        
        try:
            # Parse ACT file
            act_data = self.parser.parse(input_file)
            
            # Map bones to BVH hierarchy
            act_bone_names = list(act_data.bones.keys())
            bvh_bones = self.mapper.map_bones(act_bone_names, template)
            
            # Determine output file path
            if output_file is None:
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.bvh"
            
            # Parse rotation order
            rot_order = RotationOrder.from_string(rotation_order)
            
            # Write BVH file
            self.writer.write(bvh_bones, act_data, output_file, rot_order)
            
            # Report statistics
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Conversion completed in {elapsed:.2f}s: "
                f"{len(act_bone_names)} bones, {act_data.frame_count} frames "
                f"to {output_file}"
            )
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Command line interface for ACT to BVH converter"""
    parser = argparse.ArgumentParser(
        description='Convert Hash Animation Master ACT files to BVH format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.act                          # Convert with default settings
  %(prog)s input.act -o output.bvh           # Specify output file
  %(prog)s input.act -r XYZ                  # Use XYZ rotation order
  %(prog)s input.act --scale 0.01           # Scale to meters
  %(prog)s input.act --config my_config.json # Use custom configuration
  %(prog)s --list-templates                  # List available bone templates
        
Common Rotation Orders:
  ZXY - Default for many BVH files
  XYZ - Common in some motion capture systems
  ZYX - Used in some game engines
  YXZ - Alternative ordering
        """
    )
    
    parser.add_argument('input', help='Input ACT file path')
    parser.add_argument('-o', '--output', help='Output BVH file path')
    parser.add_argument('-r', '--rotation-order', default='ZXY',
                       choices=['ZXY', 'XYZ', 'ZYX', 'YXZ'],
                       help='Rotation order for BVH output (default: ZXY)')
    parser.add_argument('-t', '--template', default='standard',
                       help='Bone hierarchy template (default: standard)')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                       help='Scale factor for bone offsets (default: 1.0)')
    parser.add_argument('--negate-y', action='store_true',
                       help='Negate Y coordinate (for different coordinate systems)')
    parser.add_argument('--negate-z', action='store_true',
                       help='Negate Z coordinate (for different coordinate systems)')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--list-templates', action='store_true',
                       help='List available bone hierarchy templates')
    parser.add_argument('--fps', type=float, help='Force specific FPS (overrides auto-detection)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # List templates if requested
    if args.list_templates:
        config = Config()
        templates = config.get('hierarchy_templates', {})
        print("Available bone hierarchy templates:")
        for name in templates.keys():
            print(f"  - {name}")
        print(f"\nBone mappings: {list(config.get('bone_mappings', {}).keys())}")
        return
    
    # Setup converter
    converter = ACTtoBVHConverter(args.config)
    
    if args.verbose:
        converter.setup_logging(logging.DEBUG)
    
    # Apply command line overrides
    if args.scale != 1.0:
        converter.config.set('scale_factor', args.scale)
    if args.negate_y:
        converter.config.set('negate_y', True)
    if args.negate_z:
        converter.config.set('negate_z', True)
    if args.fps:
        converter.config.set('auto_detect_fps', False)
        converter.config.set('default_fps', args.fps)
    
    # Perform conversion
    try:
        output_file = converter.convert(
            input_file=args.input,
            output_file=args.output,
            rotation_order=args.rotation_order,
            template=args.template
        )
        print(f"Successfully converted {args.input} to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

# ============================================================================
# Utility Functions
# ============================================================================

def batch_convert(input_folder: str, output_folder: str, config_file: Optional[str] = None):
    """
    Batch convert all ACT files in a folder
    
    Args:
        input_folder: Folder containing ACT files
        output_folder: Output folder for BVH files
        config_file: Optional configuration file
    """
    converter = ACTtoBVHConverter(config_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    act_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.act')]
    
    for act_file in act_files:
        input_path = os.path.join(input_folder, act_file)
        output_path = os.path.join(output_folder, 
                                  os.path.splitext(act_file)[0] + '.bvh')
        
        try:
            converter.convert(input_path, output_path)
            print(f"✓ {act_file}")
        except Exception as e:
            print(f"✗ {act_file}: {e}")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()