#!/usr/bin/env python3
"""
Simple BVH parser test
"""

import os

def parse_bvh_hierarchy(lines, i, parent_name, bones_data, channel_order):
    """Recursively parse BVH hierarchy"""
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line:
            continue
            
        if line.startswith('ROOT') or line.startswith('JOINT'):
            # Parse joint name
            parts = line.split()
            joint_type = parts[0]
            joint_name = parts[1]
            
            # Skip opening brace
            while i < len(lines) and '{' not in lines[i]:
                i += 1
            i += 1
            
            # Parse OFFSET
            offset_line = lines[i].strip()
            i += 1
            if offset_line.startswith('OFFSET'):
                offset_parts = offset_line.split()
                offset = [float(offset_parts[1]), float(offset_parts[2]), float(offset_parts[3])]
            else:
                offset = [0, 0, 0]
            
            # Parse CHANNELS
            channels_line = lines[i].strip()
            i += 1
            if channels_line.startswith('CHANNELS'):
                channels_parts = channels_line.split()
                num_channels = int(channels_parts[1])
                channels = channels_parts[2:2+num_channels]
            else:
                channels = []
            
            # Store bone data
            bones_data[joint_name] = {
                'parent': parent_name,
                'offset': offset,
                'channels': channels
            }
            channel_order.append(joint_name)
            
            # Recursively parse children
            i = parse_bvh_hierarchy(lines, i, joint_name, bones_data, channel_order)
            
        elif line.startswith('End Site'):
            # Skip End Site
            while i < len(lines) and '}' not in lines[i]:
                i += 1
            i += 1
            
        elif line.startswith('}'):
            # End of current joint
            return i
            
    return i

def test_bvh_parser():
    filepath = 'dataset-1_bow_happy_001.bvh'
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found")
        return
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    i = 0
    
    # Skip to HIERARCHY
    while i < len(lines) and not lines[i].strip().startswith('HIERARCHY'):
        i += 1
    i += 1  # Skip HIERARCHY line
    
    # Parse hierarchy
    bones_data = {}
    channel_order = []
    parse_bvh_hierarchy(lines, i, None, bones_data, channel_order)
    
    print(f"Found {len(bones_data)} bones:")
    for name, data in bones_data.items():
        print(f"  {name}: parent={data['parent']}, offset={data['offset']}, channels={data['channels']}")
    
    print(f"Channel order: {channel_order}")
    
    # Find MOTION section
    while i < len(lines) and not lines[i].strip().startswith('MOTION'):
        i += 1
    
    if i >= len(lines):
        print("No MOTION section found")
        return
    
    i += 1  # Skip MOTION line
    
    # Parse frames info
    frames_line = lines[i].strip()
    if frames_line.startswith('Frames:'):
        num_frames = int(frames_line.split(':')[1].strip())
        print(f"Frames: {num_frames}")
    i += 1
    
    frame_time_line = lines[i].strip()
    if frame_time_line.startswith('Frame Time:'):
        frame_time = float(frame_time_line.split(':')[1].strip())
        print(f"Frame Time: {frame_time}")
    i += 1
    
    # Count motion data lines
    motion_lines = 0
    while i < len(lines):
        line = lines[i].strip()
        if line:
            motion_lines += 1
        i += 1
    
    print(f"Motion data lines: {motion_lines}")

if __name__ == "__main__":
    test_bvh_parser()