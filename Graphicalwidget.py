
# ============================================================================
# IMPORTS
# ============================================================================

import math
import time
import re
from typing import List, Optional, Dict, Any, Tuple
import random
from collections import defaultdict, Counter
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QScrollArea, QFrame, QTreeWidget, QTreeWidgetItem,
                             QAbstractItemView, QDockWidget, QMenu, QMessageBox, QInputDialog,
                             QTextEdit, QLineEdit, QDialog, QSplitter, QGroupBox, QColorDialog, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint, QRect, QSize, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap, QImage, QMouseEvent, QIcon

# Try to import optional dependencies
try:
    from Model import CapsuleManager, Capsule
    CAPSULE_MANAGER_AVAILABLE = True
except ImportError:
    CapsuleManager = None
    Capsule = None
    CAPSULE_MANAGER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# GraphManager class will be passed from main.py
GraphManager = None  # Will be set when widget is created

# Import ROCA components
try:
    from roca.graph_manager import Command, CommandType
except ImportError:
    Command = None
    CommandType = None

# ============================================================================
# TIMELINE WIDGET
# ============================================================================

# ============================================================================
# TIMELINE WIDGET
# ============================================================================

class TimelineWidget(QWidget):
    """Timeline for animation frames."""
    
    frame_selected = pyqtSignal(int)
    frame_duplicate_requested = pyqtSignal(int)
    frame_delete_requested = pyqtSignal(int)
    
    # Timeline constants
    TIMELINE_HEIGHT = 120
    FRAME_WIDTH_DEFAULT = 100
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = None
        self.current_frame_index = 0
        self.frame_width = 100
        self.zoom_level = 1.0
        self.selected_frames = set()  # For multi-selection
        
        # Onion skinning settings
        self.onion_skinning_enabled = False
        self.onion_skin_frames_before = 1
        self.onion_skin_frames_after = 1
        self.onion_skin_opacity = 0.3
        
        # Motion paths
        self.show_motion_paths = False
        self.motion_path_length = 10  # frames
        
        # UI setup
        self.setup_ui()
    
    @classmethod
    def set_animation_class(cls, animation_class):
        """Set the Animation class to use (called from main.py)."""
        global Animation
        Animation = animation_class
        
    def setup_ui(self):
        """Setup timeline UI."""
        layout = QVBoxLayout(self)
        
        # Control bar
        control_bar = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.clicked.connect(self.go_to_prev_frame)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.go_to_next_frame)
        
        self.generate_inbetweens_button = QPushButton("Generate In-betweens")
        self.generate_inbetweens_button.clicked.connect(self.generate_inbetweens)
        self.generate_inbetweens_button.setEnabled(False)
        
        self.pose_button = QPushButton("Create Pose")
        self.pose_button.clicked.connect(self.create_pose_capsule)
        
        self.frame_label = QLabel("Frame: 1/1")
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.set_zoom)
        
        control_bar.addWidget(self.play_button)
        control_bar.addWidget(self.prev_button)
        control_bar.addWidget(self.next_button)
        control_bar.addWidget(self.generate_inbetweens_button)
        control_bar.addWidget(self.pose_button)
        control_bar.addStretch()
        control_bar.addWidget(self.frame_label)
        control_bar.addStretch()
        control_bar.addWidget(QLabel("Zoom:"))
        control_bar.addWidget(self.zoom_slider)
        
        # Scroll area for frames
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(self.TIMELINE_HEIGHT)
        
        self.frame_container = QWidget()
        self.frame_layout = QHBoxLayout(self.frame_container)
        self.frame_layout.setSpacing(2)
        self.frame_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scroll_area.setWidget(self.frame_container)
        
        layout.addLayout(control_bar)
        layout.addWidget(self.scroll_area)
        
        # Playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.advance_playback)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
    def set_animation(self, animation):
        """Set animation to display."""
        self.animation = animation
        self.update_timeline()
        
        # Connect signals
        if animation:
            animation.frames_changed.connect(self.update_timeline)
            animation.current_frame_changed.connect(self.set_current_frame)
            
    def update_timeline(self):
        """Update timeline display."""
        # Clear existing frames
        while self.frame_layout.count():
            item = self.frame_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not self.animation:
            return
            
        # Add frame widgets
        for i, frame in enumerate(self.animation.frames):
            frame_widget = FrameWidget(frame, i, self.frame_width)
            frame_widget.selected.connect(lambda idx: self.frame_selected.emit(idx))
            frame_widget.duplicate_requested.connect(self.frame_duplicate_requested.emit)
            frame_widget.delete_requested.connect(self.frame_delete_requested.emit)
            self.frame_layout.addWidget(frame_widget)
            
        # Update frame label
        self.update_frame_label()
        
    def update_frame_label(self):
        """Update frame label text."""
        if self.animation:
            total = self.animation.get_frame_count()
            current = self.animation.current_frame_index + 1
            self.frame_label.setText(f"Frame: {current}/{total}")
            
    def set_current_frame(self, index: int):
        """Set current frame visually."""
        self.current_frame_index = index
        
        # Update all frame widgets
        for i in range(self.frame_layout.count()):
            widget = self.frame_layout.itemAt(i).widget()
            if isinstance(widget, FrameWidget):
                widget.set_selected(widget.frame_index == index or widget.frame_index in self.selected_frames)
                
        self.update_frame_label()
        self.update_generate_button()
        
        # Scroll to current frame
        self.scroll_to_frame(index)
        
    def update_frame_selections(self):
        """Update visual selection state for all frames."""
        for i in range(self.frame_layout.count()):
            widget = self.frame_layout.itemAt(i).widget()
            if isinstance(widget, FrameWidget):
                widget.set_selected(widget.frame_index in self.selected_frames)
        self.update_generate_button()
        
    def update_generate_button(self):
        """Enable/disable generate in-betweens button based on selection."""
        self.generate_inbetweens_button.setEnabled(len(self.selected_frames) >= 2)
        
    def scroll_to_frame(self, index: int):
        """Scroll to make frame visible."""
        frame_pos = index * (self.frame_width + 2)  # +2 for spacing
        scrollbar = self.scroll_area.horizontalScrollBar()
        
        # Calculate visible range
        visible_start = scrollbar.value()
        visible_end = visible_start + self.scroll_area.viewport().width()
        
        if frame_pos < visible_start or frame_pos + self.frame_width > visible_end:
            # Center the frame
            target = frame_pos - (self.scroll_area.viewport().width() - self.frame_width) // 2
            scrollbar.setValue(max(0, target))
            
    def set_zoom(self, value: int):
        """Set zoom level."""
        self.zoom_level = value / 100.0
        self.frame_width = int(80 * self.zoom_level)
        self.update_timeline()
        
    def toggle_playback(self):
        """Toggle playback."""
        if self.play_button.isChecked():
            self.playback_timer.start(1000 // (self.animation.frame_rate if self.animation else 24))
            self.play_button.setText("⏸ Pause")
        else:
            self.playback_timer.stop()
            self.play_button.setText("▶ Play")
            
    def advance_playback(self):
        """Advance to next frame during playback."""
        if self.animation:
            next_index = (self.animation.current_frame_index + 1) % self.animation.get_frame_count()
            self.animation.set_current_frame(next_index)
            
    def go_to_prev_frame(self):
        """Go to previous frame."""
        if self.animation and self.animation.current_frame_index > 0:
            self.animation.set_current_frame(self.animation.current_frame_index - 1)
            
    def go_to_next_frame(self):
        """Go to next frame."""
        if self.animation:
            next_index = min(self.animation.current_frame_index + 1, 
                           self.animation.get_frame_count() - 1)
            self.animation.set_current_frame(next_index)
            
    def dropEvent(self, event):
        """Handle capsule drops on timeline."""
        if event.mimeData().hasFormat("application/x-capsule"):
            capsule_uuid = event.mimeData().data("application/x-capsule").data().decode()
            if hasattr(self.parent(), 'capsule_manager'):
                capsule = self.parent().capsule_manager.get_capsule_by_uuid(capsule_uuid)
                if capsule:
                    self._handle_capsule_drop(capsule, event.pos())
                    event.acceptProposedAction()
                    return
        event.ignore()
        
    def _handle_capsule_drop(self, capsule, position):
        """Handle dropping a capsule onto the timeline."""
        # For character capsules, add as keyframe or reference
        if capsule.type == "character":
            # Find which frame was dropped on
            frame_index = self._get_frame_at_position(position)
            if frame_index >= 0 and self.animation:
                # Add character reference to this frame
                if not hasattr(self.animation.frames[frame_index], 'character_refs'):
                    self.animation.frames[frame_index].character_refs = []
                if capsule.uuid not in [ref['uuid'] for ref in self.animation.frames[frame_index].character_refs]:
                    self.animation.frames[frame_index].character_refs.append({
                        'uuid': capsule.uuid,
                        'name': capsule.name,
                        'type': capsule.type
                    })
                    if hasattr(self.parent(), 'status_bar'):
                        self.parent().status_bar.showMessage(f"Added {capsule.name} to frame {frame_index + 1}", 3000)
                        
        # For skill capsules, apply animation technique
        elif capsule.type == "skill":
            if hasattr(self.parent(), 'status_bar'):
                self.parent().status_bar.showMessage(f"Applied {capsule.name} animation technique", 3000)
                
        # For style capsules, apply to animation
        elif capsule.type == "style":
            if hasattr(self.parent(), 'status_bar'):
                self.parent().status_bar.showMessage(f"Applied {capsule.name} style to animation", 3000)
                
        # For animation capsules, add motion data
        elif capsule.type == "animation":
            frame_index = self._get_frame_at_position(position)
            if frame_index >= 0 and self.animation:
                if not hasattr(self.animation.frames[frame_index], 'animation_capsules'):
                    self.animation.frames[frame_index].animation_capsules = []
                if capsule.uuid not in [ref['uuid'] for ref in self.animation.frames[frame_index].animation_capsules]:
                    self.animation.frames[frame_index].animation_capsules.append({
                        'uuid': capsule.uuid,
                        'name': capsule.name,
                        'type': capsule.type
                    })
                    
                    # Check if we have both character and animation - create merged capsule
                    self._check_and_create_character_animation_merge(frame_index)
                    
                    if hasattr(self.parent(), 'status_bar'):
                        self.parent().status_bar.showMessage(f"Added animation {capsule.name} to frame {frame_index + 1}", 3000)
                        
    def _check_and_create_character_animation_merge(self, frame_index: int):
        """Check if we have both character and animation capsules and create a merged capsule."""
        if not self.animation or not hasattr(self.parent(), 'capsule_manager'):
            return
            
        frame = self.animation.frames[frame_index]
        capsule_manager = self.parent().capsule_manager
        
        # Get character and animation capsules from this frame
        character_capsules = getattr(frame, 'character_refs', [])
        animation_capsules = getattr(frame, 'animation_capsules', [])
        
        if not character_capsules or not animation_capsules:
            return
            
        # Create merged capsules for each character-animation pair
        for char_data in character_capsules:
            for anim_data in animation_capsules:
                char_capsule = capsule_manager.get_capsule_by_uuid(char_data['uuid'])
                anim_capsule = capsule_manager.get_capsule_by_uuid(anim_data['uuid'])
                
                if char_capsule and anim_capsule:
                    self._create_merged_character_animation_capsule(char_capsule, anim_capsule)
                    
    def _create_merged_character_animation_capsule(self, character_capsule, animation_capsule):
        """Create a merged capsule combining character and animation."""
        if not hasattr(self.parent(), 'capsule_manager'):
            return
            
        capsule_manager = self.parent().capsule_manager
        
        # Create merged capsule name
        merge_name = f"{character_capsule.name}_{animation_capsule.name}"
        
        # Check if merged capsule already exists
        existing = capsule_manager.get_capsule_by_name(merge_name)
        if existing:
            if hasattr(self.parent(), 'status_bar'):
                self.parent().status_bar.showMessage(f"Merged capsule '{merge_name}' already exists", 3000)
            return
            
        # Create component list for merging
        component_capsules = [
            character_capsule.to_dict() if hasattr(character_capsule, 'to_dict') else {
                'name': character_capsule.name,
                'capsule_type': character_capsule.type,
                'vector': character_capsule.vector.tolist() if hasattr(character_capsule.vector, 'tolist') else character_capsule.vector,
                'metadata': character_capsule.metadata,
                'usage_count': getattr(character_capsule, 'usage_count', 1),
                'strength': getattr(character_capsule, 'strength', 1.0)
            },
            animation_capsule.to_dict() if hasattr(animation_capsule, 'to_dict') else {
                'name': animation_capsule.name,
                'capsule_type': animation_capsule.type,
                'vector': animation_capsule.vector.tolist() if hasattr(animation_capsule.vector, 'tolist') else animation_capsule.vector,
                'metadata': animation_capsule.metadata,
                'usage_count': getattr(animation_capsule, 'usage_count', 1),
                'strength': getattr(animation_capsule, 'strength', 1.0)
            }
        ]
        
        # Use the autonomous brain's merge functionality if available
        if hasattr(self.parent(), 'autonomous_brain'):
            merged_capsule = self.parent().autonomous_brain._create_merged_capsule(
                component_capsules,
                merge_name,
                "character_animation",
                f"Combined {character_capsule.name} with {animation_capsule.name} animation",
                "timeline_combination"
            )
            
            if merged_capsule:
                capsule_manager.add_capsule(merged_capsule)
                if hasattr(self.parent(), 'status_bar'):
                    self.parent().status_bar.showMessage(f"Created merged capsule: {merge_name}", 5000)
        else:
            # Fallback: create simple merged capsule
            if NUMPY_AVAILABLE:
                merged_vector = np.mean([np.array(c['vector']) for c in component_capsules], axis=0)
            else:
                # Simple average without numpy
                vectors = [c['vector'] for c in component_capsules]
                merged_vector = [sum(x) / len(x) for x in zip(*vectors)]
            
            merged_capsule = Capsule(
                name=merge_name,
                capsule_type="character_animation",
                vector_dim=len(merged_vector),
                metadata={
                    'description': f"Combined {character_capsule.name} with {animation_capsule.name} animation",
                    'character_component': character_capsule.name,
                    'animation_component': animation_capsule.name,
                    'created_from_timeline': True
                }
            )
            merged_capsule.vector = merged_vector
            
            capsule_manager.add_capsule(merged_capsule)
            if hasattr(self.parent(), 'status_bar'):
                self.parent().status_bar.showMessage(f"Created merged capsule: {merge_name}", 5000)
                        
    def generate_inbetweens(self):
        """Generate in-between frames using ROCA."""
        if len(self.selected_frames) < 2 or not self.animation:
            return
            
        selected_indices = sorted(list(self.selected_frames))
        
        # Generate in-betweens between consecutive selected frames
        for i in range(len(selected_indices) - 1):
            start_idx = selected_indices[i]
            end_idx = selected_indices[i + 1]
            
            if end_idx - start_idx > 1:
                self._generate_inbetweens_between(start_idx, end_idx)
                
        # Clear selection and refresh
        self.selected_frames.clear()
        self.update_frame_selections()
        self.update_timeline()
        
    def _generate_inbetweens_between(self, start_idx: int, end_idx: int):
        """Generate in-between frames between two keyframes."""
        start_frame = self.animation.frames[start_idx]
        end_frame = self.animation.frames[end_idx]
        
        # Get capsules from frames
        start_capsules = getattr(start_frame, 'character_refs', []) + getattr(start_frame, 'motion_capsules', [])
        end_capsules = getattr(end_frame, 'character_refs', []) + getattr(end_frame, 'motion_capsules', [])
        
        # Use ROCA to create in-between capsules
        if hasattr(self.parent(), 'roca_graph') and hasattr(self.parent(), 'graph_manager'):
            roca_graph = self.parent().roca_graph
            graph_manager = self.parent().graph_manager
            
            # Create in-between capsules for each pair
            for start_cap_data in start_capsules:
                for end_cap_data in end_capsules:
                    if start_cap_data.get('type') == end_cap_data.get('type'):
                        # Create in-between capsule
                        command = Command(
                            CommandType.INSERT_INBETWEEN,
                            {
                                'a_id': start_cap_data['uuid'],
                                'b_id': end_cap_data['uuid']
                            }
                        )
                        result = graph_manager.execute(command)
                        if result.ok:
                            inbetween_capsule = roca_graph.capsules.get(result.data['id'])
                            if inbetween_capsule:
                                # Add to intermediate frames
                                num_inbetweens = end_idx - start_idx - 1
                                for j in range(1, num_inbetweens + 1):
                                    frame_idx = start_idx + j
                                    if frame_idx < len(self.animation.frames):
                                        frame = self.animation.frames[frame_idx]
                                        if not hasattr(frame, 'motion_capsules'):
                                            frame.motion_capsules = []
                                        frame.motion_capsules.append({
                                            'uuid': inbetween_capsule.id,
                                            'name': inbetween_capsule.metadata.get('name', 'In-between'),
                                            'type': 'motion'
                                        })
                                        
    def create_pose_capsule(self):
        """Create a motion capsule from current frame pose."""
        if not self.animation or self.current_frame_index < 0:
            return
            
        current_frame = self.animation.frames[self.current_frame_index]
        
        # Create motion data from current frame
        motion_data = {
            'frame_index': self.current_frame_index,
            'timestamp': time.time(),
            'character_refs': getattr(current_frame, 'character_refs', []),
            'bone_transforms': getattr(current_frame, 'bone_transforms', {}),
            'model_transform': getattr(current_frame, 'model_transform', {}),
        }
        
        # Create capsule
        capsule_name = f"Pose_Frame_{self.current_frame_index + 1}"
        motion_capsule = Capsule(
            name=capsule_name,
            capsule_type="motion",
            metadata={
                'motion_data': motion_data,
                'created_from_frame': self.current_frame_index,
                'type': 'motion'
            }
        )
        
        # Add to capsule manager
        if hasattr(self.parent(), 'capsule_manager'):
            self.parent().capsule_manager.add_capsule(motion_capsule)
            
        # Add to ROCA graph
        if hasattr(self.parent(), 'graph_manager'):
            command = Command(CommandType.ADD_CAPSULE, {'capsule': motion_capsule})
            self.parent().graph_manager.execute(command)
            
        # Add to current frame
        if not hasattr(current_frame, 'motion_capsules'):
            current_frame.motion_capsules = []
        current_frame.motion_capsules.append({
            'uuid': motion_capsule.id,
            'name': motion_capsule.name,
            'type': 'motion'
        })
        
        if hasattr(self.parent(), 'status_bar'):
            self.parent().status_bar.showMessage(f"Created motion capsule: {capsule_name}", 3000)
            
        self.update_timeline()
                
    def _get_frame_at_position(self, position):
        """Get frame index at drop position."""
        # Convert position to frame index
        scrollbar = self.scroll_area.horizontalScrollBar()
        scroll_offset = scrollbar.value()
        local_x = position.x() + scroll_offset
        
        frame_index = local_x // (self.frame_width + 2)  # +2 for spacing
        if self.animation and 0 <= frame_index < len(self.animation.frames):
            return frame_index
        return -1

    def set_onion_skinning_enabled(self, enabled: bool):
        """Enable or disable onion skinning."""
        self.onion_skinning_enabled = enabled
        # Emit signal to update viewport
        if hasattr(self, 'onion_skinning_changed'):
            self.onion_skinning_changed.emit(enabled, self.onion_skin_frames_before, self.onion_skin_frames_after, self.onion_skin_opacity)

    def set_onion_skin_frames(self, before: int, after: int):
        """Set number of onion skin frames before and after current frame."""
        self.onion_skin_frames_before = before
        self.onion_skin_frames_after = after
        if self.onion_skinning_enabled and hasattr(self, 'onion_skinning_changed'):
            self.onion_skinning_changed.emit(True, before, after, self.onion_skin_opacity)

    def set_onion_skin_opacity(self, opacity: float):
        """Set onion skin opacity (0.0 to 1.0)."""
        self.onion_skin_opacity = max(0.0, min(1.0, opacity))
        if self.onion_skinning_enabled and hasattr(self, 'onion_skinning_changed'):
            self.onion_skinning_changed.emit(True, self.onion_skin_frames_before, self.onion_skin_frames_after, opacity)

    def set_motion_paths_enabled(self, enabled: bool):
        """Enable or disable motion path display."""
        self.show_motion_paths = enabled
        # Emit signal to update viewport
        if hasattr(self, 'motion_paths_changed'):
            self.motion_paths_changed.emit(enabled, self.motion_path_length)

    def set_motion_path_length(self, length: int):
        """Set motion path length in frames."""
        self.motion_path_length = max(1, length)
        if self.show_motion_paths and hasattr(self, 'motion_paths_changed'):
            self.motion_paths_changed.emit(True, length)

    def get_onion_skin_frames(self):
        """Get list of frame indices for onion skinning."""
        if not self.onion_skinning_enabled or not self.animation:
            return []
        
        current = self.current_frame_index
        frames = []
        
        # Add frames before current
        for i in range(1, self.onion_skin_frames_before + 1):
            frame_idx = current - i
            if 0 <= frame_idx < len(self.animation.frames):
                frames.append(frame_idx)
        
        # Add frames after current
        for i in range(1, self.onion_skin_frames_after + 1):
            frame_idx = current + i
            if 0 <= frame_idx < len(self.animation.frames):
                frames.append(frame_idx)
        
        return frames

    def get_motion_path_data(self, bone_name: str):
        """Get motion path data for a specific bone."""
        if not self.show_motion_paths or not self.animation:
            return []
        
        current = self.current_frame_index
        path_data = []
        
        # Get positions for motion path
        for i in range(max(0, current - self.motion_path_length), min(len(self.animation.frames), current + self.motion_path_length + 1)):
            frame = self.animation.frames[i]
            if hasattr(frame, 'get_bone_position'):
                pos = frame.get_bone_position(bone_name)
                if pos:
                    path_data.append((i, pos))
        
        return path_data


class FrameWidget(QWidget):
    """Widget for a single frame in timeline."""
    
    selected = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    
    def __init__(self, frame_data, index: int, width: int = 100):
        super().__init__()
        self.frame = frame_data  # Can be AnimationFrame or simple dict
        self.frame_index = index
        self.width = width
        self.is_selected = False
        
        self.setFixedSize(width, 100)
        self.setMouseTracking(True)
        
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse press for selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl+click for multi-selection
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                parent_timeline = self.parent().parent().parent()  # TimelineWidget
                if hasattr(parent_timeline, 'selected_frames'):
                    if self.frame_index in parent_timeline.selected_frames:
                        parent_timeline.selected_frames.remove(self.frame_index)
                    else:
                        parent_timeline.selected_frames.add(self.frame_index)
                    parent_timeline.update_frame_selections()
            else:
                # Single selection
                parent_timeline = self.parent().parent().parent()
                parent_timeline.selected_frames = {self.frame_index}
                parent_timeline.update_frame_selections()
                self.selected.emit(self.frame_index)
                
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.pos())
            
    def show_context_menu(self, pos):
        """Show context menu for frame."""
        menu = QMenu(self)
        
        duplicate_action = menu.addAction("Duplicate Frame")
        duplicate_action.triggered.connect(lambda: self.duplicate_requested.emit(self.frame_index))
        
        delete_action = menu.addAction("Delete Frame")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(self.frame_index))
        
        menu.exec(self.mapToGlobal(pos))
        
    def paintEvent(self, event):
        """Paint the frame widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        if self.is_selected:
            painter.fillRect(self.rect(), QColor(100, 150, 255, 100))
        else:
            painter.fillRect(self.rect(), QColor(60, 60, 60))
            
        # Border
        pen = QPen(QColor(100, 100, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # Frame number
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, 
                        str(self.frame_index + 1))
        
        # Capsule indicators
        if hasattr(self.frame, 'character_refs') and self.frame.character_refs:
            painter.setPen(QColor(255, 100, 100))
            painter.setBrush(QColor(255, 100, 100, 150))
            painter.drawEllipse(5, 25, 10, 10)
            
        if hasattr(self.frame, 'motion_capsule') and self.frame.motion_capsule:
            painter.setPen(QColor(100, 255, 100))
            painter.setBrush(QColor(100, 255, 100, 150))
            painter.drawEllipse(20, 25, 10, 10)


class CrossDimensionalMatcher(QObject):
    """Analyzes and matches 2D drawings with 3D models."""
    
    match_found = pyqtSignal(object, object, float)  # 2d_capsule, 3d_capsule, confidence
    
    def __init__(self, capsule_manager: "CapsuleManager", parent=None):
        super().__init__(parent)
        self.capsule_manager = capsule_manager
        self.matches = []
    
    def analyze_similarity(self, capsule_2d, capsule_3d) -> tuple:
        """
        Analyze similarity between 2D and 3D capsules.
        Returns: (confidence_score, factors_list)
        """
        score = 0.0
        factors = []
        
        # Factor 1: Filename/name similarity
        name_2d = capsule_2d.name.lower()
        name_3d = capsule_3d.name.lower()
        
        # Keywords that indicate 2D-3D correspondence
        object_keywords = [
            'chair', 'table', 'desk', 'sofa', 'couch', 'bench', 'seat',
            'lamp', 'light', 'door', 'window', 'shelf', 'cabinet',
            'character', 'figure', 'person', 'model'
        ]
        
        for keyword in object_keywords:
            if keyword in name_2d and keyword in name_3d:
                score += 0.35
                factors.append(f"Object keyword: {keyword}")
                break
        
        # Factor 2: Type compatibility
        compatible_pairs = [
            ('character', '3dmodel'),
            ('2dimage', '3dmodel'),
            ('character', 'character'),
        ]
        
        for type1, type2 in compatible_pairs:
            if (capsule_2d.type == type1 and capsule_3d.type == type2) or \
               (capsule_2d.type == type2 and capsule_3d.type == type1):
                score += 0.25
                factors.append(f"Type compatible: {capsule_2d.type} + {capsule_3d.type}")
                break
        
        # Factor 3: Temporal proximity (imported close in time)
        metadata_2d = capsule_2d.metadata or {}
        metadata_3d = capsule_3d.metadata or {}
        
        time_2d = metadata_2d.get('imported_at', 0)
        time_3d = metadata_3d.get('imported_at', 0)
        
        if time_2d and time_3d:
            time_diff = abs(time_2d - time_3d)
            if time_diff < 3600:  # Within 1 hour
                score += 0.20
                factors.append("Imported close in time")
            elif time_diff < 86400:  # Within 24 hours
                score += 0.10
                factors.append("Imported on same day")
        
        # Factor 4: File extension compatibility
        ext_2d = metadata_2d.get('file_extension', '').lower()
        ext_3d = metadata_3d.get('file_extension', '').lower()
        
        image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.exr', '.psd']
        model_exts = ['.obj', '.fbx', '.3ds', '.max', '.c4d', '.gltf', '.glb']
        
        if ext_2d in image_exts and ext_3d in model_exts:
            score += 0.15
            factors.append(f"Extension compatible: image + model")
        
        # Factor 5: Metadata object class tags
        classes_2d = metadata_2d.get('object_classes', [])
        classes_3d = metadata_3d.get('object_classes', [])
        
        if classes_2d and classes_3d:
            common_classes = set(classes_2d) & set(classes_3d)
            if common_classes:
                score += 0.15
                factors.append(f"Common classes: {', '.join(common_classes)}")
        
        # Normalize score to 0-1 range
        score = min(1.0, score)
        
        return score, factors
    
    def find_matches(self, min_confidence=0.5) -> list:
        """
        Scan all capsules and find 2D↔3D matches.
        Returns list of match dictionaries.
        """
        matches = []
        
        # Separate capsules by dimension
        capsules_2d = [c for c in self.capsule_manager.capsules 
                      if c.type in ['character', '2dimage', 'unassigned']]
        capsules_3d = [c for c in self.capsule_manager.capsules 
                      if c.type in ['3dmodel', 'character']]
        
        # Compare each 2D capsule with each 3D capsule
        for capsule_2d in capsules_2d:
            for capsule_3d in capsules_3d:
                # Skip if same capsule
                if capsule_2d.uuid == capsule_3d.uuid:
                    continue
                
                confidence, factors = self.analyze_similarity(capsule_2d, capsule_3d)
                
                if confidence >= min_confidence:
                    match = {
                        'capsule_2d': capsule_2d,
                        'capsule_3d': capsule_3d,
                        'confidence': confidence,
                        'factors': factors,
                        'timestamp': time.time()
                    }
                    matches.append(match)
                    self.match_found.emit(capsule_2d, capsule_3d, confidence)
                    
                    file_logger.debug(
                        f"Cross-dimensional match: {capsule_2d.name} ↔ {capsule_3d.name} "
                        f"(confidence: {confidence:.2%}) - Factors: {', '.join(factors)}"
                    )
        
        self.matches = matches
        return matches
    
    def create_match_edges(self, graph_manager) -> int:
        """
        Create graph edges for all matches found.
        Returns count of edges created.
        """
        edges_created = 0
        
        for match in self.matches:
            try:
                capsule_2d = match['capsule_2d']
                capsule_3d = match['capsule_3d']
                confidence = match['confidence']
                
                # Create metadata for the edge
                edge_metadata = {
                    'type': 'cross_dimensional_match',
                    'capsule_2d_id': capsule_2d.uuid,
                    'capsule_3d_id': capsule_3d.uuid,
                    'confidence': confidence,
                    'created_at': time.time(),
                    'factors': match['factors']
                }
                
                # Add to graph if graph manager is available
                if graph_manager and hasattr(graph_manager, 'graph'):
                    roca_logger.debug(
                        f"Created cross-dimensional edge: {capsule_2d.name} ↔ {capsule_3d.name}"
                    )
                    edges_created += 1
                    
            except Exception as e:
                file_logger.error(f"Error creating match edge: {e}")
        
        return edges_created
        
    def paintEvent(self, event):
        """Paint frame widget."""
        painter = QPainter(self)
        
        # Draw background
        if self.is_selected:
            painter.fillRect(self.rect(), QColor(220, 240, 255))
            painter.setPen(QPen(QColor(0, 120, 215), 2))
        else:
            painter.fillRect(self.rect(), QColor(245, 245, 245))
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            
        painter.drawRect(0, 0, self.width - 1, self.height() - 1)
        
        # Draw frame thumbnail
        thumb_size = min(self.width - 10, self.height() - 30)
        thumb_rect = QRectF(5, 5, thumb_size, thumb_size)
        
        # Try to get thumbnail if frame has the method, otherwise draw placeholder
        pixmap = None
        if hasattr(self.frame, 'get_thumbnail'):
            pixmap = self.frame.get_thumbnail(QSize(int(thumb_size), int(thumb_size)))
        elif isinstance(self.frame, dict) and 'thumbnail' in self.frame:
            # Handle dict with thumbnail data
            pixmap = self.frame['thumbnail']
            
        if pixmap and not pixmap.isNull():
            painter.drawPixmap(thumb_rect.toRect(), pixmap)
        else:
            # Draw placeholder
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.setBrush(QBrush(QColor(220, 220, 220)))
            painter.drawRect(thumb_rect.toRect())
            painter.drawText(thumb_rect.toRect(), Qt.AlignmentFlag.AlignCenter, "Frame")
        
        # Draw frame number
        painter.setPen(QColor(80, 80, 80))
        painter.drawText(5, self.height() - 10, f"{self.frame_index + 1}")
        
        painter.end()
        
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.frame_index)
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.pos())
            
    def show_context_menu(self, pos: QPoint):
        """Show context menu for frame."""
        menu = QMenu(self)
        
        duplicate_action = menu.addAction("Duplicate Frame")
        delete_action = menu.addAction("Delete Frame")
        menu.addSeparator()
        capture_pose_action = menu.addAction("Capture as Pose Capsule")
        
        action = menu.exec_(self.mapToGlobal(pos))
        
        if action == duplicate_action:
            self.duplicate_requested.emit(self.frame_index)
        elif action == delete_action:
            self.delete_requested.emit(self.frame_index)
        elif action == capture_pose_action:
            # This would be connected in the main window
            pass
            
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.is_selected = selected
        self.update()


class CapsuleTreeWidget(QTreeWidget):
    """Tree widget for displaying capsules."""
    
    capsule_selected = pyqtSignal(object)  # Capsule object
    capsule_double_clicked = pyqtSignal(object)
    
    # Color constants for capsule types
    COLORS = {
        'blue': QColor(100, 149, 237),
        'green': QColor(34, 139, 34),
        'red': QColor(220, 20, 60),
        'orange': QColor(255, 140, 0),
        'purple': QColor(138, 43, 226),
        'gray': QColor(128, 128, 128),
        'cyan': QColor(0, 206, 209),
        'pink': QColor(255, 20, 147)
    }
    
    CAPSULE_COLORS = {
        'character': COLORS['blue'],
        'skill': COLORS['green'],
        'style': COLORS['orange'],
        'personality': COLORS['purple'],
        '3dmodel': COLORS['cyan'],
        '2dimage': COLORS['pink'],
        'animation': COLORS['red'],
        'unassigned': COLORS['gray']
    }
    
    def __init__(self, capsule_manager=None, parent=None):
        super().__init__(parent)
        self.capsule_manager = capsule_manager
        self.setup_ui()
        
        # Connect signals if capsule manager is available
        if self.capsule_manager and hasattr(self.capsule_manager, 'capsules_changed'):
            self.capsule_manager.capsules_changed.connect(self.refresh)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
    
    @classmethod
    def set_capsule_manager_class(cls, capsule_manager_class):
        """Set the CapsuleManager class to use (called from main.py)."""
        global CapsuleManager
        CapsuleManager = capsule_manager_class
        
    def setup_ui(self):
        """Setup tree widget UI."""
        self.setHeaderLabels(["Name", "Type", "Usage", "Score"])
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Enable drag operations
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        
        # Set column widths
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 80)
        
        self.refresh()
        
    def refresh(self):
        """Refresh tree with current capsules."""
        self.clear()
        
        # Group capsules by type
        capsules_by_type = {}
        for capsule in self.capsule_manager.capsules:
            capsule_type = "MOCAP" if capsule.type.lower().endswith("_motion") else capsule.type.capitalize()
            if capsule_type not in capsules_by_type:
                capsules_by_type[capsule_type] = []
            capsules_by_type[capsule_type].append(capsule)
            
        # Create tree items
        for capsule_type, capsules in sorted(capsules_by_type.items()):
            type_item = QTreeWidgetItem(self, [capsule_type, "", "", ""])
            type_item.setExpanded(True)
            
            for capsule in sorted(capsules, key=lambda c: c.name):
                item = QTreeWidgetItem(type_item, [
                    capsule.name,
                    capsule.type,
                    str(capsule.usage_count),
                    str(capsule.orbit_score)
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, capsule.uuid)
                
                # Set icon based on type
                color = self.CAPSULE_COLORS.get(capsule.type.lower(), self.COLORS['gray'])
                icon = self.create_color_icon(color)
                item.setIcon(0, icon)
                
                # Bold for pinned capsules
                if capsule.pinned_to_core:
                    font = item.font(0)
                    font.setBold(True)
                    item.setFont(0, font)
                    
    def create_color_icon(self, color: QColor, size: QSize = QSize(16, 16)) -> QIcon:
        """Create a colored circle icon."""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, size.width() - 4, size.height() - 4)
        painter.end()
        
        return QIcon(pixmap)
        
    def on_selection_changed(self):
        """Handle selection change."""
        items = self.selectedItems()
        if items:
            item = items[0]
            uuid_str = item.data(0, Qt.ItemDataRole.UserRole)
            if uuid_str:
                capsule = self.capsule_manager.get_capsule_by_uuid(uuid_str)
                if capsule:
                    self.capsule_selected.emit(capsule)
                    
    def on_item_double_clicked(self, item, column):
        """Handle double click."""
        uuid_str = item.data(0, Qt.ItemDataRole.UserRole)
        if uuid_str:
            capsule = self.capsule_manager.get_capsule_by_uuid(uuid_str)
            if capsule:
                self.capsule_double_clicked.emit(capsule)
                
    def show_context_menu(self, position):
        """Show context menu for capsules."""
        item = self.itemAt(position)
        if not item:
            return
            
        uuid_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not uuid_str:
            return
            
        capsule = self.capsule_manager.get_capsule_by_uuid(uuid_str)
        if not capsule:
            return
            
        menu = QMenu()
        
        pin_action = menu.addAction("Pin to Core" if not capsule.pinned_to_core else "Unpin from Core")
        menu.addSeparator()
        delete_action = menu.addAction("Delete Capsule")
        menu.addSeparator()
        merge_action = menu.addAction("Merge with...")
        export_action = menu.addAction("Export as Image...")
        
        action = menu.exec_(self.viewport().mapToGlobal(position))
        
        if action == pin_action:
            if capsule.pinned_to_core:
                capsule.pinned_to_core = False
                capsule.set_orbit_distance(1.0)
            else:
                capsule.pin_to_core()
            self.capsule_manager.capsule_updated.emit(capsule)
            self.refresh()
        elif action == delete_action:
            if capsule.name != "CorePersonality":
                reply = QMessageBox.question(
                    self, "Delete Capsule",
                    f"Are you sure you want to delete capsule '{capsule.name}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.capsule_manager.remove_capsule(capsule)
                    
    def startDrag(self, supportedActions):
        """Override startDrag to provide capsule data."""
        item = self.currentItem()
        if not item:
            return
            
        uuid_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not uuid_str:
            return
            
        capsule = self.capsule_manager.get_capsule_by_uuid(uuid_str)
        if not capsule:
            return
            
        # Create mime data with capsule information
        mime_data = QMimeData()
        mime_data.setData("application/x-capsule", capsule.uuid.encode())
        mime_data.setText(f"capsule:{capsule.uuid}")
        
        # Create drag object
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # Set drag pixmap (capsule icon)
        color = self.CAPSULE_COLORS.get(capsule.type.lower(), self.COLORS['gray'])
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(4, 4, 24, 24)
        painter.setPen(QPen(Qt.GlobalColor.black))
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, capsule.name[:2].upper())
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(16, 16))
        
        drag.exec(Qt.DropAction.CopyAction)


class OrbitalViewWidget(QWidget):
    """Orbital visualization widget for capsules with functional lanes."""

    capsule_selected = pyqtSignal(object)  # Capsule object

    # Functional lanes for different capsule types
    LANES = {
        'core': {'name': 'Core', 'min_radius': 0.0, 'max_radius': 0.3, 'color': QColor(255, 200, 0)},
        'character': {'name': 'Characters', 'min_radius': 0.4, 'max_radius': 0.7, 'color': QColor(70, 130, 180)},
        'style': {'name': 'Styles', 'min_radius': 0.8, 'max_radius': 1.1, 'color': QColor(186, 85, 211)},
        'skill': {'name': 'Skills', 'min_radius': 1.2, 'max_radius': 1.5, 'color': QColor(255, 140, 0)},
        'memory': {'name': 'Memories', 'min_radius': 1.6, 'max_radius': 1.9, 'color': QColor(60, 179, 113)},
        'workflow': {'name': 'Workflows', 'min_radius': 2.0, 'max_radius': 2.3, 'color': QColor(138, 43, 226)},
        'experimental': {'name': 'Experimental', 'min_radius': 2.4, 'max_radius': 2.7, 'color': QColor(150, 255, 150)}
    }

    @classmethod
    def set_capsule_manager_class(cls, capsule_manager_class):
        """Set the CapsuleManager class to use (called from main.py)."""
        global CapsuleManager
        CapsuleManager = capsule_manager_class

    def __init__(self, capsule_manager=None, parent=None):
        super().__init__(parent)
        self.capsule_manager = capsule_manager
        self.capsules = []
        self.selected_capsule = None
        self.hovered_capsule = None
        self.zoom_level = 1.0
        self._angle = 0.0
        self._speed = 30.0  # degrees per second
        self._radius = 100
        self.active_connections = []
        self.network_activity = 0.0
        self.bg_color = QColor(10, 10, 30)  # Dark space background
        self.text_color = QColor(200, 200, 200)  # Light gray text
        self.highlight_color = QColor(100, 200, 255)  # Blue highlight
        self.show_info = False  # Info panel visibility
        self.show_lanes = True  # Show functional lane rings

        # Smooth drift animation
        self.drift_lambda = 0.05  # Smoothing factor for drift (0-1, higher = faster)
        self.target_positions = {}  # capsule_uuid -> target_position

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(50)  # 20 FPS

    def _step(self):
        """Animation step for smooth drift and rotation."""
        # Update rotation
        self._angle += self._speed * (50.0 / 1000.0)  # Convert to seconds
        if self._angle >= 360.0:
            self._angle -= 360.0

        # Update smooth drift positions
        for capsule in self.capsules:
            capsule_uuid = getattr(capsule, 'uuid', str(id(capsule)))
            target_radius = self.get_lane_position(capsule)

            if capsule_uuid not in self.target_positions:
                # Initialize target position
                self.target_positions[capsule_uuid] = target_radius
                capsule.current_radius = target_radius
            else:
                # Smooth interpolation towards target
                current_radius = getattr(capsule, 'current_radius', target_radius)
                self.target_positions[capsule_uuid] = target_radius
                capsule.current_radius = (1.0 - self.drift_lambda) * current_radius + self.drift_lambda * target_radius

        self.update()

    def get_capsule_lane(self, capsule):
        """Get the functional lane for a capsule based on its type."""
        capsule_type = capsule.type.lower()

        # Map capsule types to lanes
        type_to_lane = {
            'personality': 'core',
            'core': 'core',
            'character': 'character',
            'animation': 'character',  # Animation capsules often relate to characters
            'character_animation': 'character',  # Merged capsules
            'style': 'style',
            'image': 'style',
            '2dimage': 'style',
            'skill': 'skill',
            'memory': 'memory',
            'fact': 'memory',
            'workflow': 'workflow',
            'topic': 'workflow',
            'experimental': 'experimental'
        }

        return type_to_lane.get(capsule_type, 'experimental')

    def get_lane_position(self, capsule):
        """Calculate the position within the capsule's lane based on gravity."""
        lane_name = self.get_capsule_lane(capsule)
        lane = self.LANES[lane_name]

        # Calculate gravity (0-1 scale) from orbit_score
        orbit_score = getattr(capsule, 'orbit_score', 0)
        # Use sigmoid function for smooth gravity calculation
        import math
        alpha = 0.12  # Controls the steepness of the sigmoid
        gravity = 1.0 / (1.0 + math.exp(-alpha * float(orbit_score)))

        # Position within lane: higher gravity = inner position
        lane_range = lane['max_radius'] - lane['min_radius']
        position_in_lane = lane['min_radius'] + (1.0 - gravity) * lane_range

        return position_in_lane

    def refresh_capsules(self):
        """Refresh the capsule list from the manager."""
        self.capsules = self.capsule_manager.capsules if self.capsule_manager else []
        self.update()

    def update_network_overlay(self, active_connections, activity_level: float):
        """Update GNN-derived overlays (active edges + heartbeat)."""
        self.active_connections = active_connections or []
        self.network_activity = max(0.0, min(1.0, float(activity_level)))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cx = self.width() // 2
            cy = self.height() // 2
            mx = event.pos().x()
            my = event.pos().y()
            for cap in self.capsules:
                # Use smooth drift positioning for hit detection
                orbit_radius = getattr(cap, 'current_radius', self.get_lane_position(cap)) * 50 * self.zoom_level
                x = cx + orbit_radius * math.cos(cap.angle)
                y = cy + orbit_radius * math.sin(cap.angle)
                dist = math.hypot(mx - x, my - y)
                if dist < 20:
                    self.selected_capsule = cap
                    self.capsule_selected.emit(cap)
                    self.update()
                    break

    def mouseMoveEvent(self, event):
        cx = self.width() // 2
        cy = self.height() // 2
        mx = event.pos().x()
        my = event.pos().y()
        self.hovered_capsule = None
        for cap in self.capsules:
            # Use smooth drift positioning for hover detection
            orbit_radius = getattr(cap, 'current_radius', self.get_lane_position(cap)) * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)
            dist = math.hypot(mx - x, my - y)
            if dist < 20:
                self.hovered_capsule = cap
                break

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.2 ** (delta / 120)
        self.zoom_level = max(0.2, min(3.0, self.zoom_level * factor))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        cx = self.width() // 2
        cy = self.height() // 2
        
        # Draw dark space background
        p.fillRect(self.rect(), self.bg_color)
        for cap in self.capsules:
            # Update angle for animation
            angle = (int(cap.uuid.replace('-', ''), 16) % 360 + self._angle) * (3.14159 / 180)
            cap.angle = angle  # Store for consistency

        # Draw functional lane rings
        if self.show_lanes:
            self.draw_lane_rings(p, cx, cy)

        # Draw capsule orbits (legacy support)
        orbit_radii = sorted(set(int(cap.orbit_distance * 100) / 100.0 for cap in self.capsules))
        p.setPen(QPen(QColor(60, 80, 120), 1))
        p.setOpacity(0.35)
        for radius in orbit_radii:
            orbit_radius = radius * 50 * self.zoom_level
            p.drawEllipse(int(cx - orbit_radius), int(cy - orbit_radius), int(orbit_radius * 2), int(orbit_radius * 2))
        p.setOpacity(1.0)
        for cap in self.capsules:
            # Use smooth drift positioning
            orbit_radius = getattr(cap, 'current_radius', self.get_lane_position(cap)) * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)
            kind = cap.type.lower()
            if kind == "character":
                color = QColor(70, 130, 180)  # Steel blue
            elif kind == "pose":
                color = QColor(60, 179, 113)  # Medium sea green
            elif kind == "skill":
                color = QColor(255, 140, 0)  # Dark orange
            elif kind == "style":
                color = QColor(186, 85, 211)  # Medium orchid
            elif kind == "personality":
                color = QColor(30, 144, 255)  # Dodger blue
            else:
                color = QColor(150, 255, 150)  # Light green
            # Use usage_count as a proxy for certainty
            certainty = min(1.0, cap.usage_count / 20.0) if hasattr(cap, 'usage_count') else 0.5
            size = max(6, int(14 * certainty))
            glow_radius = size + 6
            for i in range(3, 0, -1):
                glow_color = QColor(color)
                glow_color.setAlpha(int(40 / (i + 1)))
                p.setBrush(QBrush(glow_color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(int(x - glow_radius / 2 - i * 2), int(y - glow_radius / 2 - i * 2), int(glow_radius + i * 4), int(glow_radius + i * 4))
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(int(x - size / 2), int(y - size / 2), size, size)
            if self.zoom_level > 0.8 or cap == self.selected_capsule or cap == self.hovered_capsule:
                p.setPen(QPen(self.text_color))
                f = p.font()
                f.setPointSize(8)
                p.setFont(f)
                label = cap.name[:3] if len(cap.name) > 3 else cap.name
                p.drawText(int(x + size / 2 + 5), int(y + 3), label)
            if cap == self.selected_capsule:
                p.setPen(QPen(self.highlight_color, 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 5), int(y - size / 2 - 5), size + 10, size + 10)
                p.setPen(QPen(self.highlight_color, 1))
                p.setOpacity(0.5)
                p.drawLine(int(cx), int(cy), int(x), int(y))
                p.setOpacity(1.0)
            elif cap == self.hovered_capsule:
                p.setPen(QPen(QColor(200, 200, 150), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 3), int(y - size / 2 - 3), size + 6, size + 6)

        # Draw active knowledge-network pathways
        if self.active_connections:
            p.setOpacity(0.55)
            pen = QPen(QColor(120, 220, 255))
            pen.setWidth(1)
            p.setPen(pen)
            for cap1, cap2 in self.active_connections:
                if cap1 not in self.capsules or cap2 not in self.capsules:
                    continue
                r1 = cap1.orbit_distance * 50 * self.zoom_level
                r2 = cap2.orbit_distance * 50 * self.zoom_level
                ax = cx + r1 * math.cos(cap1.angle)
                ay = cy + r1 * math.sin(cap1.angle)
                bx = cx + r2 * math.cos(cap2.angle)
                by = cy + r2 * math.sin(cap2.angle)
                p.drawLine(int(ax), int(ay), int(bx), int(by))
            p.setOpacity(1.0)
        self.draw_central_nucleus(p, cx, cy)
        if self.show_info:
            self.draw_info_panel(p)
        p.end()

    def draw_lane_rings(self, p: QPainter, cx: int, cy: int):
        """Draw functional lane rings with labels."""
        p.setOpacity(0.6)

        for lane_name, lane_info in self.LANES.items():
            # Draw outer ring
            outer_radius = lane_info['max_radius'] * 50 * self.zoom_level
            p.setPen(QPen(lane_info['color'], 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx - outer_radius), int(cy - outer_radius),
                         int(outer_radius * 2), int(outer_radius * 2))

            # Draw inner ring (if not core)
            if lane_info['min_radius'] > 0:
                inner_radius = lane_info['min_radius'] * 50 * self.zoom_level
                p.setPen(QPen(lane_info['color'], 1, Qt.PenStyle.DashLine))
                p.drawEllipse(int(cx - inner_radius), int(cy - inner_radius),
                             int(inner_radius * 2), int(inner_radius * 2))

            # Draw lane label
            if self.zoom_level > 0.5:
                label_radius = (lane_info['min_radius'] + lane_info['max_radius']) / 2 * 50 * self.zoom_level
                label_angle = 0  # Top of the circle
                label_x = cx + label_radius * math.cos(label_angle)
                label_y = cy + label_radius * math.sin(label_angle) - 15

                p.setPen(QPen(lane_info['color']))
                font = p.font()
                font.setPointSize(8)
                font.setBold(True)
                p.setFont(font)
                p.drawText(int(label_x - 30), int(label_y), 60, 20,
                          Qt.AlignmentFlag.AlignCenter, lane_info['name'])

        p.setOpacity(1.0)

    def draw_central_nucleus(self, p: QPainter, cx: int, cy: int):
        # Heartbeat intensity modulated by network_activity
        for i in range(30, 0, -5):
            glow_color = QColor(255, 200, 0)
            base = 60 + int(40 * self.network_activity)
            glow_color.setAlpha(int(base * (30 - i) / 30))
            p.setBrush(QBrush(glow_color))
            p.setPen(QPen(Qt.PenStyle.NoPen))
            p.drawEllipse(int(cx - i), int(cy - i), i * 2, i * 2)
        gradient_color = QColor(255, 220, 100)
        p.setBrush(QBrush(gradient_color))
        p.setPen(QPen(QColor(255, 240, 150), 2))
        p.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)
        p.setBrush(QBrush(QColor(255, 255, 200)))
        p.setPen(QPen(Qt.PenStyle.NoPen))
        p.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        p.setPen(QPen(QColor(255, 255, 255)))
        f = p.font()
        f.setPointSize(11)
        f.setBold(True)
        p.setFont(f)
        p.drawText(int(cx + 20), int(cy - 5), "Identity")
        p.drawText(int(cx + 20), int(cy + 10), "Nucleus")

    def draw_info_panel(self, p: QPainter):
        panel_w = 80
        panel_h = 60
        panel_bg = QColor(0, 0, 0)
        panel_bg.setAlpha(200)
        p.fillRect(10, 10, panel_w, panel_h, panel_bg)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawRect(10, 10, panel_w, panel_h)
        p.setPen(QPen(QColor(255, 255, 255)))
        f = p.font()
        f.setPointSize(7)
        f.setBold(True)
        p.setFont(f)
        p.drawText(15, 22, "ROCA")
        f.setBold(False)
        f.setPointSize(6)
        p.setFont(f)
        y = 34
        p.drawText(15, y, f"◐ {len(self.capsules)}")
        if self.selected_capsule:
            p.drawText(15, y + 10, "✓ Sel.")
        else:
            p.drawText(15, y + 10, f"Zm:{self.zoom_level:.1f}")


class UserLearningSystem(QObject):
    """AI system for learning user patterns across multiple modalities."""
    
    learning_updated = pyqtSignal(str)  # Signal when learning data is updated
    
    def __init__(self, capsule_manager=None, graph_manager=None):
        super().__init__()
        self.capsule_manager = capsule_manager
        self.graph_manager = graph_manager
        
        # Learning data storage
        self.artistic_patterns = {
            'stroke_patterns': [],
            'color_preferences': Counter(),
            'shape_preferences': Counter(),
            'style_evolution': []
        }
        
        self.speech_patterns = {
            'vocabulary': Counter(),
            'sentence_structure': [],
            'tone_patterns': [],
            'speech_rate': [],
            'common_phrases': Counter()
        }
        
        self.writing_patterns = {
            'vocabulary': Counter(),
            'sentence_length': [],
            'punctuation_usage': Counter(),
            'topic_interests': Counter(),
            'writing_style': []
        }
        
        self.learning_capsules = {}  # Store learning capsules by type
        
    def analyze_drawing(self, drawing_data: Dict[str, Any]):
        """Analyze a drawing to learn artistic patterns."""
        if not drawing_data:
            return
            
        # Extract features from drawing
        strokes = drawing_data.get('strokes', [])
        colors = drawing_data.get('colors', [])
        shapes = self._identify_shapes(strokes)
        
        # Update patterns
        self.artistic_patterns['stroke_patterns'].append({
            'length': len(strokes),
            'avg_stroke_length': sum(len(s) for s in strokes) / len(strokes) if strokes else 0,
            'timestamp': time.time()
        })
        
        for color in colors:
            self.artistic_patterns['color_preferences'][color] += 1
            
        for shape in shapes:
            self.artistic_patterns['shape_preferences'][shape] += 1
            
        # Create or update artistic style capsule
        self._update_artistic_style_capsule()
        self.learning_updated.emit("artistic_style")
        
    def analyze_speech(self, speech_data: Dict[str, Any]):
        """Analyze speech patterns from microphone input."""
        if not speech_data:
            return
            
        text = speech_data.get('transcription', '')
        if not text:
            return
            
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        
        # Update speech patterns
        for word in words:
            self.speech_patterns['vocabulary'][word] += 1
            
        for sentence in sentences:
            if sentence.strip():
                self.speech_patterns['sentence_structure'].append({
                    'length': len(sentence.split()),
                    'structure': self._analyze_sentence_structure(sentence)
                })
                
        # Update speech patterns capsule
        self._update_speech_pattern_capsule()
        self.learning_updated.emit("speech_pattern")
        
    def analyze_text(self, text: str, source: str = "chat"):
        """Analyze written text patterns."""
        if not text:
            return
            
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        
        # Update writing patterns
        for word in words:
            self.writing_patterns['vocabulary'][word] += 1
            
        for sentence in sentences:
            if sentence.strip():
                self.writing_patterns['sentence_length'].append(len(sentence.split()))
                
        # Analyze punctuation
        punctuation = re.findall(r'[.!?,\-:;()"\']', text)
        for p in punctuation:
            self.writing_patterns['punctuation_usage'][p] += 1
            
        # Identify topics/interests
        topics = self._identify_topics(text)
        for topic in topics:
            self.writing_patterns['topic_interests'][topic] += 1
            
        # Update writing style capsule
        self._update_writing_style_capsule()
        self.learning_updated.emit("writing_style")
        
    def _identify_shapes(self, strokes):
        """Identify geometric shapes from stroke patterns."""
        shapes = []
        for stroke in strokes:
            if len(stroke) < 3:
                continue
                
            # Simple shape detection based on stroke characteristics
            start_point = stroke[0]
            end_point = stroke[-1]
            
            # Check for circles (closed loops)
            if self._is_closed_loop(stroke):
                shapes.append('circle')
            # Check for straight lines
            elif self._is_straight_line(stroke):
                shapes.append('line')
            # Check for curves
            elif self._is_curve(stroke):
                shapes.append('curve')
            else:
                shapes.append('freeform')
                
        return shapes
        
    def _is_closed_loop(self, stroke):
        """Check if stroke forms a closed loop."""
        if len(stroke) < 10:
            return False
        start = stroke[0]
        end = stroke[-1]
        distance = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        return distance < 20  # Close enough to be considered closed
        
    def _is_straight_line(self, stroke):
        """Check if stroke is approximately straight."""
        if len(stroke) < 3:
            return True
            
        # Calculate variance from straight line
        x_coords = [p[0] for p in stroke]
        y_coords = [p[1] for p in stroke]
        
        # Simple linearity check
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        if x_range > y_range:
            # Horizontal-ish line
            avg_y = sum(y_coords) / len(y_coords)
            variance = sum((y - avg_y)**2 for y in y_coords) / len(y_coords)
            return variance < 100
        else:
            # Vertical-ish line
            avg_x = sum(x_coords) / len(x_coords)
            variance = sum((x - avg_x)**2 for x in x_coords) / len(x_coords)
            return variance < 100
            
    def _is_curve(self, stroke):
        """Check if stroke is curved."""
        return not self._is_straight_line(stroke) and not self._is_closed_loop(stroke)
        
    def _analyze_sentence_structure(self, sentence):
        """Analyze sentence structure patterns."""
        words = sentence.split()
        if not words:
            return 'empty'
            
        # Simple structure analysis
        if words[0].lower() in ['i', 'we', 'you', 'they', 'he', 'she', 'it']:
            return 'personal'
        elif words[0].lower() in ['the', 'a', 'an']:
            return 'descriptive'
        elif any(q in sentence for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        else:
            return 'declarative'
            
    def _identify_topics(self, text):
        """Identify topics/interests from text."""
        topics = []
        text_lower = text.lower()
        
        # Define topic keywords
        topic_keywords = {
            'art': ['draw', 'paint', 'color', 'canvas', 'brush', 'art', 'design'],
            'animation': ['animate', 'frame', 'motion', 'timeline', 'keyframe', 'character'],
            '3d': ['model', 'mesh', 'vertex', 'polygon', 'render', 'texture', 'material'],
            'technology': ['computer', 'software', 'program', 'code', 'algorithm', 'ai'],
            'creative': ['create', 'imagine', 'design', 'build', 'make', 'craft']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
                
        return topics
        
    def _update_artistic_style_capsule(self):
        """Create or update the user's artistic style capsule."""
        if not self.capsule_manager or not self.graph_manager:
            return
            
        capsule_name = "User_Artistic_Style"
        style_data = {
            'stroke_patterns': self.artistic_patterns['stroke_patterns'][-10:],  # Last 10 drawings
            'top_colors': dict(self.artistic_patterns['color_preferences'].most_common(5)),
            'preferred_shapes': dict(self.artistic_patterns['shape_preferences'].most_common(3)),
            'learning_progression': len(self.artistic_patterns['stroke_patterns'])
        }
        
        # Create or update capsule
        if capsule_name in self.learning_capsules:
            # Update existing
            capsule = self.capsule_manager.get_capsule_by_uuid(self.learning_capsules[capsule_name])
            if capsule:
                capsule.metadata['style_data'] = style_data
                capsule.metadata['last_updated'] = time.time()
        else:
            # Create new
            capsule = Capsule(
                name=capsule_name,
                capsule_type="artistic_style",
                metadata={
                    'style_data': style_data,
                    'created': time.time(),
                    'last_updated': time.time(),
                    'type': 'artistic_style'
                }
            )
            self.capsule_manager.add_capsule(capsule)
            
            # Add to ROCA graph
            command = Command(CommandType.ADD_CAPSULE, {'capsule': capsule})
            self.graph_manager.execute(command)
            
            self.learning_capsules[capsule_name] = capsule.id
            
    def _update_speech_pattern_capsule(self):
        """Create or update the user's speech pattern capsule."""
        if not self.capsule_manager or not self.graph_manager:
            return
            
        capsule_name = "User_Speech_Patterns"
        speech_data = {
            'vocabulary_size': len(self.speech_patterns['vocabulary']),
            'top_words': dict(self.speech_patterns['vocabulary'].most_common(10)),
            'avg_sentence_length': sum(s['length'] for s in self.speech_patterns['sentence_structure'][-20:]) / max(1, len(self.speech_patterns['sentence_structure'][-20:])),
            'communication_style': self._analyze_communication_style()
        }
        
        # Create or update capsule
        if capsule_name in self.learning_capsules:
            capsule = self.capsule_manager.get_capsule_by_uuid(self.learning_capsules[capsule_name])
            if capsule:
                capsule.metadata['speech_data'] = speech_data
                capsule.metadata['last_updated'] = time.time()
        else:
            capsule = Capsule(
                name=capsule_name,
                capsule_type="speech_pattern",
                metadata={
                    'speech_data': speech_data,
                    'created': time.time(),
                    'last_updated': time.time(),
                    'type': 'speech_pattern'
                }
            )
            self.capsule_manager.add_capsule(capsule)
            
            command = Command(CommandType.ADD_CAPSULE, {'capsule': capsule})
            self.graph_manager.execute(command)
            
            self.learning_capsules[capsule_name] = capsule.id
            
    def _update_writing_style_capsule(self):
        """Create or update the user's writing style capsule."""
        if not self.capsule_manager or not self.graph_manager:
            return
            
        capsule_name = "User_Writing_Style"
        writing_data = {
            'vocabulary_size': len(self.writing_patterns['vocabulary']),
            'top_words': dict(self.writing_patterns['vocabulary'].most_common(10)),
            'avg_sentence_length': sum(self.writing_patterns['sentence_length'][-20:]) / max(1, len(self.writing_patterns['sentence_length'][-20:])),
            'punctuation_style': dict(self.writing_patterns['punctuation_usage'].most_common(5)),
            'interests': dict(self.writing_patterns['topic_interests'].most_common(3))
        }
        
        # Create or update capsule
        if capsule_name in self.learning_capsules:
            capsule = self.capsule_manager.get_capsule_by_uuid(self.learning_capsules[capsule_name])
            if capsule:
                capsule.metadata['writing_data'] = writing_data
                capsule.metadata['last_updated'] = time.time()
        else:
            capsule = Capsule(
                name=capsule_name,
                capsule_type="writing_style",
                metadata={
                    'writing_data': writing_data,
                    'created': time.time(),
                    'last_updated': time.time(),
                    'type': 'writing_style'
                }
            )
            self.capsule_manager.add_capsule(capsule)
            
            command = Command(CommandType.ADD_CAPSULE, {'capsule': capsule})
            self.graph_manager.execute(command)
            
            self.learning_capsules[capsule_name] = capsule.id
            
    def _analyze_communication_style(self):
        """Analyze overall communication style."""
        if not self.speech_patterns['sentence_structure']:
            return 'unknown'
            
        structures = [s['structure'] for s in self.speech_patterns['sentence_structure']]
        most_common = Counter(structures).most_common(1)[0][0]
        
        return most_common
        
    def get_learning_summary(self):
        """Get a summary of learned patterns."""
        return {
            'artistic_patterns': {
                'drawings_analyzed': len(self.artistic_patterns['stroke_patterns']),
                'preferred_shapes': list(self.artistic_patterns['shape_preferences'].keys())[:3],
                'color_palette': list(self.artistic_patterns['color_preferences'].keys())[:3]
            },
            'speech_patterns': {
                'vocabulary_size': len(self.speech_patterns['vocabulary']),
                'communication_style': self._analyze_communication_style()
            },
            'writing_patterns': {
                'vocabulary_size': len(self.writing_patterns['vocabulary']),
                'interests': list(self.writing_patterns['topic_interests'].keys())[:3]
            }
        }


class ChatWidget(QWidget):
    """Interactive chat widget for ROCA capsule network communication."""
    
    message_sent = pyqtSignal(str)  # User message
    capsule_created = pyqtSignal(object)  # New capsule from conversation
    
    def __init__(self, voice_system=None, parent=None):
        super().__init__(parent)
        self.capsule_manager = None
        self.graph_manager = None
        self.orbital_widget = None
        self.message_handler = None
        self.threadpool = None
        self.learning_system = None
        self.voice_system = voice_system
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the chat interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMaximumHeight(200)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a2e;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a3e;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
                padding: 4px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a6e;
                color: #ffffff;
                border: 1px solid #666688;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a5a7e;
            }
            QPushButton:pressed {
                background-color: #3a3a5e;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        
        # Status label
        self.status_label = QLabel("ROCA Chat - Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #8888aa;
                font-size: 9px;
                padding: 2px;
            }
        """)
        
        layout.addWidget(QLabel("ROCA Network Chat"))
        layout.addWidget(self.chat_display)
        layout.addLayout(input_layout)
        layout.addWidget(self.status_label)
        
        # Add voice system widget if available
        if self.voice_system:
            try:
                from voice_system import VoiceWidget
                self.voice_widget = VoiceWidget(self.voice_system)
                layout.addWidget(self.voice_widget)
            except ImportError:
                pass
        
        # Set size policy
        self.setMaximumHeight(400)  # Increased to accommodate voice widget
        
    def set_managers(self, capsule_manager, graph_manager, orbital_widget=None):
        """Set the capsule and graph managers."""
        self.capsule_manager = capsule_manager
        self.graph_manager = graph_manager
        self.orbital_widget = orbital_widget
        
        # Initialize learning system
        self.learning_system = UserLearningSystem(capsule_manager, graph_manager)
        self.learning_system.learning_updated.connect(self.on_learning_updated)
        
        # Import threadpool for async operations
        try:
            from PyQt6.QtCore import QThreadPool
            self.threadpool = QThreadPool.globalInstance()
        except ImportError:
            self.threadpool = None
            
    def set_message_handler(self, handler):
        """Set a custom message handler function."""
        self.message_handler = handler
        
    def send_message(self):
        """Send a message to the ROCA network."""
        message = self.message_input.text().strip()
        if not message:
            return
            
        self.message_input.clear()
        self.append_message("You", message)
        
        # Process the message
        self.process_message(message)
        
    def process_message(self, message):
        """Process the user message."""
        # Analyze text with learning system
        if self.learning_system:
            self.learning_system.analyze_text(message, source="chat")
        
        if self.message_handler:
            # Use custom handler
            try:
                response = self.message_handler(message)
                self.append_message("ROCA", response)
            except Exception as e:
                self.append_message("ROCA", f"Error: {str(e)}")
        else:
            # Default processing - create capsule from message
            self.create_capsule_from_message(message)
            
    def create_capsule_from_message(self, message):
        """Create a capsule from the user's message."""
        if not self.capsule_manager or not self.graph_manager:
            self.append_message("ROCA", "Capsule system not available")
            return
            
        # Determine capsule type based on message content
        capsule_type = self.infer_capsule_type(message)
        
        # Create capsule
        capsule = Capsule(
            name=f"Chat: {message[:30]}...",
            capsule_type=capsule_type,
            metadata={
                'source': 'chat',
                'message': message,
                'timestamp': time.time(),
                'type': capsule_type
            }
        )
        
        # Add to capsule manager
        self.capsule_manager.add_capsule(capsule)
        
        # Add to ROCA graph
        command = Command(CommandType.ADD_CAPSULE, {'capsule': capsule})
        result = self.graph_manager.execute(command)
        
        if result.ok:
            self.append_message("ROCA", f"Created {capsule_type} capsule: {capsule.name}")
            self.capsule_created.emit(capsule)
            
            # Add to orbital view if available
            if self.orbital_widget and hasattr(self.orbital_widget, 'capsules'):
                if capsule not in self.orbital_widget.capsules:
                    self.orbital_widget.capsules.append(capsule)
                    self.orbital_widget.update()
        else:
            self.append_message("ROCA", f"Failed to create capsule: {result.message}")
            
    def infer_capsule_type(self, message):
        """Infer capsule type from message content."""
        message_lower = message.lower()
        
        # Check for keywords
        if any(word in message_lower for word in ['character', 'person', 'figure', 'actor']):
            return 'character'
        elif any(word in message_lower for word in ['move', 'motion', 'animation', 'pose']):
            return 'motion'
        elif any(word in message_lower for word in ['style', 'look', 'appearance', 'aesthetic']):
            return 'style'
        elif any(word in message_lower for word in ['skill', 'technique', 'method', 'how']):
            return 'skill'
        elif any(word in message_lower for word in ['model', 'object', '3d', 'mesh']):
            return '3dmodel'
        elif any(word in message_lower for word in ['image', 'picture', 'visual', '2d']):
            return '2dimage'
        else:
            return 'concept'  # Default type
            
    def append_message(self, sender, message):
        """Append a message to the chat display."""
        import html
        safe_message = html.escape(message)
        timestamp = time.strftime("%H:%M:%S")
        
        if sender == "You":
            color = "#4a9eff"
        else:
            color = "#ff6b6b"
            
        self.chat_display.append(f'<span style="color: #888;">[{timestamp}]</span> '
                               f'<span style="color: {color}; font-weight: bold;">{sender}:</span> '
                               f'<span style="color: #fff;">{safe_message}</span>')
        
        # Auto scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def update_status(self, status):
        """Update the status label."""
        self.status_label.setText(f"ROCA Chat - {status}")
        
    def on_learning_updated(self, learning_type):
        """Handle learning system updates."""
        if learning_type == "writing_style":
            summary = self.learning_system.get_learning_summary()
            writing_info = summary.get('writing_patterns', {})
            vocab_size = writing_info.get('vocabulary_size', 0)
            interests = writing_info.get('interests', [])
            
            status_msg = f"Learned writing patterns - Vocab: {vocab_size}, Interests: {', '.join(interests[:2])}"
            self.update_status(status_msg)
            
            # Show learning feedback
            self.append_message("ROCA", f"📚 Learned from your message! Writing style updated. Interests: {', '.join(interests)}")
        elif learning_type == "speech_pattern":
            summary = self.learning_system.get_learning_summary()
            speech_info = summary.get('speech_patterns', {})
            vocab_size = speech_info.get('vocabulary_size', 0)
            style = speech_info.get('communication_style', 'unknown')
            
            self.append_message("ROCA", f"🎤 Speech patterns analyzed! Style: {style}")
        elif learning_type == "artistic_style":
            summary = self.learning_system.get_learning_summary()
            art_info = summary.get('artistic_patterns', {})
            drawings = art_info.get('drawings_analyzed', 0)
            shapes = art_info.get('preferred_shapes', [])
            
            self.append_message("ROCA", f"🎨 Artistic patterns learned! {drawings} drawings analyzed, preferred shapes: {', '.join(shapes)}")


class InspectorWidget(QWidget):
    """Widget for reviewing and accepting cluster proposals."""
    
    proposal_accepted = pyqtSignal(dict)  # Proposal dictionary
    proposal_rejected = pyqtSignal(dict)  # Proposal dictionary
    
    def __init__(self, capsule_manager: CapsuleManager, parent=None):
        super().__init__(parent)
        self.capsule_manager = capsule_manager
        self.proposals: List[Dict] = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup inspector UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Cluster Inspector")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_proposals)
        
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_proposals)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(refresh_button)
        header.addWidget(clear_button)
        
        # Instructions
        instructions = QLabel(
            "Review AI-generated proposals for linking unassigned images to characters/poses."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-style: italic;")
        
        # Proposal list
        self.proposal_list = QListWidget()
        self.proposal_list.setAlternatingRowColors(True)
        self.proposal_list.itemDoubleClicked.connect(self.on_proposal_double_clicked)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.accept_button = QPushButton("Accept Selected")
        self.accept_button.clicked.connect(self.accept_selected)
        self.accept_button.setEnabled(False)
        
        self.reject_button = QPushButton("Reject Selected")
        self.reject_button.clicked.connect(self.reject_selected)
        self.reject_button.setEnabled(False)
        
        self.accept_all_button = QPushButton("Accept All")
        self.accept_all_button.clicked.connect(self.accept_all)
        
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.reject_button)
        button_layout.addStretch()
        button_layout.addWidget(self.accept_all_button)
        
        # Connect signals
        self.proposal_list.itemSelectionChanged.connect(self.update_button_states)
        
        # Assemble layout
        layout.addLayout(header)
        layout.addWidget(instructions)
        layout.addWidget(self.proposal_list, 1)
        layout.addLayout(button_layout)
        
    def set_proposals(self, proposals: List[Dict]):
        """Set proposals to display."""
        self.proposals = proposals
        self.refresh_list()
        
    def refresh_list(self):
        """Refresh proposal list display."""
        self.proposal_list.clear()
        
        for i, proposal in enumerate(self.proposals):
            cap = proposal.get('cap')
            if not cap:
                continue
                
            # Create list item
            item = QListWidgetItem()
            
            # Create custom widget for the item
            widget = QWidget()
            widget_layout = QHBoxLayout(widget)
            widget_layout.setContentsMargins(5, 5, 5, 5)
            
            # Thumbnail
            thumb_label = QLabel()
            thumb_label.setFixedSize(64, 64)
            
            thumb_path = None
            if isinstance(cap.metadata, dict):
                thumbs = cap.metadata.get('thumbnails')
                if thumbs and len(thumbs) > 0:
                    thumb_path = thumbs[0]
                    
            if thumb_path and os.path.exists(thumb_path):
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
                    thumb_label.setPixmap(pixmap)
            else:
                thumb_label.setText("No\nImage")
                thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                thumb_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
                
            # Text info
            text_widget = QWidget()
            text_layout = QVBoxLayout(text_widget)
            text_layout.setContentsMargins(0, 0, 0, 0)
            
            # Capsule name
            name_label = QLabel(cap.name)
            name_font = name_label.font()
            name_font.setBold(True)
            name_label.setFont(name_font)
            
            # Proposal text
            proposal_text = self.get_proposal_text(proposal)
            proposal_label = QLabel(proposal_text)
            proposal_label.setWordWrap(True)
            
            # Confidence
            confidence = max(proposal.get('character_score', 0), 
                           proposal.get('pose_score', 0))
            confidence_label = QLabel(f"Confidence: {confidence:.0%}")
            
            text_layout.addWidget(name_label)
            text_layout.addWidget(proposal_label)
            text_layout.addWidget(confidence_label)
            
            # Add to widget
            widget_layout.addWidget(thumb_label)
            widget_layout.addWidget(text_widget, 1)
            
            # Set widget as item
            item.setSizeHint(widget.sizeHint())
            self.proposal_list.addItem(item)
            self.proposal_list.setItemWidget(item, widget)
            
            # Store proposal reference
            item.setData(Qt.ItemDataRole.UserRole, i)
            
    def get_proposal_text(self, proposal: Dict) -> str:
        """Get text description of proposal."""
        cap = proposal.get('cap')
        if not cap:
            return "Invalid proposal"
            
        character = proposal.get('character')
        pose = proposal.get('pose')
        
        if character and pose:
            return f"Assign to: {character} → {pose}"
        elif character:
            return f"Assign to character: {character}"
        elif pose:
            return f"Assign to pose: {pose}"
        else:
            return "No assignment suggested"
            
    def refresh_proposals(self):
        """Refresh proposals (placeholder - would call AI clustering)."""
        # In a real implementation, this would call the clustering algorithm
        QMessageBox.information(self, "Refresh", 
                              "Proposal refresh would call AI clustering here.")
        
    def clear_proposals(self):
        """Clear all proposals."""
        self.proposals = []
        self.refresh_list()
        
    def update_button_states(self):
        """Update button states based on selection."""
        has_selection = len(self.proposal_list.selectedItems()) > 0
        self.accept_button.setEnabled(has_selection)
        self.reject_button.setEnabled(has_selection)
        
    def accept_selected(self):
        """Accept selected proposals."""
        for item in self.proposal_list.selectedItems():
            index = item.data(Qt.ItemDataRole.UserRole)
            if 0 <= index < len(self.proposals):
                self.proposal_accepted.emit(self.proposals[index])
                
    def reject_selected(self):
        """Reject selected proposals."""
        for item in self.proposal_list.selectedItems():
            index = item.data(Qt.ItemDataRole.UserRole)
            if 0 <= index < len(self.proposals):
                self.proposal_rejected.emit(self.proposals[index])
                
    def accept_all(self):
        """Accept all proposals."""
        for proposal in self.proposals:
            self.proposal_accepted.emit(proposal)
            
    def on_proposal_double_clicked(self, item):
        """Handle double click on proposal."""
        index = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= index < len(self.proposals):
            # Show preview dialog
            self.show_proposal_preview(self.proposals[index])
            
    def show_proposal_preview(self, proposal: Dict):
        """Show detailed preview of a proposal."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Proposal Preview")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Image preview
        cap = proposal.get('cap')
        if cap and isinstance(cap.metadata, dict):
            thumbs = cap.metadata.get('thumbnails')
            if thumbs and len(thumbs) > 0:
                pixmap = QPixmap(thumbs[0])
                if not pixmap.isNull():
                    image_label = QLabel()
                    image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))
                    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(image_label)
                    
        # Proposal details
        details = QTextEdit()
        details.setReadOnly(True)
        details.setText(f"""
        Capsule: {cap.name if cap else 'Unknown'}
        Type: {cap.type if cap else 'Unknown'}
        
        Character Match: {proposal.get('character', 'None')}
        Character Score: {proposal.get('character_score', 0):.2%}
        
        Pose Match: {proposal.get('pose', 'None')}
        Pose Score: {proposal.get('pose_score', 0):.2%}
        
        Tags: {', '.join([t[0] for t in proposal.get('tags', [])])}
        """)
        layout.addWidget(details)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.exec()


class GraphInspectorWidget(QDockWidget):
    """Dockable widget for inspecting the ROCA graph: capsules, edges, and event log."""
    
    @classmethod
    def set_graph_manager_class(cls, graph_manager_class):
        """Set the GraphManager class to use (called from main.py)."""
        global GraphManager
        GraphManager = graph_manager_class
    
    def __init__(self, graph_manager=None, capsule_manager=None, parent=None):
        super().__init__("Graph Inspector", parent)
        self.graph_manager = graph_manager
        self.capsule_manager = capsule_manager
        self.setup_ui()
        self.refresh()
        
    def setup_ui(self):
        """Setup the inspector UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header with refresh button
        header_layout = QHBoxLayout()
        title = QLabel("ROCA Graph Inspector")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Capsules tab
        capsules_tab = QWidget()
        capsules_layout = QVBoxLayout(capsules_tab)
        self.capsules_list = QListWidget()
        self.capsules_list.itemDoubleClicked.connect(self.on_capsule_double_clicked)
        capsules_layout.addWidget(QLabel("Capsules:"))
        capsules_layout.addWidget(self.capsules_list)
        self.tab_widget.addTab(capsules_tab, "Capsules")
        
        # Edges tab
        edges_tab = QWidget()
        edges_layout = QVBoxLayout(edges_tab)
        self.edges_list = QListWidget()
        edges_layout.addWidget(QLabel("Similarity Edges:"))
        edges_layout.addWidget(self.edges_list)
        self.tab_widget.addTab(edges_tab, "Edges")
        
        # Event log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        log_layout.addWidget(QLabel("Event Log:"))
        log_layout.addWidget(self.event_log)
        self.tab_widget.addTab(log_tab, "Event Log")
        
        # Cluster Inspector tab
        self.cluster_inspector = InspectorWidget(self.capsule_manager)
        self.tab_widget.addTab(self.cluster_inspector, "Cluster Inspector")
        
        layout.addWidget(self.tab_widget)
        
        self.setWidget(widget)
        
    def refresh(self):
        """Refresh the inspector display."""
        # Capsules
        self.capsules_list.clear()
        for capsule_id, capsule in self.graph_manager.graph.capsules.items():
            name = getattr(capsule, 'metadata', {}).get('name', capsule_id)
            item = QListWidgetItem(f"{name} ({capsule_id})")
            item.setData(Qt.ItemDataRole.UserRole, capsule_id)
            self.capsules_list.addItem(item)
        
        # Edges
        self.edges_list.clear()
        for capsule_id, neighbors in self.graph_manager.graph.similarity_edges.items():
            for neighbor_id in neighbors:
                capsule_name = getattr(self.graph_manager.graph.capsules.get(capsule_id), 'metadata', {}).get('name', capsule_id)
                neighbor_name = getattr(self.graph_manager.graph.capsules.get(neighbor_id), 'metadata', {}).get('name', neighbor_id)
                self.edges_list.addItem(f"{capsule_name} ↔ {neighbor_name}")
        
        # Event log
        log_text = ""
        for i, command in enumerate(self.graph_manager.event_log[-50:]):  # Last 50 events
            log_text += f"{i+1}: {command.type.value} - {command.payload}\n"
        self.event_log.setPlainText(log_text)
        
    def on_capsule_double_clicked(self, item):
        """Handle double-click on capsule to show details."""
        capsule_id = item.data(Qt.ItemDataRole.UserRole)
        capsule = self.graph_manager.graph.capsules.get(capsule_id)
        if capsule:
            # Show a dialog with capsule details
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Capsule Details: {capsule_id}")
            layout = QVBoxLayout(dialog)
            
            details = QTextEdit()
            details.setPlainText(str(vars(capsule)))
            details.setReadOnly(True)
            layout.addWidget(details)
            
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            button_box.accepted.connect(dialog.accept)
            layout.addWidget(button_box)
            
            dialog.exec()


# ============================================================================
# MICROPHONE WIDGET
# ============================================================================

class MicrophoneWidget(QWidget):
    """Widget for recording and transcribing speech input."""
    
    speech_detected = pyqtSignal(dict)  # Emits transcription data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self.recognizer = None
        self.microphone = None
        
        self.setup_ui()
        self.initialize_speech_recognition()
        
    def setup_ui(self):
        """Setup the microphone interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Speech Input")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Status label
        self.status_label = QLabel("Ready to record")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #8888aa;
                font-size: 10px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Set size policy
        self.setMaximumHeight(80)
        
    def initialize_speech_recognition(self):
        """Initialize speech recognition components."""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
        except ImportError as e:
            self.status_label.setText("Speech recognition not available")
            print(f"Speech recognition import failed: {e}")
        except Exception as e:
            self.status_label.setText("Microphone setup failed")
            print(f"Microphone setup failed: {e}")
            
    def start_recording(self):
        """Start recording speech."""
        if not self.recognizer or not self.microphone:
            return
            
        self.is_recording = True
        self.status_label.setText("🎤 Listening...")
        
        # Start recording in a separate thread
        try:
            import speech_recognition as sr
            from PyQt6.QtCore import QThread, pyqtSignal
            
            class RecordingThread(QThread):
                finished = pyqtSignal(dict)
                error = pyqtSignal(str)
                
                def __init__(self, recognizer, microphone):
                    super().__init__()
                    self.recognizer = recognizer
                    self.microphone = microphone
                    
                def run(self):
                    try:
                        with self.microphone as source:
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            
                        # Transcribe
                        text = self.recognizer.recognize_google(audio)
                        
                        result = {
                            'transcription': text,
                            'timestamp': time.time(),
                            'confidence': 0.8  # Google doesn't provide confidence
                        }
                        
                        self.finished.emit(result)
                        
                    except sr.WaitTimeoutError:
                        self.error.emit("No speech detected")
                    except sr.UnknownValueError:
                        self.error.emit("Could not understand audio")
                    except sr.RequestError as e:
                        self.error.emit(f"Speech service error: {e}")
                    except Exception as e:
                        self.error.emit(f"Recording error: {e}")
            
            self.recording_thread = RecordingThread(self.recognizer, self.microphone)
            self.recording_thread.finished.connect(self.on_recording_finished)
            self.recording_thread.error.connect(self.on_recording_error)
            self.recording_thread.start()
            
        except Exception as e:
            self.on_recording_error(f"Failed to start recording: {e}")
            
    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        self.status_label.setText("Processing...")
        
    def on_recording_finished(self, result):
        """Handle successful recording."""
        self.is_recording = False
        
        transcription = result.get('transcription', '')
        self.status_label.setText(f"Transcribed: {transcription[:30]}{'...' if len(transcription) > 30 else ''}")
        
        # Emit the speech data
        self.speech_detected.emit(result)
        
    def on_recording_error(self, error_msg):
        """Handle recording error."""
        self.is_recording = False
        self.status_label.setText(f"Error: {error_msg[:30]}{'...' if len(error_msg) > 30 else ''}")


# ============================================================================
# DRAWING WIDGET
# ============================================================================

class DrawingWidget(QWidget):
    """Widget for capturing artistic input through drawing."""
    
    drawing_completed = pyqtSignal(dict)  # Emits drawing data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.last_point = None
        self.current_stroke = []
        self.strokes = []
        self.colors = []
        self.current_color = QColor(0, 0, 0)  # Black
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the drawing interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Artistic Input")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # Canvas
        self.canvas = QWidget()
        self.canvas.setMinimumSize(300, 200)
        self.canvas.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #444466;
                border-radius: 4px;
            }
        """)
        self.canvas.mousePressEvent = self.canvas_mouse_press
        self.canvas.mouseMoveEvent = self.canvas_mouse_move
        self.canvas.mouseReleaseEvent = self.canvas_mouse_release
        self.canvas.paintEvent = self.canvas_paint
        layout.addWidget(self.canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Color buttons
        colors = [
            ("Black", QColor(0, 0, 0)),
            ("Red", QColor(255, 0, 0)),
            ("Green", QColor(0, 255, 0)),
            ("Blue", QColor(0, 0, 255)),
            ("Yellow", QColor(255, 255, 0)),
            ("Purple", QColor(128, 0, 128))
        ]
        
        for name, color in colors:
            color_btn = QPushButton(f"●")
            color_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    border: 2px solid #666688;
                    border-radius: 12px;
                    min-width: 24px;
                    max-width: 24px;
                    min-height: 24px;
                    max-height: 24px;
                }}
                QPushButton:hover {{
                    border: 2px solid #8888aa;
                }}
            """)
            color_btn.clicked.connect(lambda checked, c=color: self.set_color(c))
            controls_layout.addWidget(color_btn)
        
        controls_layout.addStretch()
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a4a4a;
                color: #ffffff;
                border: 1px solid #886666;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #7a5a5a;
            }
        """)
        clear_btn.clicked.connect(self.clear_canvas)
        controls_layout.addWidget(clear_btn)
        
        # Analyze button
        analyze_btn = QPushButton("Analyze Drawing")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a6a4a;
                color: #ffffff;
                border: 1px solid #668866;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a7a5a;
            }
        """)
        analyze_btn.clicked.connect(self.analyze_drawing)
        controls_layout.addWidget(analyze_btn)
        
        layout.addLayout(controls_layout)
        
        # Status
        self.status_label = QLabel("Draw on the canvas above")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #8888aa;
                font-size: 10px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.status_label)
        
    def set_color(self, color):
        """Set the current drawing color."""
        self.current_color = color
        
    def clear_canvas(self):
        """Clear the canvas."""
        self.strokes.clear()
        self.colors.clear()
        self.canvas.update()
        self.status_label.setText("Canvas cleared")
        
    def canvas_mouse_press(self, event):
        """Handle mouse press on canvas."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.current_stroke = [event.pos()]
            self.colors.append(self.current_color)
            
    def canvas_mouse_move(self, event):
        """Handle mouse move on canvas."""
        if self.drawing and self.last_point:
            # Draw line from last point to current point
            painter = QPainter(self.canvas)
            painter.setPen(QPen(self.current_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            
            self.last_point = event.pos()
            self.current_stroke.append(event.pos())
            
    def canvas_mouse_release(self, event):
        """Handle mouse release on canvas."""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            if self.current_stroke:
                self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self.status_label.setText(f"Drawing completed - {len(self.strokes)} strokes")
            
    def canvas_paint(self, event):
        """Paint the canvas (for redraws)."""
        # Canvas is painted by direct drawing, no need for paint event handling
        pass
        
    def analyze_drawing(self):
        """Analyze the current drawing and emit the data."""
        if not self.strokes:
            self.status_label.setText("No drawing to analyze")
            return
            
        # Convert strokes to serializable format
        stroke_data = []
        for stroke in self.strokes:
            stroke_data.append([(p.x(), p.y()) for p in stroke])
            
        color_data = [color.name() for color in self.colors]
        
        drawing_data = {
            'strokes': stroke_data,
            'colors': color_data,
            'timestamp': time.time(),
            'canvas_size': (self.canvas.width(), self.canvas.height())
        }
        
        self.drawing_completed.emit(drawing_data)
        self.status_label.setText(f"Analyzed drawing with {len(self.strokes)} strokes")


# ============================================================================
# LEARNING DOCK WIDGET
# ============================================================================

class LearningDockWidget(QDockWidget):
    """Dock widget containing microphone and drawing widgets for user learning."""
    
    def __init__(self, learning_system, parent=None):
        super().__init__("User Learning Input", parent)
        self.learning_system = learning_system
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the learning input interface."""
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Drawing widget
        self.drawing_widget = DrawingWidget()
        self.drawing_widget.drawing_completed.connect(self.on_drawing_completed)
        layout.addWidget(self.drawing_widget)
        
        self.setWidget(container)
        self.setMinimumWidth(350)
        self.setMinimumHeight(400)
        
    def on_drawing_completed(self, drawing_data):
        """Handle drawing input for learning."""
        if self.learning_system:
            self.learning_system.analyze_drawing(drawing_data)


# ============================================================================
# TEXTURE EDITOR DIALOG
# ============================================================================

class TextureEditorDialog(QDialog):
    """Dialog for comprehensive texture editing and artistic input."""

    texture_updated = pyqtSignal(dict)  # Emits texture data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Texture Editor & Artistic Input")
        self.setModal(False)  # Non-modal dialog
        self.resize(800, 600)

        self.current_texture = None
        self.brush_size = 5
        self.brush_color = QColor(0, 0, 0)
        self.brush_opacity = 1.0

        self.setup_ui()

    def setup_ui(self):
        """Setup the texture editor interface."""
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Tools and settings
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)

        # Right panel - Canvas and layers
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)

        content_splitter.setSizes([200, 600])
        layout.addWidget(content_splitter)

        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QHBoxLayout()

        # File operations
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout(file_group)

        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_texture)
        file_layout.addWidget(new_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_texture)
        file_layout.addWidget(load_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_texture)
        file_layout.addWidget(save_btn)

        toolbar.addWidget(file_group)

        # Brush settings
        brush_group = QGroupBox("Brush")
        brush_layout = QHBoxLayout(brush_group)

        size_label = QLabel("Size:")
        brush_layout.addWidget(size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(self.brush_size)
        self.size_slider.valueChanged.connect(self.on_brush_size_changed)
        brush_layout.addWidget(self.size_slider)

        self.size_value = QLabel(f"{self.brush_size}")
        brush_layout.addWidget(self.size_value)

        toolbar.addWidget(brush_group)

        # Color picker
        color_group = QGroupBox("Color")
        color_layout = QHBoxLayout(color_group)

        self.color_button = QPushButton()
        self.color_button.setFixedSize(30, 30)
        self.color_button.setStyleSheet(f"background-color: {self.brush_color.name()}; border: 1px solid #000;")
        self.color_button.clicked.connect(self.pick_color)
        color_layout.addWidget(self.color_button)

        opacity_label = QLabel("Opacity:")
        color_layout.addWidget(opacity_label)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.brush_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        color_layout.addWidget(self.opacity_slider)

        self.opacity_value = QLabel(f"{int(self.brush_opacity * 100)}%")
        color_layout.addWidget(self.opacity_value)

        toolbar.addWidget(color_group)

        toolbar.addStretch()
        return toolbar

    def create_left_panel(self):
        """Create the left panel with tools and settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tools
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)

        self.brush_tool = QRadioButton("Brush")
        self.brush_tool.setChecked(True)
        tools_layout.addWidget(self.brush_tool)

        self.eraser_tool = QRadioButton("Eraser")
        tools_layout.addWidget(self.eraser_tool)

        self.fill_tool = QRadioButton("Fill")
        tools_layout.addWidget(self.fill_tool)

        layout.addWidget(tools_group)

        # Drawing widget (artistic input)
        drawing_group = QGroupBox("Artistic Input")
        drawing_layout = QVBoxLayout(drawing_group)

        self.drawing_widget = DrawingWidget()
        self.drawing_widget.drawing_completed.connect(self.on_drawing_completed)
        drawing_layout.addWidget(self.drawing_widget)

        layout.addWidget(drawing_group)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        """Create the right panel with canvas and layers."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Canvas area
        canvas_group = QGroupBox("Canvas")
        canvas_layout = QVBoxLayout(canvas_group)

        # Create a simple canvas for now (can be expanded later)
        self.canvas = QLabel("Texture Canvas - Click to start painting")
        self.canvas.setMinimumSize(400, 400)
        self.canvas.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                qproperty-alignment: AlignCenter;
            }
        """)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_layout.addWidget(self.canvas)

        layout.addWidget(canvas_group)

        # Layer controls (placeholder for future expansion)
        layers_group = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_group)

        layers_label = QLabel("Layer system coming soon...")
        layers_layout.addWidget(layers_label)

        layout.addWidget(layers_group)

        return panel

    def new_texture(self):
        """Create a new texture."""
        self.current_texture = {
            'width': 512,
            'height': 512,
            'data': None,  # Will hold image data
            'name': 'New Texture'
        }
        self.status_label.setText("New texture created")
        self.update_canvas()

    def load_texture(self):
        """Load a texture from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Texture", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            # Load texture logic here
            self.status_label.setText(f"Loaded texture: {file_path}")

    def save_texture(self):
        """Save the current texture."""
        if not self.current_texture:
            QMessageBox.warning(self, "No Texture", "No texture to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Texture", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if file_path:
            # Save texture logic here
            self.status_label.setText(f"Saved texture: {file_path}")

    def pick_color(self):
        """Pick a brush color."""
        color = QColorDialog.getColor(self.brush_color, self, "Pick Brush Color")
        if color.isValid():
            self.brush_color = color
            self.color_button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #000;")

    def on_brush_size_changed(self, value):
        """Handle brush size change."""
        self.brush_size = value
        self.size_value.setText(str(value))

    def on_opacity_changed(self, value):
        """Handle opacity change."""
        self.brush_opacity = value / 100.0
        self.opacity_value.setText(f"{value}%")

    def on_drawing_completed(self, drawing_data):
        """Handle drawing completion from artistic input."""
        self.status_label.setText("Drawing captured for texture processing")
        # Process the drawing data for texture creation
        self.texture_updated.emit(drawing_data)

    def update_canvas(self):
        """Update the canvas display."""
        if self.current_texture:
            self.canvas.setText(f"Texture: {self.current_texture['name']} ({self.current_texture['width']}x{self.current_texture['height']})")
        else:
            self.canvas.setText("No texture loaded - Click 'New' to create one")

    def showEvent(self, event):
        """Called when dialog is shown."""
        super().showEvent(event)
        self.status_label.setText("Texture Editor opened")

    def closeEvent(self, event):
        """Called when dialog is closed."""
        self.status_label.setText("Texture Editor closed")
        super().closeEvent(event)


# ============================================================================
# VOICE DIALOG
# ============================================================================

class VoiceDialog(QDialog):
    """Dialog window for voice system controls."""

    def __init__(self, voice_system, parent=None):
        super().__init__(parent)
        self.voice_system = voice_system
        self.setWindowTitle("Voice System")
        self.setModal(False)  # Non-modal dialog
        self.setMinimumSize(500, 400)

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Setup the voice dialog interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎤 Voice System Control")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Voice widget (embedded)
        try:
            from voice_system import VoiceWidget
            self.voice_widget = VoiceWidget(self.voice_system)
            layout.addWidget(self.voice_widget)
        except ImportError:
            # Fallback if VoiceWidget not available
            error_label = QLabel("Voice system components not available.\nPlease check your installation.")
            error_label.setStyleSheet("color: red; padding: 20px;")
            layout.addWidget(error_label)

        # Status bar
        self.status_bar = QLabel("Voice system ready")
        self.status_bar.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 10px;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_bar)

        # Buttons
        button_layout = QHBoxLayout()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def connect_signals(self):
        """Connect signals from voice system."""
        if hasattr(self.voice_system, 'speech_recognized'):
            self.voice_system.speech_recognized.connect(self.on_speech_recognized)

    def on_speech_recognized(self, text: str, confidence: float):
        """Handle speech recognition results."""
        self.status_bar.setText(f"Recognized: '{text}' (confidence: {confidence:.2f})")

    def showEvent(self, event):
        """Called when dialog is shown."""
        super().showEvent(event)
        self.status_bar.setText("Voice dialog opened")

    def closeEvent(self, event):
        """Called when dialog is closed."""
        # Stop listening if active
        if hasattr(self.voice_system, 'is_active') and self.voice_system.is_active:
            self.voice_system.stop_listening()
        self.status_bar.setText("Voice dialog closed")
        super().closeEvent(event)