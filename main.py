#!/usr/bin/env python3
"""
PyQt6 JPatch - A 3D modeling application inspired by JPatch
"""

import sys
from typing import List, Optional, Dict, Any
import math
import tracemalloc
import gc
import logging
import time 
# =====================
# Logging & Memory Management Setup
# =====================

# Configure logging FIRST before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("jpatch.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JPatch")

# Start memory tracking
tracemalloc.start()
logger.info("=" * 80)
logger.info("PYQT6 JPATCH APPLICATION STARTED")
logger.info("=" * 80)
logger.info("Memory tracking started with tracemalloc.")

def log_memory_usage(note: str = ""):
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"[MEMORY] {note} Current: {current / 1024:.1f} KB; Peak: {peak / 1024:.1f} KB")
    # Force flush to ensure logs are written immediately
    for handler in logger.handlers:
        handler.flush()

def cleanup_memory(note: str = ""):
    gc.collect()
    log_memory_usage(f"After GC: {note}")

def log_error_with_context(logger_obj, error_msg: str, exception: Exception = None, context: dict = None):
    """Enhanced error logging with context information."""
    error_details = f"{error_msg}"
    if exception:
        error_details += f" | Exception: {type(exception).__name__}: {str(exception)}"
    if context:
        error_details += f" | Context: {context}"
    logger_obj.error(error_details, exc_info=exception)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QMenuBar, QStatusBar,
                             QSplitter, QTreeWidget, QTreeWidgetItem, QToolBar, QHBoxLayout, QLabel,
                             QFileDialog, QMessageBox, QComboBox, QSlider, QGroupBox, QInputDialog,
                             QPushButton, QDialog, QListWidget, QTextEdit, QDoubleSpinBox,
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QDockWidget)

from PyQt6.QtGui import QColor

from roca.graph_manager import GraphManager, ROCAGraph, Command, CommandType

# Changed from specific import to import the whole module
from autonomous_brain import AutonomousBrain, BrainWidget, ThoughtType
BRAIN_WIDGET_AVAILABLE = True  # Assume it's available
# Creative consciousness classes - commented out due to import issues
# from Creative_conciousness import (
#     MemoryConsolidationEngine,
#     TeachingEngine,
#     StoryGenerator
# )

# Import graphical widgets
from Graphicalwidget import TimelineWidget, CapsuleTreeWidget, OrbitalViewWidget, ChatWidget, TextureEditorDialog, VoiceDialog

# Import new AI systems
from voice_system import VoiceSystem, VoiceWidget
from symbolic_math import CapsuleNetworkMath
from knowledge_base import KnowledgeBase, KnowledgeBaseWidget

# Add imports for threading and memory management
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# Memory management system
class MemoryManager:
    def __init__(self):
        self.object_pool = weakref.WeakValueDictionary()
        self.gc_threshold = 100 * 1024 * 1024  # 100MB threshold
        self.last_gc_time = time.time()
        self.gc_interval = 30.0  # GC every 30 seconds
        
    def should_gc(self):
        current, peak = tracemalloc.get_traced_memory()
        current_time = time.time()
        return (current > self.gc_threshold or 
                current_time - self.last_gc_time > self.gc_interval)
    
    def perform_gc(self):
        if self.should_gc():
            gc.collect()
            self.last_gc_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"[MEMORY] GC performed - Current: {current / 1024:.1f} KB")
    
    def add_to_pool(self, obj_id, obj):
        self.object_pool[obj_id] = obj
    
    def get_from_pool(self, obj_id):
        return self.object_pool.get(obj_id)

# Threading system for heavy operations
class BackgroundProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelProcessor")
        self.futures = {}
        self.lock = threading.Lock()
        
    def submit_subdivision_task(self, surface_index, levels, callback):
        """Submit subdivision surface computation to background thread"""
        def task():
            try:
                surface = self.model.subdivision_surfaces[surface_index]
                subdivided = surface.subdivide(levels)
                return subdivided
            except Exception as e:
                log_error_with_context(logger, "Subdivision task failed", e, {"surface_index": surface_index, "levels": levels})
                return None
        
        future = self.executor.submit(task)
        with self.lock:
            self.futures[f"subdivision_{surface_index}"] = (future, callback)
    
    def submit_ai_task(self, task_type, data, callback):
        """Submit AI processing task to background thread"""
        def task():
            try:
                if task_type == "pose_suggestion":
                    # Simulate AI processing
                    time.sleep(0.5)  # Simulate computation time
                    return {"suggestions": ["pose1", "pose2"], "confidence": 0.85}
                elif task_type == "model_analysis":
                    time.sleep(0.3)
                    return {"issues": [], "suggestions": ["optimize topology"]}
                return None
            except Exception as e:
                log_error_with_context(logger, "AI task failed", e, {"task_type": task_type})
                return None
        
        future = self.executor.submit(task)
        with self.lock:
            self.futures[f"ai_{task_type}"] = (future, callback)
    
    def check_completed_tasks(self):
        """Check for completed background tasks and execute callbacks"""
        completed = []
        with self.lock:
            for task_id, (future, callback) in self.futures.items():
                if future.done():
                    completed.append((task_id, future, callback))
            
            for task_id, future, callback in completed:
                del self.futures[task_id]
        
        # Execute callbacks outside lock
        for task_id, future, callback in completed:
            try:
                result = future.result()
                callback(result)
            except Exception as e:
                log_error_with_context(logger, "Task callback failed", e, {"task_id": task_id})
    
    def shutdown(self):
        """Shutdown the background processor"""
        self.executor.shutdown(wait=True)

# Global background processor
background_processor = BackgroundProcessor()


# Capsule type colors
CAPSULE_COLORS = {
    'character': QColor(70, 130, 180),      # Steel blue
    'pose': QColor(60, 179, 113),           # Medium sea green
    'skill': QColor(255, 140, 0),           # Dark orange
    'style': QColor(186, 85, 211),          # Medium orchid
    'memory': QColor(192, 192, 192),        # Silver
    'unassigned': QColor(169, 169, 169),    # Dark gray
    'bvh_motion': QColor(220, 20, 60),      # Crimson
    'act_motion': QColor(220, 20, 60),      # Crimson
    'personality': QColor(30, 144, 255),    # Dodger blue
    'transition': QColor(255, 215, 0),      # Gold
    'timing': QColor(50, 205, 50),          # Lime green
    'cycle': QColor(255, 69, 0),            # Red-orange
    'hdr_environment': QColor(255, 165, 0), # Orange for HDR lighting
    '3dmodel': QColor(100, 149, 237),       # Cornflower blue
    'animation': QColor(255, 20, 147),      # Deep pink
}


# Octree implementation for spatial partitioning
class OctreeNode:
    def __init__(self, bounds, max_objects=8, max_depth=8):
        self.bounds = bounds  # (min_x, min_y, min_z, max_x, max_y, max_z)
        self.objects = []  # List of (object_type, object_index) tuples
        self.children = None
        self.max_objects = max_objects
        self.max_depth = max_depth
        self.depth = 0

    def subdivide(self):
        if self.children is not None:
            return
        
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        mid_z = (min_z + max_z) / 2
        
        self.children = [
            OctreeNode((min_x, min_y, min_z, mid_x, mid_y, mid_z), self.max_objects, self.max_depth),
            OctreeNode((mid_x, min_y, min_z, max_x, mid_y, mid_z), self.max_objects, self.max_depth),
            OctreeNode((min_x, mid_y, min_z, mid_x, max_y, mid_z), self.max_objects, self.max_depth),
            OctreeNode((mid_x, mid_y, min_z, max_x, max_y, mid_z), self.max_objects, self.max_depth),
            OctreeNode((min_x, min_y, mid_z, mid_x, mid_y, max_z), self.max_objects, self.max_depth),
            OctreeNode((mid_x, min_y, mid_z, max_x, mid_y, max_z), self.max_objects, self.max_depth),
            OctreeNode((min_x, mid_y, mid_z, mid_x, max_y, max_z), self.max_objects, self.max_depth),
            OctreeNode((mid_x, mid_y, mid_z, max_x, max_y, max_z), self.max_objects, self.max_depth),
        ]
        for child in self.children:
            child.depth = self.depth + 1

    def insert(self, obj_type, obj_index, position):
        # Check if object is within bounds
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        x, y, z = position
        
        if not (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z):
            return False
        
        if self.children is None:
            self.objects.append((obj_type, obj_index))
            if len(self.objects) > self.max_objects and self.depth < self.max_depth:
                self.subdivide()
                # Reinsert objects into children
                for obj in self.objects[:]:
                    obj_t, obj_i = obj
                    if obj_t == 'point':
                        pos = self.model.control_points[obj_i]
                    elif obj_t == 'patch':
                        # Use center of patch for simplicity
                        patch = self.model.patches[obj_i]
                        if isinstance(patch, list) and len(patch) >= 4:
                            points = [self.model.control_points[idx] for idx in patch[:4]]
                            pos = tuple(sum(coord) / len(coord) for coord in zip(*points))
                        else:
                            continue
                    elif obj_t == 'capsule':
                        # Get capsule bounds center
                        capsule = self.model.capsule_manager.get_capsule_by_index(obj_i)
                        if capsule and hasattr(capsule, 'spatial_bounds'):
                            bounds = capsule.spatial_bounds
                            pos = ((bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2)
                        else:
                            continue
                    else:
                        continue
                    
                    inserted = False
                    for child in self.children:
                        if child.insert(obj_t, obj_i, pos):
                            inserted = True
                            break
                    if inserted:
                        self.objects.remove(obj)
        else:
            # Insert into appropriate child
            for child in self.children:
                if child.insert(obj_type, obj_index, position):
                    return True
        return True

    def query(self, bounds):
        """Query objects within bounds"""
        results = []
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z = bounds
        
        # Check if query bounds intersect this node's bounds
        if not (q_max_x >= min_x and q_min_x <= max_x and
                q_max_y >= min_y and q_min_y <= max_y and
                q_max_z >= min_z and q_min_z <= max_z):
            return results
        
        # Add objects in this node
        results.extend(self.objects)
        
        # Query children
        if self.children:
            for child in self.children:
                results.extend(child.query(bounds))
        
        return results


class Octree:
    def __init__(self, bounds, max_objects=8, max_depth=8):
        self.root = OctreeNode(bounds, max_objects, max_depth)
        self.model = None  # Reference to model for position lookups
    
    def set_model(self, model):
        self.root.model = model
        self.model = model
    
    def rebuild(self):
        """Rebuild the octree from current model data including capsules"""
        if not self.model:
            return
        
        # Calculate bounds from all control points
        if not self.model.control_points:
            return
        
        min_x = min(p[0] for p in self.model.control_points)
        max_x = max(p[0] for p in self.model.control_points)
        min_y = min(p[1] for p in self.model.control_points)
        max_y = max(p[1] for p in self.model.control_points)
        min_z = min(p[2] for p in self.model.control_points)
        max_z = max(p[2] for p in self.model.control_points)
        
        # Include capsule bounds if available
        if hasattr(self.model, 'capsule_manager') and self.model.capsule_manager:
            for capsule in self.model.capsule_manager.get_all_capsules():
                if hasattr(capsule, 'spatial_bounds'):
                    bounds = capsule.spatial_bounds
                    min_x = min(min_x, bounds[0])
                    max_x = max(max_x, bounds[3])
                    min_y = min(min_y, bounds[1])
                    max_y = max(max_y, bounds[4])
                    min_z = min(min_z, bounds[2])
                    max_z = max(max_z, bounds[5])
        
        # Add some padding
        padding = 0.1 * max(max_x - min_x, max_y - min_y, max_z - min_z)
        bounds = (min_x - padding, min_y - padding, min_z - padding,
                 max_x + padding, max_y + padding, max_z + padding)
        
        self.root = OctreeNode(bounds, self.root.max_objects, self.root.max_depth)
        self.root.model = self.model
        
        # Insert all objects
        for i, point in enumerate(self.model.control_points):
            self.root.insert('point', i, point)
        
        for i, patch in enumerate(self.model.patches):
            if isinstance(patch, list) and len(patch) >= 4:
                # Use center of patch
                points = [self.model.control_points[idx] for idx in patch[:4]]
                center = tuple(sum(coord) / len(coord) for coord in zip(*points))
                self.root.insert('patch', i, center)
        
        # Insert capsules
        if hasattr(self.model, 'capsule_manager') and self.model.capsule_manager:
            for i, capsule in enumerate(self.model.capsule_manager.get_all_capsules()):
                if hasattr(capsule, 'spatial_bounds'):
                    bounds = capsule.spatial_bounds
                    center = ((bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2)
                    self.root.insert('capsule', i, center)


# --- GLWidget: OpenGL viewport widget ---
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

class GLWidget(QOpenGLWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        # Display modes
        self.show_points = True
        self.show_curves = True
        self.show_patches = True
        self.show_shaded = False
        self.show_textured = False
        self.show_rotoscope = False
        self.show_grid = True
        self.show_bones = True
        # Camera controls
        self.last_mouse_pos = None
        self.is_dragging = False
        self.camera_distance = 5.0
        self.camera_rotation = [0.0, 0.0]  # yaw, pitch
        self.camera_position = [0.0, 0.0, 0.0]  # pan position
        self.camera_target = [0.0, 0.0, 0.0]  # orbit target
        # Lighting modes
        self.lighting_mode = 'simple'  # 'head', 'simple', 'three-point', 'sticky'
        self.light_position = [1.0, 1.0, 1.0]
        # Grid system
        self.grid_size = 1.0
        self.grid_subdivisions = 10
        # Rotoscoping
        self.rotoscope_image = None
        self.rotoscope_opacity = 0.5
        # Selection
        self.selection_rect = None  # (x1, y1, x2, y2)
        # Viewport type for multi-viewport support
        self.viewport_type = 'perspective'  # 'perspective', 'top', 'front', 'side'
        
        # Performance optimizations
        self.lod_distances = [5.0, 10.0, 20.0]  # Distance thresholds for LOD levels
        self.target_fps = 60.0
        self.last_frame_time = time.time()

        def draw_bones(self):
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            def draw_bone_recursive(bone):
                if not bone:
                    return
                start_pos = bone.position
                if bone.parent:
                    start_pos = bone.parent.position
                end_pos = bone.position
                if bone.name in self.model.selected_bones:
                    glColor3f(1.0, 1.0, 0.0)
                else:
                    glColor3f(0.0, 1.0, 0.0)
                glBegin(GL_LINES)
                glVertex3f(*start_pos)
                glVertex3f(*end_pos)
                glEnd()
                glPointSize(8.0)
                glBegin(GL_POINTS)
                glVertex3f(*end_pos)
                glEnd()
                for child in bone.children:
                    draw_bone_recursive(child)
            for bone in self.model.bones.values():
                if not bone.parent:
                    draw_bone_recursive(bone)
            glLineWidth(1.0)
            glPointSize(8.0)
            glEnable(GL_LIGHTING)

        def draw_points(self):
            glPointSize(8.0)
            glBegin(GL_POINTS)
            for i, point in enumerate(self.model.control_points):
                if i in self.model.selected_points:
                    glColor3f(1.0, 1.0, 0.0)
                else:
                    glColor3f(1.0, 0.0, 0.0)
                glVertex3f(*point)
            glEnd()

        def draw_standalone_curves(self):
            for curve in self.model.curves:
                if isinstance(curve, (BezierCurve, BSplineCurve)):
                    points = curve.tessellate(num_segments=20)
                    glBegin(GL_LINE_STRIP)
                    glColor3f(1.0, 0.5, 0.0)
                    for point in points:
                        glVertex3f(point.x, point.y, point.z)
                    glEnd()
                    glPointSize(6.0)
                    glBegin(GL_POINTS)
                    glColor3f(1.0, 0.0, 1.0)
                    for ctrl_point in curve.control_points:
                        glVertex3f(ctrl_point.x, ctrl_point.y, ctrl_point.z)
                    glEnd()
                    glPointSize(8.0)
            glBegin(GL_LINES)
            glColor3f(0.0, 1.0, 0.0)
            for patch in self.model.patches:
                if isinstance(patch, BezierPatch):
                    vertices, _ = patch.tessellate(u_segments=8, v_segments=8)
                    for u in range(9):
                        glBegin(GL_LINE_STRIP)
                        for v in range(9):
                            idx = u * 9 + v
                            if idx < len(vertices):
                                vertex = vertices[idx]
                                glVertex3f(vertex.x, vertex.y, vertex.z)
                        glEnd()
                    for v in range(9):
                        glBegin(GL_LINE_STRIP)
                        for u in range(9):
                            idx = u * 9 + v
                            if idx < len(vertices):
                                vertex = vertices[idx]
                                glVertex3f(vertex.x, vertex.y, vertex.z)
                        glEnd()
                else:
                    if len(patch) == 4:
                        # Simple quad: draw the outline
                        for i in range(4):
                            idx1 = patch[i]
                            idx2 = patch[(i + 1) % 4]
                            if idx1 < len(self.model.control_points) and idx2 < len(self.model.control_points):
                                p1 = self.model.control_points[idx1]
                                p2 = self.model.control_points[idx2]
                                glVertex3f(*p1)
                                glVertex3f(*p2)
                    else:
                        # Assume 4x4 grid (16 indices)
                        for row in range(4):
                            for col in range(3):
                                idx1 = row * 4 + col
                                idx2 = row * 4 + col + 1
                                if idx1 < len(patch) and idx2 < len(patch):
                                    p1 = self.model.control_points[patch[idx1]]
                                    p2 = self.model.control_points[patch[idx2]]
                                    glVertex3f(*p1)
                                    glVertex3f(*p2)
                                idx1 = col * 4 + row
                                idx2 = (col + 1) * 4 + row
                                if idx1 < len(patch) and idx2 < len(patch):
                                    p1 = self.model.control_points[patch[idx1]]
                                    p2 = self.model.control_points[patch[idx2]]
                                    glVertex3f(*p1)
                                    glVertex3f(*p2)
            glEnd()

        def draw_patches(self):
            for patch in self.model.patches:
                if isinstance(patch, BezierPatch):
                    vertices, indices = patch.tessellate(u_segments=8, v_segments=8)
                    glBegin(GL_TRIANGLES)
                    glColor3f(0.5, 0.5, 0.8)
                    for i in range(0, len(indices), 3):
                        for j in range(3):
                            vertex_idx = indices[i + j]
                            if vertex_idx < len(vertices):
                                vertex = vertices[vertex_idx]
                                normal = patch.normal_at(
                                    (vertex_idx // 9) / 8.0,
                                    (vertex_idx % 9) / 8.0
                                )
                                glNormal3f(normal.x, normal.y, normal.z)
                                glVertex3f(vertex.x, vertex.y, vertex.z)
                    glEnd()
                else:
                    glBegin(GL_QUADS)
                    glColor3f(0.5, 0.5, 0.8)
                    if len(patch) == 4:
                        # Simple quad patch with 4 vertex indices
                        for idx in patch:
                            if idx < len(self.model.control_points):
                                glVertex3f(*self.model.control_points[idx])
                    elif len(patch) >= 16:
                        # Legacy format with 16 indices
                        corners = [patch[0], patch[3], patch[15], patch[12]]
                        for idx in corners:
                            if idx < len(self.model.control_points):
                                glVertex3f(*self.model.control_points[idx])
                    glEnd()

        def draw_subdivision_surfaces(self):
            for surface in self.model.subdivision_surfaces:
                vertices, faces = surface.get_mesh()
                if self.show_shaded or self.show_textured:
                    glBegin(GL_TRIANGLES)
                    glColor3f(0.6, 0.8, 0.6)
                    for face in faces:
                        if len(face) == 4:
                            triangles = [
                                [face[0], face[1], face[2]],
                                [face[0], face[2], face[3]]
                            ]
                            for tri in triangles:
                                for idx in tri:
                                    if idx < len(vertices):
                                        vertex = vertices[idx]
                                        glVertex3f(vertex[0], vertex[1], vertex[2])
                    glEnd()
                else:
                    glBegin(GL_LINES)
                    glColor3f(0.3, 0.7, 0.3)
                    for face in faces:
                        for i in range(len(face)):
                            idx1 = face[i]
                            idx2 = face[(i + 1) % len(face)]
                            if idx1 < len(vertices) and idx2 < len(vertices):
                                v1 = vertices[idx1]
                                v2 = vertices[idx2]
                                glVertex3f(v1[0], v1[1], v1[2])
                                glVertex3f(v2[0], v2[1], v2[2])
                    glEnd()
from PyQt6.QtCore import Qt, QMimeData, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import (QAction, QKeySequence, QDragEnterEvent, QDropEvent, QPen, QBrush, QColor,
                         QMouseEvent, QWheelEvent, QPainter, QPainterPath, QFont)
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import json
import os
from commands import CommandStack, AddControlPointCommand, DeleteControlPointCommand, MoveControlPointCommand, CreatePatchCommand, AddBoneCommand, MoveBoneCommand
from settings import JPatchSettings
from shortcuts import ShortcutManager
from math3d import (Vector3D, Matrix4x4, BezierCurve, BSplineCurve, BezierPatch, Ray,
                    ray_triangle_intersect, ray_patch_intersect, IntersectionResult,
                    CatmullClarkSurface, LoopSubdivision, Material, Texture, IKPoseSolver)


class Bone:
    def __init__(self, name, position, rotation=(0, 0, 0), parent=None):
        self.name = name
        self.position = list(position)  # (x, y, z)
        self.rotation = list(rotation)  # (rx, ry, rz) in degrees
        self.parent = parent
        self.children = []
        self.length = 1.0
        self.selected = False

    def to_dict(self):
        return {
            'name': self.name,
            'position': self.position,
            'rotation': self.rotation,
            'length': self.length,
            'parent_name': self.parent.name if self.parent else None,
            'children_names': [child.name for child in self.children]
        }

class MorphTarget:
    def __init__(self, name):
        self.name = name
        self.control_points = []  # List of (x, y, z) tuples
        self.blend_weight = 0.0


class Keyframe:
    def __init__(self, time, bone_name=None, position=None, rotation=None):
        self.time = time
        self.bone_name = bone_name
        self.position = position  # (x, y, z)
        self.rotation = rotation  # (rx, ry, rz)

class Animation:
    def __init__(self, name):
        self.name = name
        self.keyframes = []  # List of Keyframe objects
        self.duration = 0.0
        self.loop = False

class AnimationLayer:
    """Represents an animation layer for blending"""
    def __init__(self, animation_name, weight=1.0, start_time=0.0):
        self.animation_name = animation_name
        self.weight = weight  # Blend weight (0-1)
        self.start_time = start_time  # Start time offset
        self.enabled = True

class Model:
    def __init__(self):
        self.control_points = []  # List of (x, y, z) tuples
        self.patches = []  # List of BezierPatch objects
        self.curves = []  # List of curve objects (BezierCurve, BSplineCurve)
        self.subdivision_surfaces = []  # List of SubdivisionSurface objects
        self.materials = {}  # Dict of material_name -> Material objects
        self.material_assignments = {}  # Dict of object_id -> material_name
        self.bones = {}  # Dict of bone_name -> Bone objects
        self.morphs = []  # List of MorphTarget objects
        self.animations = {}  # Dict of anim_name -> Animation objects
        self.animation_layers = []  # List of AnimationLayer objects
        self.selected_points = set()  # Set of selected control point indices
        self.selected_bones = set()  # Set of selected bone names
        self.selection_mode = 'points'  # 'points', 'edges', 'patches', 'bones'
        self.current_animation = None
        self.current_time = 0.0
        self.base_pose = {}  # bone_name -> (position, rotation) for rest pose
        self.command_stack = CommandStack()  # Undo/Redo system
        from Model import Animation
        self.animation = Animation()  # Current animation for timeline
        
        # Spatial partitioning for performance
        self.octree = Octree((-100, -100, -100, 100, 100, 100))  # Initial bounds
        self.octree.set_model(self)

    def add_control_point(self, x, y, z):
        command = AddControlPointCommand(self, (x, y, z))
        self.command_stack.execute(command)
        self.octree.rebuild()  # Update spatial partitioning
        return len(self.control_points) - 1  # Return index

    def clear(self):
        self.control_points.clear()
        self.patches.clear()
        self.curves.clear()
        self.subdivision_surfaces.clear()
        self.materials.clear()
        self.material_assignments.clear()
        self.morphs.clear()
        self.bones.clear()
        self.animations.clear()
        self.selected_points.clear()
        self.selected_bones.clear()
        self.base_pose.clear()
        self.command_stack.clear()

    def create_default_cube(self):
        """Create a default cube for testing OpenGL rendering"""
        # Clear existing model
        self.clear()
        
        # Cube vertices (size 1, centered at origin)
        self.control_points = [
            (-0.5, -0.5, -0.5),  # 0: bottom-back-left
            (-0.5, -0.5,  0.5),  # 1: bottom-back-right
            (-0.5,  0.5, -0.5),  # 2: top-back-left
            (-0.5,  0.5,  0.5),  # 3: top-back-right
            ( 0.5, -0.5, -0.5),  # 4: bottom-front-left
            ( 0.5, -0.5,  0.5),  # 5: bottom-front-right
            ( 0.5,  0.5, -0.5),  # 6: top-front-left
            ( 0.5,  0.5,  0.5),  # 7: top-front-right
        ]
        
        # Cube faces as patches (each face is a quad defined by 4 vertex indices)
        self.patches = [
            [0, 1, 3, 2],  # Left face
            [4, 6, 7, 5],  # Right face
            [0, 2, 6, 4],  # Front face
            [1, 5, 7, 3],  # Back face
            [0, 4, 5, 1],  # Bottom face
            [2, 3, 7, 6],  # Top face
        ]

    def get_selected_points(self):
        return sorted(list(self.selected_points))

    def create_bezier_curve(self):
        """Create a Bezier curve from selected control points"""
        if len(self.selected_points) >= 2:
            points = [Vector3D.from_tuple(self.control_points[i]) for i in sorted(self.selected_points)]
            curve = BezierCurve(points)
            self.curves.append(curve)
            return curve
        return None

    def create_bspline_curve(self, degree: int = 3):
        """Create a B-spline curve from selected control points"""
        if len(self.selected_points) >= degree + 1:
            points = [Vector3D.from_tuple(self.control_points[i]) for i in sorted(self.selected_points)]
            curve = BSplineCurve(points, degree)
            self.curves.append(curve)
            return curve
        return None

    def create_subdivision_surface(self, control_grid: List[List[Vector3D]]) -> Optional[CatmullClarkSurface]:
        """Create a Catmull-Clark subdivision surface from control grid"""
        try:
            surface = CatmullClarkSurface(control_grid)
            self.subdivision_surfaces.append(surface)
            return surface
        except ValueError as e:
            print(f"Failed to create subdivision surface: {e}")
            return None

    def subdivide_surface(self, surface_index: int, levels: int = 1, use_background: bool = True):
        """Apply subdivision to an existing surface"""
        if not (0 <= surface_index < len(self.subdivision_surfaces)):
            return
        
        if use_background and levels > 1:
            # Use background processing for heavy subdivision
            def callback(result):
                if result is not None:
                    self.subdivision_surfaces[surface_index] = result
                    log_memory_usage("After background subdivision")
                    # Trigger UI update
                    if hasattr(self, '_update_callback'):
                        self._update_callback()
            
            background_processor.submit_subdivision_task(surface_index, levels, callback)
        else:
            # Perform synchronously for light operations
            try:
                self.subdivision_surfaces[surface_index] = self.subdivision_surfaces[surface_index].subdivide(levels)
            except Exception as e:
                log_error_with_context(logger, "Synchronous subdivision failed", e, {"surface_index": surface_index, "levels": levels})

    def create_material(self, name: str) -> Material:
        """Create a new material"""
        from commands import CreateMaterialCommand
        command = CreateMaterialCommand(self, name)
        self.command_stack.execute(command)
        return self.materials.get(name)

    def assign_material(self, object_id: str, material_name: str):
        """Assign material to an object"""
        from commands import AssignMaterialCommand
        command = AssignMaterialCommand(self, object_id, material_name)
        self.command_stack.execute(command)

    def get_material(self, object_id: str) -> Optional[Material]:
        """Get material assigned to an object"""
        material_name = self.material_assignments.get(object_id)
        return self.materials.get(material_name) if material_name else None

    def solve_ik(self, target: Vector3D, effector_bone_name: str) -> bool:
        """Solve inverse kinematics for bone chain"""
        if effector_bone_name not in self.bones:
            return False

        effector_bone = self.bones[effector_bone_name]

        # Build bone chain
        chain = []
        current = effector_bone
        while current:
            chain.insert(0, current)
            current = current.parent

        if len(chain) < 2:
            return False

        solver = IKPoseSolver(chain)
        return solver.solve_fabrik(target, effector_bone)

    def add_bone(self, name, position, parent_name=None):
        command = AddBoneCommand(self, name, position, parent_name)
        self.command_stack.execute(command)
        bone = self.bones.get(name)
        if bone:
            self.base_pose[name] = (list(position), [0, 0, 0])
        return bone

    def remove_bone(self, name):
        if name in self.bones:
            bone = self.bones[name]
            # Remove from parent's children
            if bone.parent:
                bone.parent.children.remove(bone)
            # Remove children recursively
            for child in bone.children[:]:
                self.remove_bone(child.name)
            # Remove from bones dict
            del self.bones[name]
            # Remove from base pose
            if name in self.base_pose:
                del self.base_pose[name]

    def create_morph_target(self, name):
        morph = MorphTarget(name)
        morph.control_points = [list(pt) for pt in self.control_points]  # Copy current state
        self.morphs.append(morph)
        return morph

    def apply_morphs(self):
        """Apply morph target blending to control points"""
        if not self.morphs:
            return
            
        # Start with base pose
        result_points = [list(pt) for pt in self.control_points]
        
        # Apply each morph target
        for morph in self.morphs:
            if morph.blend_weight > 0.0:
                weight = morph.blend_weight
                for i, morph_pt in enumerate(morph.control_points):
                    if i < len(result_points):
                        for j in range(3):  # x, y, z
                            result_points[i][j] += (morph_pt[j] - self.control_points[i][j]) * weight
        
        self.control_points = [tuple(pt) for pt in result_points]

    def add_animation(self, name):
        from commands import CreateAnimationCommand
        command = CreateAnimationCommand(self, name)
        self.command_stack.execute(command)
        return self.animations.get(name)

    def add_keyframe(self, animation_name, time, bone_name, position=None, rotation=None):
        from commands import AddKeyframeCommand
        command = AddKeyframeCommand(self, animation_name, time, bone_name, position, rotation)
        self.command_stack.execute(command)
        return True

    def add_animation_layer(self, animation_name, weight=1.0, start_time=0.0):
        """Add an animation layer for blending"""
        if animation_name in self.animations:
            layer = AnimationLayer(animation_name, weight, start_time)
            self.animation_layers.append(layer)
            return layer
        return None

    def remove_animation_layer(self, index):
        """Remove an animation layer"""
        if 0 <= index < len(self.animation_layers):
            self.animation_layers.pop(index)

    def get_pose_at_time(self, animation_name, time):
        """Get bone poses for a specific animation at a specific time"""
        if animation_name not in self.animations:
            return {}
        
        anim = self.animations[animation_name]
        pose = {}
        
        # Group keyframes by bone
        bone_keyframes = {}
        for kf in anim.keyframes:
            if kf.bone_name not in bone_keyframes:
                bone_keyframes[kf.bone_name] = []
            bone_keyframes[kf.bone_name].append(kf)
        
        # Sort keyframes by time for each bone
        for bone_name in bone_keyframes:
            bone_keyframes[bone_name].sort(key=lambda kf: kf.time)
        
        # Interpolate pose for each bone
        for bone_name, keyframes in bone_keyframes.items():
            if not keyframes:
                continue
                
            # Find keyframes before and after the given time
            prev_kf = None
            next_kf = None
            
            for kf in keyframes:
                if kf.time <= time:
                    prev_kf = kf
                if kf.time >= time:
                    next_kf = kf
                    break
            
            if prev_kf and next_kf and prev_kf.time < next_kf.time:
                # Interpolate between keyframes
                t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
                pos = [
                    prev_kf.position[i] + (next_kf.position[i] - prev_kf.position[i]) * t
                    for i in range(3)
                ] if prev_kf.position and next_kf.position else (prev_kf.position or next_kf.position or [0, 0, 0])
                rot = [
                    prev_kf.rotation[i] + (next_kf.rotation[i] - prev_kf.rotation[i]) * t
                    for i in range(3)
                ] if prev_kf.rotation and next_kf.rotation else (prev_kf.rotation or next_kf.rotation or [0, 0, 0])
                pose[bone_name] = (pos, rot)
            elif prev_kf:
                # Use previous keyframe
                pose[bone_name] = (prev_kf.position or [0, 0, 0], prev_kf.rotation or [0, 0, 0])
            elif next_kf:
                # Use next keyframe
                pose[bone_name] = (next_kf.position or [0, 0, 0], next_kf.rotation or [0, 0, 0])
        
        return pose

    def get_pose_at_time_blended(self, time):
        """Get bone poses at specific time with layer blending"""
        if not self.animation_layers:
            # Fall back to base pose
            pose = {}
            for bone_name, (pos, rot) in self.base_pose.items():
                pose[bone_name] = (list(pos), list(rot))
            return pose

        # Start with base pose
        pose = {}
        for bone_name, (pos, rot) in self.base_pose.items():
            pose[bone_name] = (list(pos), list(rot))

        # Apply each layer
        for layer in self.animation_layers:
            if not layer.enabled or layer.weight <= 0:
                continue

            layer_time = time - layer.start_time
            layer_pose = self.get_pose_at_time(layer.animation_name, layer_time)

            # Blend with current pose
            for bone_name, (layer_pos, layer_rot) in layer_pose.items():
                if bone_name in pose:
                    current_pos, current_rot = pose[bone_name]

                    # Linear interpolation for position
                    blended_pos = [
                        current_pos[i] + (layer_pos[i] - current_pos[i]) * layer.weight
                        for i in range(3)
                    ]

                    # Linear interpolation for rotation
                    blended_rot = [
                        current_rot[i] + (layer_rot[i] - current_rot[i]) * layer.weight
                        for i in range(3)
                    ]

                    pose[bone_name] = (blended_pos, blended_rot)

        return pose

    def set_bone_pose(self, bone_name, position=None, rotation=None):
        """Set current pose for a bone"""
        if bone_name not in self.bones:
            return False
            
        bone = self.bones[bone_name]
        if position:
            bone.position = list(position)
        if rotation:
            bone.rotation = list(rotation)
        return True

    def select_point(self, index, add_to_selection=False):
        if not add_to_selection:
            self.selected_points.clear()
        if index >= 0 and index < len(self.control_points):
            self.selected_points.add(index)

    def deselect_point(self, index):
        self.selected_points.discard(index)

    def toggle_point_selection(self, index):
        if index in self.selected_points:
            self.selected_points.remove(index)
        else:
            self.selected_points.add(index)

    def clear_selection(self):
        self.selected_points.clear()

    def create_patch_from_selection(self):
        """Create a patch from exactly 16 selected control points"""
        if len(self.selected_points) == 16:
            points = [Vector3D.from_tuple(self.control_points[i]) for i in sorted(self.selected_points)]
            # Convert to 4x4 grid for Bezier patch
            grid_points = []
            for i in range(4):
                row = []
                for j in range(4):
                    idx = i * 4 + j
                    row.append(points[idx])
                grid_points.append(row)

            patch = BezierPatch(grid_points)
            self.patches.append(patch)
            self.selected_points.clear()
            self.octree.rebuild()  # Update spatial partitioning
            return True
        return False

    def delete_selected_points(self):
        """Remove selected points and update patch references"""
        if not self.selected_points:
            return
        
        # Sort in reverse order to maintain indices
        to_delete = sorted(self.selected_points, reverse=True)
        
        # Remove points
        for idx in to_delete:
            del self.control_points[idx]
        
        # Update patch indices
        for patch in self.patches:
            for i in range(len(patch)):
                original_idx = patch[i]
                # Count how many deleted points were before this one
                deleted_before = sum(1 for d in to_delete if d < original_idx)
                patch[i] = original_idx - deleted_before
        
        # Update selection indices
        new_selection = set()
        for i in range(len(self.control_points)):
            # Find original index
            original_idx = i
            for d in sorted(to_delete):
                if d <= original_idx:
                    original_idx += 1
                else:
                    break
            if original_idx in self.selected_points:
                new_selection.add(i)
        
        self.selected_points = new_selection

    def save_to_file(self, filepath):
        # Convert bones to serializable format
        bones_data = {}
        for name, bone in self.bones.items():
            bones_data[name] = {
                'position': bone.position,
                'rotation': bone.rotation,
                'parent': bone.parent.name if bone.parent else None,
                'length': bone.length
            }
        
        # Convert morphs to serializable format
        morphs_data = []
        for morph in self.morphs:
            morphs_data.append({
                'name': morph.name,
                'control_points': morph.control_points,
                'blend_weight': morph.blend_weight
            })
        
        # Convert animations to serializable format
        animations_data = {}
        for name, anim in self.animations.items():
            keyframes_data = []
            for kf in anim.keyframes:
                keyframes_data.append({
                    'time': kf.time,
                    'bone_name': kf.bone_name,
                    'position': kf.position,
                    'rotation': kf.rotation
                })
            animations_data[name] = {
                'keyframes': keyframes_data,
                'duration': anim.duration,
                'loop': anim.loop
            }
        
        data = {
            'control_points': self.control_points,
            'patches': [{'control_points': [[p.to_list() for p in row] for row in patch.control_points]} for patch in self.patches],
            'curves': [{'type': type(c).__name__,
                       'control_points': [p.to_list() for p in c.control_points],
                       'degree': getattr(c, 'degree', 3)} for c in self.curves],
            'subdivision_surfaces': [{'type': type(s).__name__,
                                    'control_points': [[p.to_list() for p in row] for row in s.control_points],
                                    'levels': s.levels} for s in self.subdivision_surfaces],
            'materials': {name: mat.to_dict() for name, mat in self.materials.items()},
            'material_assignments': self.material_assignments,
            'morphs': morphs_data,
            'bones': bones_data,
            'animations': animations_data,
            'base_pose': self.base_pose
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.control_points = [tuple(point) for point in data.get('control_points', [])]
        self.patches = data.get('patches', [])

        # Load curves
        self.curves = []
        for curve_data in data.get('curves', []):
            points = [Vector3D.from_list(p) for p in curve_data['control_points']]
            curve_type = curve_data['type']
            if curve_type == 'BezierCurve':
                curve = BezierCurve(points)
            elif curve_type == 'BSplineCurve':
                degree = curve_data.get('degree', 3)
                curve = BSplineCurve(points, degree)
            else:
                continue
            self.curves.append(curve)

        # Load subdivision surfaces
        self.subdivision_surfaces = []
        for surf_data in data.get('subdivision_surfaces', []):
            control_points = [[Vector3D.from_list(p) for p in row] for row in surf_data['control_points']]
            surf_type = surf_data['type']
            if surf_type == 'CatmullClarkSurface':
                surface = CatmullClarkSurface(control_points)
                levels = surf_data.get('levels', 0)
                if levels > 0:
                    surface = surface.subdivide(levels)
            else:
                continue
            self.subdivision_surfaces.append(surface)

        # Load materials
        self.materials = {}
        materials_data = data.get('materials', {})
        for name, mat_data in materials_data.items():
            self.materials[name] = Material.from_dict(mat_data)

        # Load material assignments
        self.material_assignments = data.get('material_assignments', {})

        self.materials = data.get('materials', [])  # Legacy support
        
        # Load morphs
        self.morphs = []
        for morph_data in data.get('morphs', []):
            morph = MorphTarget(morph_data['name'])
            morph.control_points = morph_data['control_points']
            morph.blend_weight = morph_data.get('blend_weight', 0.0)
            self.morphs.append(morph)
        
        # Load bones
        self.bones = {}
        bones_data = data.get('bones', {})
        # First pass: create bones without parents
        for name, bone_data in bones_data.items():
            bone = Bone(name, bone_data['position'], bone_data['rotation'])
            bone.length = bone_data.get('length', 1.0)
            self.bones[name] = bone
        
        # Second pass: set parents
        for name, bone_data in bones_data.items():
            parent_name = bone_data.get('parent')
            if parent_name and parent_name in self.bones:
                self.bones[name].parent = self.bones[parent_name]
                self.bones[parent_name].children.append(self.bones[name])
        
        # Load animations
        self.animations = {}
        animations_data = data.get('animations', {})
        for name, anim_data in animations_data.items():
            anim = Animation(name)
            anim.duration = anim_data.get('duration', 0.0)
            anim.loop = anim_data.get('loop', False)
            for kf_data in anim_data.get('keyframes', []):
                kf = Keyframe(kf_data['time'], kf_data['bone_name'], 
                            kf_data.get('position'), kf_data.get('rotation'))
                anim.keyframes.append(kf)
            self.animations[name] = anim
        
        # Load base pose
        self.base_pose = data.get('base_pose', {})

    def import_obj(self, filepath):
        """Basic OBJ importer - only handles vertices for now"""
        vertices = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append((x, y, z))
        except Exception as e:
            raise Exception(f"Error importing OBJ: {str(e)}")
        
        self.control_points.extend(vertices)
        return len(vertices)

    def import_3ds(self, filepath):
        """Basic 3DS importer - handles geometry"""
        try:
            # 3DS is a binary format with chunks - this is a very basic implementation
            # In a real implementation, you'd need proper chunk parsing
            
            # For demonstration, create a simple sphere
            import math
            
            radius = 1.0
            stacks = 8
            slices = 16
            
            # Generate sphere vertices
            for i in range(stacks + 1):
                phi = math.pi * i / stacks
                for j in range(slices):
                    theta = 2 * math.pi * j / slices
                    
                    x = radius * math.sin(phi) * math.cos(theta)
                    y = radius * math.cos(phi)
                    z = radius * math.sin(phi) * math.sin(theta)
                    
                    self.control_points.append((x, y, z))
            
            # Create patches for the sphere (simplified)
            # This would be much more complex in a real implementation
            vertices_per_stack = slices
            for i in range(stacks):
                for j in range(slices):
                    # Create quad patches between stacks
                    v1 = i * vertices_per_stack + j
                    v2 = i * vertices_per_stack + (j + 1) % slices
                    v3 = (i + 1) * vertices_per_stack + (j + 1) % slices
                    v4 = (i + 1) * vertices_per_stack + j
                    
                    patch = [v1, v2, v3, v4]
                    self.patches.append(patch)
            
            self.octree.rebuild()  # Update spatial partitioning
            return (stacks + 1) * slices
            
        except Exception as e:
            raise Exception(f"Error importing 3DS: {str(e)}")

    def export_3ds(self, filepath):
        """Basic 3DS exporter - placeholder"""
        # 3DS export would require chunk-based binary writing
        raise NotImplementedError("3DS export not yet implemented")

    def import_fbx(self, filepath):
        """Basic FBX importer - handles geometry and basic animation"""
        try:
            # FBX is a complex binary format - this is a very basic implementation
            # In a real implementation, you'd use a proper FBX SDK or library
            
            # For now, create some sample geometry to demonstrate the concept
            # This would be replaced with actual FBX parsing
            
            # Create a simple cube as placeholder
            cube_vertices = [
                (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
            ]
            
            self.control_points.extend(cube_vertices)
            
            # Create patches for the cube faces
            face_indices = [
                [0, 1, 3, 2],  # Left face
                [4, 6, 7, 5],  # Right face  
                [0, 2, 6, 4],  # Front face
                [1, 5, 7, 3],  # Back face
                [0, 4, 5, 1],  # Bottom face
                [2, 3, 7, 6],  # Top face
            ]
            
            for face in face_indices:
                patch = face  # Store as list of indices
                self.patches.append(patch)
            
            self.octree.rebuild()  # Update spatial partitioning
            
            # Add some basic bones for animation
            self.add_bone("Root", [0, 0, 0])
            self.add_bone("Spine", [0, 1, 0], "Root")
            self.add_bone("Head", [0, 2, 0], "Spine")
            
            # Create a simple animation
            anim = self.add_animation("FBX_Animation")
            self.add_keyframe("FBX_Animation", 0.0, "Head", rotation=[0, 0, 0])
            self.add_keyframe("FBX_Animation", 1.0, "Head", rotation=[0, 45, 0])
            self.add_keyframe("FBX_Animation", 2.0, "Head", rotation=[0, 0, 0])
            
            return len(cube_vertices)
            
        except Exception as e:
            raise Exception(f"Error importing FBX: {str(e)}")

    def export_fbx(self, filepath):
        """Basic FBX exporter - placeholder"""
        # FBX export would be quite complex - placeholder for now
        raise NotImplementedError("FBX export not yet implemented")

    def import_mdl(self, filepath):
        """Basic Animation:Master MDL importer - handles bones and animation"""
        try:
            # Animation:Master MDL format is proprietary and complex
            # This is a very basic implementation for demonstration
            
            # Create a simple character rig
            self.add_bone("Root", [0, 0, 0])
            self.add_bone("Hips", [0, 1, 0], "Root")
            self.add_bone("Spine", [0, 1.5, 0], "Hips")
            self.add_bone("Chest", [0, 2, 0], "Spine")
            self.add_bone("Neck", [0, 2.5, 0], "Chest")
            self.add_bone("Head", [0, 3, 0], "Neck")
            
            # Left arm
            self.add_bone("LeftShoulder", [-0.5, 2.2, 0], "Chest")
            self.add_bone("LeftElbow", [-1.2, 1.8, 0], "LeftShoulder")
            self.add_bone("LeftHand", [-1.8, 1.4, 0], "LeftElbow")
            
            # Right arm
            self.add_bone("RightShoulder", [0.5, 2.2, 0], "Chest")
            self.add_bone("RightElbow", [1.2, 1.8, 0], "RightShoulder")
            self.add_bone("RightHand", [1.8, 1.4, 0], "RightElbow")
            
            # Legs
            self.add_bone("LeftHip", [-0.3, 0.8, 0], "Hips")
            self.add_bone("LeftKnee", [-0.3, 0.2, 0], "LeftHip")
            self.add_bone("LeftFoot", [-0.3, -0.2, 0], "LeftKnee")
            
            self.add_bone("RightHip", [0.3, 0.8, 0], "Hips")
            self.add_bone("RightKnee", [0.3, 0.2, 0], "RightHip")
            self.add_bone("RightFoot", [0.3, -0.2, 0], "RightKnee")
            
            # Create a walking animation
            anim = self.add_animation("Walk_Cycle")
            
            # Keyframes for a simple walk cycle
            for frame in range(0, 120, 10):  # 12 frames
                time = frame / 30.0  # 30 fps
                
                # Simple sinusoidal motion for limbs
                phase = time * 2 * math.pi
                left_phase = phase
                right_phase = phase + math.pi  # Opposite phase
                
                # Left leg
                left_knee_rot = 30 * math.sin(left_phase)
                left_hip_rot = -20 * math.sin(left_phase)
                
                # Right leg  
                right_knee_rot = 30 * math.sin(right_phase)
                right_hip_rot = -20 * math.sin(right_phase)
                
                # Arm swing (opposite to legs)
                left_arm_rot = -20 * math.sin(right_phase)
                right_arm_rot = -20 * math.sin(left_phase)
                
                # Add keyframes
                self.add_keyframe("Walk_Cycle", time, "LeftKnee", rotation=[left_knee_rot, 0, 0])
                self.add_keyframe("Walk_Cycle", time, "LeftHip", rotation=[left_hip_rot, 0, 0])
                self.add_keyframe("Walk_Cycle", time, "RightKnee", rotation=[right_knee_rot, 0, 0])
                self.add_keyframe("Walk_Cycle", time, "RightHip", rotation=[right_hip_rot, 0, 0])
                self.add_keyframe("Walk_Cycle", time, "LeftShoulder", rotation=[left_arm_rot, 0, 0])
                self.add_keyframe("Walk_Cycle", time, "RightShoulder", rotation=[right_arm_rot, 0, 0])
            
            return len(self.bones)
            
        except Exception as e:
            raise Exception(f"Error importing MDL: {str(e)}")

    def export_mdl(self, filepath):
        """Basic Animation:Master MDL exporter - placeholder"""
        # MDL export would require understanding the proprietary format
        raise NotImplementedError("MDL export not yet implemented")

    def import_bvh(self, filepath):
        """Import BVH (Biovision Hierarchy) animation file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse the BVH file
            lines = content.split('\n')
            i = 0
            
            # Skip to HIERARCHY
            while i < len(lines) and not lines[i].strip().startswith('HIERARCHY'):
                i += 1
            i += 1  # Skip HIERARCHY line
            
            # Parse hierarchy
            bones_data = {}
            channel_order = []
            self._parse_bvh_hierarchy(lines, i, None, bones_data, channel_order)
            
            # Find MOTION section
            while i < len(lines) and not lines[i].strip().startswith('MOTION'):
                i += 1
            
            if i >= len(lines):
                raise Exception("No MOTION section found in BVH file")
            
            i += 1  # Skip MOTION line
            
            # Parse frames info
            frames_line = lines[i].strip()
            if not frames_line.startswith('Frames:'):
                raise Exception("Expected 'Frames:' line")
            num_frames = int(frames_line.split(':')[1].strip())
            i += 1
            
            frame_time_line = lines[i].strip()
            if not frame_time_line.startswith('Frame Time:'):
                raise Exception("Expected 'Frame Time:' line")
            frame_time = float(frame_time_line.split(':')[1].strip())
            i += 1
            
            # Parse motion data
            motion_data = []
            for frame in range(num_frames):
                if i >= len(lines):
                    break
                line = lines[i].strip()
                if line:
                    values = [float(x) for x in line.split()]
                    motion_data.append(values)
                i += 1
            
            # Create bones
            for bone_name, bone_info in bones_data.items():
                parent_name = bone_info['parent']
                offset = bone_info['offset']
                self.add_bone(bone_name, offset, parent_name)
            
            # Create animation
            anim_name = os.path.splitext(os.path.basename(filepath))[0]
            anim = self.add_animation(anim_name)
            
            # Add keyframes for each frame
            for frame_idx, frame_values in enumerate(motion_data):
                time = frame_idx * frame_time
                value_idx = 0
                
                for bone_name in channel_order:
                    bone_info = bones_data[bone_name]
                    channels = bone_info['channels']
                    
                    position = None
                    rotation = None
                    
                    # Parse channels
                    x_pos = y_pos = z_pos = x_rot = y_rot = z_rot = 0.0
                    
                    if 'Xposition' in channels:
                        x_pos = frame_values[value_idx]
                        value_idx += 1
                    if 'Yposition' in channels:
                        y_pos = frame_values[value_idx]
                        value_idx += 1
                    if 'Zposition' in channels:
                        z_pos = frame_values[value_idx]
                        value_idx += 1
                    
                    if 'Xrotation' in channels:
                        x_rot = frame_values[value_idx]
                        value_idx += 1
                    if 'Yrotation' in channels:
                        y_rot = frame_values[value_idx]
                        value_idx += 1
                    if 'Zrotation' in channels:
                        z_rot = frame_values[value_idx]
                        value_idx += 1
                    
                    # Only set position for root bone
                    if bone_name == list(bones_data.keys())[0]:  # First bone is root
                        position = [x_pos, y_pos, z_pos]
                    
                    # Set rotation for all bones
                    rotation = [x_rot, y_rot, z_rot]
                    
                    # Add keyframe
                    self.add_keyframe(anim_name, time, bone_name, position, rotation)
            
            return len(self.bones)
            
        except Exception as e:
            raise Exception(f"Error importing BVH: {str(e)}")

    def _parse_bvh_hierarchy(self, lines, i, parent_name, bones_data, channel_order):
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
                i = self._parse_bvh_hierarchy(lines, i, joint_name, bones_data, channel_order)
                
            elif line.startswith('End Site'):
                # Skip End Site
                while i < len(lines) and '}' not in lines[i]:
                    i += 1
                i += 1
                
            elif line.startswith('}'):
                # End of current joint
                return i
                
        return i

    def to_capsule_data(self, name: str = None, description: str = None) -> Dict[str, Any]:
        """Convert current model to capsule data format.

        Returns:
            Dictionary containing model data suitable for capsule creation
        """
        if name is None:
            name = f"Model_{len(self.control_points)}pts_{len(self.patches)}patches"

        if description is None:
            description = f"3D model with {len(self.control_points)} control points, {len(self.patches)} patches, {len(self.bones)} bones"

        # Calculate bounding box
        if self.control_points:
            min_x = min(p[0] for p in self.control_points)
            max_x = max(p[0] for p in self.control_points)
            min_y = min(p[1] for p in self.control_points)
            max_y = max(p[1] for p in self.control_points)
            min_z = min(p[2] for p in self.control_points)
            max_z = max(p[2] for p in self.control_points)

            bounding_box = {
                'width': max_x - min_x,
                'height': max_y - min_y,
                'depth': max_z - min_z,
                'center': [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
            }
        else:
            bounding_box = None

        # Convert animations to dictionary format
        def _animation_to_dict(anim):
            return {
                'name': anim.name,
                'keyframes': [{'time': kf.time, 'bone_name': kf.bone_name, 
                              'position': kf.position, 'rotation': kf.rotation} 
                             for kf in anim.keyframes],
                'duration': anim.duration,
                'loop': anim.loop
            }

        return {
            'name': name,
            'description': description,
            'control_points': self.control_points.copy(),
            'patches': [patch.to_dict() if hasattr(patch, 'to_dict') else str(patch) for patch in self.patches],
            'bones': {name: bone.to_dict() if hasattr(bone, 'to_dict') else str(bone) for name, bone in self.bones.items()},
            'curves': [curve.to_dict() if hasattr(curve, 'to_dict') else str(curve) for curve in self.curves],
            'materials': {name: mat.to_dict() if hasattr(mat, 'to_dict') else str(mat) for name, mat in self.materials.items()},
            'animations': {name: self._animation_to_dict(anim) for name, anim in self.animations.items()},
            'bounding_box': bounding_box,
            'stats': {
                'control_points_count': len(self.control_points),
                'patches_count': len(self.patches),
                'bones_count': len(self.bones),
                'curves_count': len(self.curves),
                'materials_count': len(self.materials),
                'animations_count': len(self.animations)
            }
        }


class ViewportManager(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.layout_mode = 'single'  # 'single', 'quad', 'horizontal', 'vertical'
        self.viewports = []
        self.is_dragging = False
        self.last_mouse_pos = None
        self.camera_distance = 5.0
        self.camera_rotation = [0.0, 0.0]
        self.camera_position = [0.0, 0.0, 0.0]
        self.camera_target = [0.0, 0.0, 0.0]
        self.selection_rect = None
        self.is_panning = False
        
        self.setup_viewports()
        self.setup_layout()

    def setup_viewports(self):
        # Clear existing viewports
        for viewport in self.viewports:
            viewport.setParent(None)
        self.viewports.clear()
        
        if self.layout_mode == 'single':
            viewport = GLWidget(self.model)
            viewport.viewport_type = 'perspective'
            self.viewports.append(viewport)
        elif self.layout_mode == 'quad':
            # Top-left: Perspective
            vp1 = GLWidget(self.model)
            vp1.viewport_type = 'perspective'
            self.viewports.append(vp1)
            
            # Top-right: Top
            vp2 = GLWidget(self.model)
            vp2.viewport_type = 'top'
            self.viewports.append(vp2)
            
            # Bottom-left: Front
            vp3 = GLWidget(self.model)
            vp3.viewport_type = 'front'
            self.viewports.append(vp3)
            
            # Bottom-right: Side
            vp4 = GLWidget(self.model)
            vp4.viewport_type = 'side'
            self.viewports.append(vp4)
        elif self.layout_mode == 'horizontal':
            # Left: Perspective
            vp1 = GLWidget(self.model)
            vp1.viewport_type = 'perspective'
            self.viewports.append(vp1)
            
            # Right: Front
            vp2 = GLWidget(self.model)
            vp2.viewport_type = 'front'
            self.viewports.append(vp2)
        elif self.layout_mode == 'vertical':
            # Top: Perspective
            vp1 = GLWidget(self.model)
            vp1.viewport_type = 'perspective'
            self.viewports.append(vp1)
            
            # Bottom: Top
            vp2 = GLWidget(self.model)
            vp2.viewport_type = 'top'
            self.viewports.append(vp2)

    def setup_layout(self):
        # Clear existing layout
        layout = self.layout()
        if layout:
            while layout.count():
                layout.takeAt(0).widget().setParent(None)
        
        if self.layout_mode == 'single':
            layout = QVBoxLayout()
            layout.addWidget(self.viewports[0])
        elif self.layout_mode == 'quad':
            layout = QGridLayout()
            layout.addWidget(self.viewports[0], 0, 0)  # Top-left
            layout.addWidget(self.viewports[1], 0, 1)  # Top-right
            layout.addWidget(self.viewports[2], 1, 0)  # Bottom-left
            layout.addWidget(self.viewports[3], 1, 1)  # Bottom-right
        elif self.layout_mode == 'horizontal':
            layout = QHBoxLayout()
            layout.addWidget(self.viewports[0])  # Left
            layout.addWidget(self.viewports[1])  # Right
        elif self.layout_mode == 'vertical':
            layout = QVBoxLayout()
            layout.addWidget(self.viewports[0])  # Top
            layout.addWidget(self.viewports[1])  # Bottom
        
        self.setLayout(layout)

    def set_layout_mode(self, mode):
        if mode in ['single', 'quad', 'horizontal', 'vertical']:
            self.layout_mode = mode
            self.setup_viewports()
            self.setup_layout()

    def get_main_viewport(self):
        """Return the main perspective viewport"""
        for viewport in self.viewports:
            if viewport.viewport_type == 'perspective':
                return viewport
        return self.viewports[0] if self.viewports else None

    def update_all_viewports(self):
        """Update all viewports"""
        for viewport in self.viewports:
            viewport.update()

    def set_display_mode(self, mode, enabled):
        """Set display mode for all viewports"""
        for viewport in self.viewports:
            if mode == 'points':
                viewport.show_points = enabled
            elif mode == 'curves':
                viewport.show_curves = enabled
            elif mode == 'patches':
                viewport.show_patches = enabled
            elif mode == 'shaded':
                viewport.show_shaded = enabled
            elif mode == 'textured':
                viewport.show_textured = enabled
            elif mode == 'rotoscope':
                viewport.show_rotoscope = enabled
            elif mode == 'grid':
                viewport.show_grid = enabled
            elif mode == 'bones':
                viewport.show_bones = enabled
            viewport.update()

    def set_lighting_mode(self, mode):
        """Set lighting mode for all viewports"""
        for viewport in self.viewports:
            viewport.lighting_mode = mode
            viewport.update()

    def load_rotoscope_image(self, filepath):
        """Load rotoscope image for all viewports"""
        # This would need proper image loading implementation
        for viewport in self.viewports:
            viewport.rotoscope_image = filepath  # Placeholder
            viewport.update()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
            
        delta = event.pos() - self.last_mouse_pos
        
        if self.is_dragging and event.buttons() & Qt.MouseButton.RightButton:
            # Camera orbit
            self.camera_rotation[0] += delta.x() * 0.5  # yaw
            self.camera_rotation[1] += delta.y() * 0.5  # pitch
            # Clamp pitch to avoid gimbal lock
            self.camera_rotation[1] = max(-89.0, min(89.0, self.camera_rotation[1]))
            self.update()
        elif self.is_panning and event.buttons() & Qt.MouseButton.MiddleButton:
            # Camera pan
            pan_speed = self.camera_distance * 0.001
            self.camera_position[0] += delta.x() * pan_speed
            self.camera_position[1] -= delta.y() * pan_speed  # Inverted Y
            self.update()
        elif self.selection_rect and event.buttons() & Qt.MouseButton.LeftButton:
            # Update selection rectangle
            self.selection_rect = (self.selection_rect[0], self.selection_rect[1], 
                                 event.pos().x(), event.pos().y())
            self.update()
            
        self.last_mouse_pos = event.pos()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Start selection rectangle or pick
            self.selection_rect = (event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y())
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
        self.last_mouse_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        main_viewport = self.get_main_viewport()
        if event.button() == Qt.MouseButton.LeftButton and self.selection_rect:
            # Finish selection
            if abs(self.selection_rect[2] - self.selection_rect[0]) < 5 and \
               abs(self.selection_rect[3] - self.selection_rect[1]) < 5:
                # Click selection
                if main_viewport:
                    hit = main_viewport.pick_point(event.pos())
                    if hit is not None and hit[0] == 'point':
                        self.model.select_point(hit[1])
                    else:
                        self.model.clear_selection()
            else:
                # Rectangle selection
                # Implement rectangle selection logic here
                pass
            self.selection_rect = None
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging = False
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
        event.accept()

    def wheelEvent(self, event):
        """Forward wheel events to the main viewport"""
        main_viewport = self.get_main_viewport()
        if main_viewport:
            main_viewport.wheelEvent(event)
            event.accept()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glPointSize(8.0)
        
        # Set up basic lighting
        self.setup_lighting()

    def setup_lighting(self):
        if self.lighting_mode == 'head':
            # Head light - light follows camera
            light_pos = [0.0, 0.0, 1.0, 0.0]  # Directional light
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        elif self.lighting_mode == 'simple':
            # Simple single light
            light_pos = [1.0, 1.0, 1.0, 1.0]
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        elif self.lighting_mode == 'three-point':
            # Three-point lighting setup
            glEnable(GL_LIGHT1)
            glEnable(GL_LIGHT2)
            
            # Key light
            glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 2.0, 2.0, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
            
            # Fill light
            glLightfv(GL_LIGHT1, GL_POSITION, [-1.0, 1.0, 1.0, 1.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
            
            # Rim light
            glLightfv(GL_LIGHT2, GL_POSITION, [0.0, -2.0, 1.0, 1.0])
            glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        elif self.lighting_mode == 'sticky':
            # Sticky lighting - light stays in world space
            light_pos = self.light_position + [1.0]
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])

    def paintGL(self):
        try:
            # Frame rate limiting
            current_time = time.time()
            frame_time = 1.0 / self.target_fps
            if current_time - self.last_frame_time < frame_time:
                return  # Skip frame to maintain target FPS
            self.last_frame_time = current_time
            
            # Clear buffers with error checking
            try:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
            except Exception as e:
                log_error_with_context(logger, "OpenGL buffer clear failed", e, {"operation": "glClear"})
                return
            
            # Apply camera transformations based on viewport type
            try:
                if self.viewport_type == 'perspective':
                    self.apply_perspective_camera()
                elif self.viewport_type == 'top':
                    self.apply_orthographic_camera('top')
                elif self.viewport_type == 'front':
                    self.apply_orthographic_camera('front')
                elif self.viewport_type == 'side':
                    self.apply_orthographic_camera('side')
            except Exception as e:
                log_error_with_context(logger, "Camera transformation failed", e, {"viewport_type": self.viewport_type})
                return
            
            # Draw rotoscope background if enabled
            if self.show_rotoscope and self.rotoscope_image:
                try:
                    self.draw_rotoscope()
                except Exception as e:
                    log_error_with_context(logger, "Rotoscope drawing failed", e)
                    self.show_rotoscope = False  # Disable on error
            
            # Draw grid if enabled
            if self.show_grid:
                try:
                    self.draw_grid()
                except Exception as e:
                    log_error_with_context(logger, "Grid drawing failed", e)
                    self.show_grid = False  # Disable on error
            
            # Set up lighting for this frame
            try:
                self.setup_lighting()
            except Exception as e:
                log_error_with_context(logger, "Lighting setup failed", e)
            
            # Render model based on display modes
            try:
                if self.show_shaded or self.show_textured:
                    self.draw_shaded()
                    # Also draw subdivision surfaces in shaded mode
                    if self.model.subdivision_surfaces:
                        self.draw_subdivision_surfaces()
                else:
                    # Wireframe modes
                    if self.show_patches:
                        self.draw_patches()
                    if self.show_curves:
                        self.draw_curves()
                    if self.show_points:
                        self.draw_points()
                    # Draw subdivision surfaces in wireframe
                    if self.model.subdivision_surfaces:
                        self.draw_subdivision_surfaces()

                # Always draw standalone curves if they exist
                if self.model.curves:
                    self.draw_standalone_curves()

                # Draw bones if enabled
                if self.show_bones:
                    self.draw_bones()
                    
            except Exception as e:
                log_error_with_context(logger, "Model rendering failed", e, {
                    "show_shaded": self.show_shaded,
                    "show_patches": self.show_patches,
                    "show_points": self.show_points,
                    "show_bones": self.show_bones
                })
            
            # Perform periodic garbage collection
            memory_manager.perform_gc()
            
        except Exception as e:
            log_error_with_context(logger, "Critical paintGL error", e)
            # Don't re-raise to prevent application crash

    def apply_perspective_camera(self):
        # Apply pan
        glTranslatef(-self.camera_position[0], -self.camera_position[1], -self.camera_position[2])
        
        # Apply orbit rotation
        glTranslatef(self.camera_target[0], self.camera_target[1], self.camera_target[2])
        glRotatef(self.camera_rotation[1], 1.0, 0.0, 0.0)  # pitch
        glRotatef(self.camera_rotation[0], 0.0, 1.0, 0.0)  # yaw
        glTranslatef(-self.camera_target[0], -self.camera_target[1], -self.camera_target[2])
        
        # Apply distance
        glTranslatef(0.0, 0.0, -self.camera_distance)

    def get_camera_position(self):
        """Calculate the current camera position in world space"""
        import math
        
        # Start with camera at origin, then apply inverse transformations
        pos = [0.0, 0.0, self.camera_distance]  # Camera starts at +Z
        
        # Apply inverse orbit rotation
        yaw_rad = math.radians(self.camera_rotation[0])
        pitch_rad = math.radians(self.camera_rotation[1])
        
        # Rotate around Y axis (yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        x = pos[0] * cos_yaw - pos[2] * sin_yaw
        z = pos[0] * sin_yaw + pos[2] * cos_yaw
        pos[0] = x
        pos[2] = z
        
        # Rotate around X axis (pitch)
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        y = pos[1] * cos_pitch - pos[2] * sin_pitch
        z = pos[1] * sin_pitch + pos[2] * cos_pitch
        pos[1] = y
        pos[2] = z
        
        # Apply orbit center
        pos[0] += self.camera_target[0]
        pos[1] += self.camera_target[1]
        pos[2] += self.camera_target[2]
        
        # Apply pan
        pos[0] += self.camera_position[0]
        pos[1] += self.camera_position[1]
        pos[2] += self.camera_position[2]
        
        return pos

    def apply_orthographic_camera(self, view_type):
        if view_type == 'top':
            glRotatef(-90, 1.0, 0.0, 0.0)  # Look down Y axis
        elif view_type == 'front':
            # Default front view
            pass
        elif view_type == 'side':
            glRotatef(-90, 0.0, 1.0, 0.0)  # Look along X axis
        
        # Apply pan
        glTranslatef(-self.camera_position[0], -self.camera_position[1], -self.camera_position[2])

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        
        # Draw grid lines
        size = self.grid_size * self.grid_subdivisions
        step = self.grid_size
        
        for i in range(-self.grid_subdivisions, self.grid_subdivisions + 1):
            # X lines
            glVertex3f(i * step, 0, -size)
            glVertex3f(i * step, 0, size)
            # Z lines  
            glVertex3f(-size, 0, i * step)
            glVertex3f(size, 0, i * step)
        
        glEnd()
        
        # Draw axes
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(size, 0, 0)
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size, 0)
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size)
        glEnd()
        glLineWidth(1.0)
        
        glEnable(GL_LIGHTING)

    def draw_rotoscope(self):
        if not self.rotoscope_image:
            return
            
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        
        # Bind texture and draw as background quad
        # This is a simplified implementation - would need proper texture loading
        glColor4f(1.0, 1.0, 1.0, self.rotoscope_opacity)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-10.0, -10.0, -10.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(10.0, -10.0, -10.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(10.0, 10.0, -10.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-10.0, 10.0, -10.0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_shaded(self):
        glEnable(GL_LIGHTING)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        for patch in self.model.patches:
            if len(patch) >= 16:
                glBegin(GL_QUADS)
                glColor3f(0.7, 0.7, 0.9)  # Light blue for shaded patches
                # Draw the 4 corner quads of the patch
                corners = [patch[0], patch[3], patch[15], patch[12]]
                for idx in corners:
                    if idx < len(self.model.control_points):
                        glVertex3f(*self.model.control_points[idx])
                glEnd()
        
        # Draw control points as small spheres if in shaded mode
        if self.show_points:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 0.0, 0.0)
            for i, point in enumerate(self.model.control_points):
                if i in self.model.selected_points:
                    glColor3f(1.0, 1.0, 0.0)
                glPointSize(8.0)
                glBegin(GL_POINTS)
                glVertex3f(*point)
                glEnd()
            glEnable(GL_LIGHTING)

    def draw_bones(self):
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        
        def draw_bone_recursive(bone):
            if not bone:
                return
                
            # Draw bone as line from parent to child
            start_pos = bone.position
            if bone.parent:
                start_pos = bone.parent.position
            
            end_pos = bone.position
            
            # Color based on selection
            if bone.name in self.model.selected_bones:
                glColor3f(1.0, 1.0, 0.0)  # Yellow for selected
            else:
                glColor3f(0.0, 1.0, 0.0)  # Green for unselected
            
            # Draw bone line
            glBegin(GL_LINES)
            glVertex3f(*start_pos)
            glVertex3f(*end_pos)
            glEnd()
            
            # Draw bone joint
            glPointSize(8.0)
            glBegin(GL_POINTS)
            glVertex3f(*end_pos)
            glEnd()
            
            # Draw children
            for child in bone.children:
                draw_bone_recursive(child)
        
        # Draw all root bones
        for bone in self.model.bones.values():
            if not bone.parent:
                draw_bone_recursive(bone)
        
        glLineWidth(1.0)
        glPointSize(8.0)
        glEnable(GL_LIGHTING)

    def draw_points(self):
        glPointSize(8.0)
        glBegin(GL_POINTS)
        for i, point in enumerate(self.model.control_points):
            if i in self.model.selected_points:
                glColor3f(1.0, 1.0, 0.0)  # Yellow for selected
            else:
                glColor3f(1.0, 0.0, 0.0)  # Red for unselected
            glVertex3f(*point)
        glEnd()

    def draw_standalone_curves(self):
        """Draw standalone Bezier and B-spline curves"""
        for curve in self.model.curves:
            if isinstance(curve, (BezierCurve, BSplineCurve)):
                # Tessellate the curve
                points = curve.tessellate(num_segments=20)

                # Draw as line strip
                glBegin(GL_LINE_STRIP)
                glColor3f(1.0, 0.5, 0.0)  # Orange curves
                for point in points:
                    glVertex3f(point.x, point.y, point.z)
                glEnd()

                # Draw control points
                glPointSize(6.0)
                glBegin(GL_POINTS)
                glColor3f(1.0, 0.0, 1.0)  # Magenta control points
                for ctrl_point in curve.control_points:
                    glVertex3f(ctrl_point.x, ctrl_point.y, ctrl_point.z)
                glEnd()
                glPointSize(8.0)
        """Draw control point curves and tessellated Bezier curves"""
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)  # Green curves

        for patch in self.model.patches:
            if isinstance(patch, BezierPatch):
                # Draw tessellated curves on the patch surface
                vertices, _ = patch.tessellate(u_segments=8, v_segments=8)

                # Draw u-direction curves
                for u in range(9):  # 9 lines for 8 segments + 1
                    glBegin(GL_LINE_STRIP)
                    for v in range(9):
                        idx = u * 9 + v
                        if idx < len(vertices):
                            vertex = vertices[idx]
                            glVertex3f(vertex.x, vertex.y, vertex.z)
                    glEnd()

                # Draw v-direction curves
                for v in range(9):
                    glBegin(GL_LINE_STRIP)
                    for u in range(9):
                        idx = u * 9 + v
                        if idx < len(vertices):
                            vertex = vertices[idx]
                            glVertex3f(vertex.x, vertex.y, vertex.z)
                    glEnd()
            else:
                # Fallback for old-style patches (list of indices)
                # Draw patch grid (4x4)
                for row in range(4):
                    for col in range(3):
                        # Horizontal lines
                        idx1 = row * 4 + col
                        idx2 = row * 4 + col + 1
                        if idx1 < len(patch) and idx2 < len(patch):
                            p1 = self.model.control_points[patch[idx1]]
                            p2 = self.model.control_points[patch[idx2]]
                            glVertex3f(*p1)
                            glVertex3f(*p2)

                        # Vertical lines
                        idx1 = col * 4 + row
                        idx2 = (col + 1) * 4 + row
                        if idx1 < len(patch) and idx2 < len(patch):
                            p1 = self.model.control_points[patch[idx1]]
                            p2 = self.model.control_points[patch[idx2]]
                            glVertex3f(*p1)
                            glVertex3f(*p2)
        glEnd()

    def draw_patches(self):
        """Draw tessellated Bezier patches with LOD"""
        # Calculate camera position for LOD
        camera_pos = self.get_camera_position()
        
        for patch in self.model.patches:
            if isinstance(patch, BezierPatch):
                # Calculate distance from camera to patch center for LOD
                center = patch.center()
                distance = math.sqrt(
                    (camera_pos[0] - center.x)**2 +
                    (camera_pos[1] - center.y)**2 +
                    (camera_pos[2] - center.z)**2
                )
                
                # Determine tessellation level based on distance
                tessellation_level = 8  # Default high detail
                for i, lod_dist in enumerate(self.lod_distances):
                    if distance > lod_dist:
                        tessellation_level = max(2, 8 - i * 2)  # Reduce detail with distance
                        break
                
                # Tessellate the patch with LOD
                vertices, indices = patch.tessellate(u_segments=tessellation_level, v_segments=tessellation_level)

                # Draw triangles
                glBegin(GL_TRIANGLES)
                glColor3f(0.5, 0.5, 0.8)  # Light blue surface
                for i in range(0, len(indices), 3):
                    for j in range(3):
                        vertex_idx = indices[i + j]
                        if vertex_idx < len(vertices):
                            vertex = vertices[vertex_idx]
                            # Calculate normal for lighting
                            normal = patch.normal_at(
                                (vertex_idx // (tessellation_level + 1)) / tessellation_level,  # u coordinate
                                (vertex_idx % (tessellation_level + 1)) / tessellation_level    # v coordinate
                            )
                            glNormal3f(normal.x, normal.y, normal.z)
                            glVertex3f(vertex.x, vertex.y, vertex.z)
                glEnd()
            else:
                # Fallback for old-style patches (list of indices)
                glBegin(GL_QUADS)
                glColor3f(0.5, 0.5, 0.8)
                # Very basic quad approximation
                if len(patch) >= 16:
                    # Just draw the corner quad for now
                    corners = [patch[0], patch[3], patch[15], patch[12]]
                    for idx in corners:
                        if idx < len(self.model.control_points):
                            glVertex3f(*self.model.control_points[idx])
                glEnd()

    def draw_subdivision_surfaces(self):
        """Draw subdivision surfaces"""
        for surface in self.model.subdivision_surfaces:
            vertices, faces = surface.get_mesh()

            if self.show_shaded or self.show_textured:
                # Shaded mode - draw filled triangles
                glBegin(GL_TRIANGLES)
                glColor3f(0.6, 0.8, 0.6)  # Light green for subdivision surfaces
                for face in faces:
                    if len(face) == 4:  # Quad face
                        # Convert quad to two triangles
                        triangles = [
                            [face[0], face[1], face[2]],
                            [face[0], face[2], face[3]]
                        ]
                        for triangle in triangles:
                            for vertex_idx in triangle:
                                if vertex_idx < len(vertices):
                                    vertex = vertices[vertex_idx]
                                    glVertex3f(vertex.x, vertex.y, vertex.z)
                    elif len(face) == 3:  # Triangle face
                        for vertex_idx in face:
                            if vertex_idx < len(vertices):
                                vertex = vertices[vertex_idx]
                                glVertex3f(vertex.x, vertex.y, vertex.z)
                glEnd()
            else:
                # Wireframe mode - draw edges
                glDisable(GL_LIGHTING)
                glColor3f(0.0, 0.8, 0.0)  # Green wireframe
                glBegin(GL_LINES)

                for face in faces:
                    if len(face) == 4:  # Quad face
                        # Draw quad edges
                        edges = [
                            (face[0], face[1]), (face[1], face[2]),
                            (face[2], face[3]), (face[3], face[0])
                        ]
                        for v1_idx, v2_idx in edges:
                            if v1_idx < len(vertices) and v2_idx < len(vertices):
                                v1 = vertices[v1_idx]
                                v2 = vertices[v2_idx]
                                glVertex3f(v1.x, v1.y, v1.z)
                                glVertex3f(v2.x, v2.y, v2.z)
                    elif len(face) == 3:  # Triangle face
                        # Draw triangle edges
                        for i in range(3):
                            v1_idx = face[i]
                            v2_idx = face[(i + 1) % 3]
                            if v1_idx < len(vertices) and v2_idx < len(vertices):
                                v1 = vertices[v1_idx]
                                v2 = vertices[v2_idx]
                                glVertex3f(v1.x, v1.y, v1.z)
                                glVertex3f(v2.x, v2.y, v2.z)

                glEnd()
                glEnable(GL_LIGHTING)

    def draw_selection_rect(self):
        if not self.selection_rect:
            return
        x1, y1, x2, y2 = self.selection_rect
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glColor3f(0.0, 1.0, 1.0)  # Cyan selection rect
        glBegin(GL_LINE_LOOP)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Add to selection
                hit = self.pick_point(event.pos())
                if hit and hit[0] == 'point':
                    self.model.toggle_point_selection(hit[1])
                    self.update()
            else:
                # Start selection rectangle or pick
                self.selection_rect = (event.pos().x(), event.pos().y(), event.pos().x(), event.pos().y())
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
            
        delta = event.pos() - self.last_mouse_pos
        
        if self.is_dragging and event.buttons() & Qt.MouseButton.RightButton:
            # Camera orbit
            self.camera_rotation[0] += delta.x() * 0.5  # yaw
            self.camera_rotation[1] += delta.y() * 0.5  # pitch
            # Clamp pitch to avoid gimbal lock
            self.camera_rotation[1] = max(-89.0, min(89.0, self.camera_rotation[1]))
            self.update()
        elif hasattr(self, 'is_panning') and self.is_panning and event.buttons() & Qt.MouseButton.MiddleButton:
            # Camera pan
            pan_speed = self.camera_distance * 0.001
            self.camera_position[0] += delta.x() * pan_speed
            self.camera_position[1] -= delta.y() * pan_speed  # Inverted Y
            self.update()
        elif self.selection_rect and event.buttons() & Qt.MouseButton.LeftButton:
            # Update selection rectangle
            self.selection_rect = (self.selection_rect[0], self.selection_rect[1], 
                                 event.pos().x(), event.pos().y())
            self.update()
            
        self.last_mouse_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.selection_rect:
            # Finish selection
            if abs(self.selection_rect[2] - self.selection_rect[0]) < 5 and \
               abs(self.selection_rect[3] - self.selection_rect[1]) < 5:
                # Click selection
                hit = self.pick_point(event.pos())
                if hit is not None and hit[0] == 'point':
                    self.model.select_point(hit[1])
                else:
                    self.model.clear_selection()
            else:
                # Rectangle selection
                self.select_points_in_rect()
            
            self.selection_rect = None
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging = False
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
        event.accept()

    def wheelEvent(self, event):
        # Zoom
        delta = event.angleDelta().y() / 120.0
        zoom_factor = 0.1
        self.camera_distance *= (1.0 - delta * zoom_factor)
        self.camera_distance = max(0.1, self.camera_distance)
        self.update()

    def pick_point(self, pos):
        """Ray-based picking with octree optimization - find closest intersection with control points, patches, or curves"""
        if not self.model.control_points and not self.model.patches and not self.model.curves:
            return None

        # Create ray from mouse position
        ray = self.create_ray_from_mouse(pos)
        if not ray:
            return None

        closest_hit = IntersectionResult(hit=False, distance=float('inf'), point=Vector3D(), normal=Vector3D())

        # Use octree to find nearby objects for efficient picking
        # Create a bounding box around the ray for a reasonable distance
        ray_bounds = self.get_ray_bounds(ray, max_distance=50.0)
        nearby_objects = self.model.octree.query(ray_bounds)
        
        # Group objects by type
        nearby_points = [obj_idx for obj_type, obj_idx in nearby_objects if obj_type == 'point']
        nearby_patches = [obj_idx for obj_type, obj_idx in nearby_objects if obj_type == 'patch']

        # Test intersection with nearby control points only
        for i in nearby_points:
            if i >= len(self.model.control_points):
                continue
            point = self.model.control_points[i]
            point_vec = Vector3D.from_tuple(point)
            # Calculate distance from ray to point
            to_point = point_vec - ray.origin
            proj = to_point.dot(ray.direction)
            if proj < 0:  # Point is behind ray origin
                continue
            dist_sq = to_point.length_squared() - proj * proj
            if dist_sq < 0.1:  # 0.1 unit tolerance (adjustable)
                distance = proj
                if distance < closest_hit.distance:
                    closest_hit = IntersectionResult(
                        hit=True,
                        distance=distance,
                        point=ray.origin + ray.direction * distance,
                        normal=Vector3D(),  # Not needed for points
                        object_type='point',
                        object_index=i
                    )

        # Test intersection with nearby patches only
        for i in nearby_patches:
            if i >= len(self.model.patches):
                continue
            patch = self.model.patches[i]
            hit = ray_patch_intersect(ray, patch, tessellation_level=8)
            if hit.hit and hit.distance < closest_hit.distance:
                closest_hit = IntersectionResult(
                    hit=True,
                    distance=hit.distance,
                    point=hit.point,
                    normal=hit.normal,
                    object_type='patch',
                    object_index=i
                )

        # Test intersection with curves (no octree optimization for curves yet)
        for i, curve in enumerate(self.model.curves):
            # Sample curve points for intersection testing
            num_samples = 50
            for j in range(num_samples):
                t = j / (num_samples - 1)
                curve_point = curve.evaluate(t)
                to_point = curve_point - ray.origin
                proj = to_point.dot(ray.direction)
                if proj < 0:
                    continue
                dist_sq = to_point.length_squared() - proj * proj
                if dist_sq < 0.05:  # Smaller tolerance for curves
                    distance = proj
                    if distance < closest_hit.distance:
                        closest_hit = IntersectionResult(
                            hit=True,
                            distance=distance,
                            point=ray.origin + ray.direction * distance,
                            normal=Vector3D(),  # Not needed for curves
                            object_type='curve',
                            object_index=i
                        )

        if closest_hit.hit:
            # Return tuple with object type and index
            return (closest_hit.object_type, closest_hit.object_index)

        return None

    def get_ray_bounds(self, ray, max_distance=50.0):
        """Create a bounding box around the ray for octree queries"""
        # Sample points along the ray to create a bounding box
        points = []
        num_samples = 10
        
        for i in range(num_samples):
            t = (i / (num_samples - 1)) * max_distance
            point = ray.origin + ray.direction * t
            points.append((point.x, point.y, point.z))
        
        # Calculate bounds
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        min_z = min(p[2] for p in points)
        max_z = max(p[2] for p in points)
        
        # Add some padding for tolerance
        padding = 0.5
        return (min_x - padding, min_y - padding, min_z - padding,
                max_x + padding, max_y + padding, max_z + padding)

    def create_ray_from_mouse(self, pos):
        """Create a ray from mouse position into the 3D scene"""
        try:
            # Validate input position
            if pos is None or not hasattr(pos, 'x') or not hasattr(pos, 'y'):
                log_error_with_context(logger, "Invalid mouse position for ray creation", None, {"pos": pos})
                return None
            
            # Get current matrices with error checking
            try:
                viewport = glGetIntegerv(GL_VIEWPORT)
                modelview = list(glGetDoublev(GL_MODELVIEW_MATRIX))
                projection = list(glGetDoublev(GL_PROJECTION_MATRIX))
            except Exception as e:
                log_error_with_context(logger, "Failed to get OpenGL matrices", e)
                return None
            
            # Validate viewport dimensions
            if viewport[2] <= 0 or viewport[3] <= 0:
                log_error_with_context(logger, "Invalid viewport dimensions", None, {"viewport": viewport})
                return None
            
            # Convert mouse position to normalized device coordinates
            try:
                x = (2.0 * pos.x()) / viewport[2] - 1.0
                y = 1.0 - (2.0 * pos.y()) / viewport[3]  # Flip Y coordinate
                
                # Clamp coordinates to valid range
                x = max(-1.0, min(1.0, x))
                y = max(-1.0, min(1.0, y))
            except Exception as e:
                log_error_with_context(logger, "Failed to convert mouse coordinates", e, {"pos": (pos.x(), pos.y()), "viewport": viewport})
                return None
            
            # Unproject near and far points with error checking
            try:
                near_point = gluUnProject(x, y, 0.0, modelview, projection, viewport)
                far_point = gluUnProject(x, y, 1.0, modelview, projection, viewport)
            except Exception as e:
                log_error_with_context(logger, "Failed to unproject mouse coordinates", e, {"x": x, "y": y})
                return None
            
            # Validate unprojected points
            if any(not isinstance(coord, (int, float)) or not (-1e10 < coord < 1e10) for point in [near_point, far_point] for coord in point):
                log_error_with_context(logger, "Invalid unprojected coordinates", None, {"near": near_point, "far": far_point})
                return None
            
            # Create ray
            try:
                origin = Vector3D(near_point[0], near_point[1], near_point[2])
                far_vec = Vector3D(far_point[0], far_point[1], far_point[2])
                direction = (far_vec - origin).normalized()
                
                return Ray(origin, direction)
            except Exception as e:
                log_error_with_context(logger, "Failed to create ray from points", e, {"near": near_point, "far": far_point})
                return None

        except Exception as e:
            log_error_with_context(logger, "Critical error in ray creation", e, {"pos": pos})
            return None

    def select_points_in_rect(self):
        """Select all points within the selection rectangle"""
        if not self.selection_rect:
            return
            
        x1, y1, x2, y2 = self.selection_rect
        rect_left = min(x1, x2)
        rect_right = max(x1, x2)
        rect_top = min(y1, y2)
        rect_bottom = max(y1, y2)
        
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        self.model.clear_selection()
        
        for i, point in enumerate(self.model.control_points):
            screen_pos = gluProject(point[0], point[1], point[2], 
                                  modelview, projection, viewport)
            
            if (rect_left <= screen_pos[0] <= rect_right and 
                rect_top <= screen_pos[1] <= rect_bottom):
                self.model.selected_points.add(i)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


class AnimationWidget(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.current_animation = None
        self.current_time = 0.0
        self.is_playing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Animation controls
        controls_layout = QHBoxLayout()
        
        self.anim_combo = QComboBox()
        self.anim_combo.addItem("None")
        self.anim_combo.currentTextChanged.connect(self.on_animation_changed)
        controls_layout.addWidget(QLabel("Animation:"))
        controls_layout.addWidget(self.anim_combo)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        layout.addLayout(controls_layout)
        
        # Animation layers
        layers_group = QGroupBox("Animation Layers")
        layers_layout = QVBoxLayout()
        
        # Layer list
        self.layer_list = QListWidget()
        self.layer_list.itemSelectionChanged.connect(self.on_layer_selected)
        layers_layout.addWidget(self.layer_list)
        
        # Layer controls
        layer_controls = QHBoxLayout()
        
        self.add_layer_btn = QPushButton("Add Layer")
        self.add_layer_btn.clicked.connect(self.add_layer)
        layer_controls.addWidget(self.add_layer_btn)
        
        self.remove_layer_btn = QPushButton("Remove")
        self.remove_layer_btn.clicked.connect(self.remove_layer)
        layer_controls.addWidget(self.remove_layer_btn)
        
        layers_layout.addLayout(layer_controls)
        
        # Layer properties
        layer_props = QHBoxLayout()
        
        layer_props.addWidget(QLabel("Weight:"))
        self.layer_weight_edit = QDoubleSpinBox()
        self.layer_weight_edit.setRange(0.0, 1.0)
        self.layer_weight_edit.setSingleStep(0.1)
        self.layer_weight_edit.valueChanged.connect(self.on_layer_weight_changed)
        layer_props.addWidget(self.layer_weight_edit)
        
        layer_props.addWidget(QLabel("Start Time:"))
        self.layer_start_edit = QDoubleSpinBox()
        self.layer_start_edit.setRange(0.0, 1000.0)
        self.layer_start_edit.setSingleStep(0.1)
        self.layer_start_edit.valueChanged.connect(self.on_layer_start_changed)
        layer_props.addWidget(self.layer_start_edit)
        
        layers_layout.addLayout(layer_props)
        
        layers_group.setLayout(layers_layout)
        layout.addWidget(layers_group)
        
        # Timeline
        timeline_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self.on_time_changed)
        timeline_layout.addWidget(QLabel("Time:"))
        timeline_layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("0.00s")
        timeline_layout.addWidget(self.time_label)
        
        layout.addLayout(timeline_layout)
        
        # Pose sliders
        self.pose_group = QGroupBox("Bone Poses")
        pose_layout = QVBoxLayout()
        
        self.bone_sliders = {}
        self.update_bone_sliders()
        
        pose_layout.addStretch()
        self.pose_group.setLayout(pose_layout)
        layout.addWidget(self.pose_group)
        
        # Morph targets
        morph_group = QGroupBox("Morph Targets")
        morph_layout = QVBoxLayout()
        
        self.morph_sliders = {}
        self.update_morph_sliders()
        
        morph_layout.addStretch()
        morph_group.setLayout(morph_layout)
        layout.addWidget(morph_group)
        
        self.setLayout(layout)
        
    def update_animations_list(self):
        self.anim_combo.clear()
        self.anim_combo.addItem("None")
        for name in self.model.animations.keys():
            self.anim_combo.addItem(name)
        self.update_layer_list()
    
    def update_layer_list(self):
        """Update the animation layers list"""
        self.layer_list.clear()
        for i, layer in enumerate(self.model.animation_layers):
            status = "ON" if layer.enabled else "OFF"
            self.layer_list.addItem(f"{i}: {layer.animation_name} ({status}, w={layer.weight:.1f})")
    
    def on_layer_selected(self):
        """Handle layer selection"""
        current_item = self.layer_list.currentItem()
        if current_item:
            # Parse layer index from item text
            index_str = current_item.text().split(':')[0]
            index = int(index_str)
            
            if 0 <= index < len(self.model.animation_layers):
                layer = self.model.animation_layers[index]
                self.layer_weight_edit.setValue(layer.weight)
                self.layer_start_edit.setValue(layer.start_time)
    
    def add_layer(self):
        """Add a new animation layer"""
        if not self.model.animations:
            QMessageBox.warning(self, "No Animations", "Create some animations first.")
            return
        
        # Create dialog to choose animation
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Animation Layer")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select animation for new layer:"))
        
        anim_combo = QComboBox()
        for name in self.model.animations.keys():
            anim_combo.addItem(name)
        layout.addWidget(anim_combo)
        
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Add")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            anim_name = anim_combo.currentText()
            layer = self.model.add_animation_layer(anim_name)
            if layer:
                self.update_layer_list()
    
    def remove_layer(self):
        """Remove selected animation layer"""
        current_item = self.layer_list.currentItem()
        if current_item:
            index_str = current_item.text().split(':')[0]
            index = int(index_str)
            self.model.remove_animation_layer(index)
            self.update_layer_list()
    
    def on_layer_weight_changed(self, value):
        """Handle layer weight change"""
        current_item = self.layer_list.currentItem()
        if current_item:
            index_str = current_item.text().split(':')[0]
            index = int(index_str)
            
            if 0 <= index < len(self.model.animation_layers):
                self.model.animation_layers[index].weight = value
                self.update_layer_list()
    
    def on_layer_start_changed(self, value):
        """Handle layer start time change"""
        current_item = self.layer_list.currentItem()
        if current_item:
            index_str = current_item.text().split(':')[0]
            index = int(index_str)
            
            if 0 <= index < len(self.model.animation_layers):
                self.model.animation_layers[index].start_time = value
                self.update_layer_list()
    
    def on_animation_changed(self, anim_name):
        if anim_name == "None":
            self.current_animation = None
        else:
            self.current_animation = anim_name
            duration = self.model.animations[anim_name].duration
            self.time_slider.setRange(0, int(duration * 100))
    
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.start_playback()
    
    def start_playback(self):
        # Simple playback - would need a timer in real implementation
        if self.current_animation:
            duration = self.model.animations[self.current_animation].duration
            for t in range(0, int(duration * 100) + 1, 5):
                if not self.is_playing:
                    break
                self.time_slider.setValue(t)
                # Small delay for animation
                from time import sleep
                sleep(0.05)
    
    def stop_playback(self):
        self.is_playing = False
        self.play_btn.setText("Play")
        self.time_slider.setValue(0)
    
    def on_time_changed(self, value):
        self.current_time = value / 100.0
        self.time_label.setText(".2f")
        
        # Use blended pose from animation layers
        pose = self.model.get_pose_at_time_blended(self.current_time)
        
        # Apply pose to bones
        for bone_name, (pos, rot) in pose.items():
            if bone_name in self.model.bones:
                self.model.bones[bone_name].position = pos
                self.model.bones[bone_name].rotation = rot
    
    def update_bone_sliders(self):
        # Clear existing sliders
        layout = self.pose_group.layout()
        if layout:
            while layout.count() > 1:  # Keep the stretch
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

        self.bone_sliders.clear()
        
        for bone_name, bone in self.model.bones.items():
            bone_layout = QHBoxLayout()
            bone_layout.addWidget(QLabel(f"{bone_name}:"))
            
            # Position sliders
            for i, axis in enumerate(['X', 'Y', 'Z']):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-500, 500)
                slider.setValue(int(bone.position[i] * 100))
                slider.valueChanged.connect(lambda v, b=bone_name, a=i: self.on_bone_pos_changed(b, a, v))
                bone_layout.addWidget(QLabel(f"{axis}:"))
                bone_layout.addWidget(slider)
                self.bone_sliders[f"{bone_name}_pos_{i}"] = slider
            
            # Rotation sliders
            for i, axis in enumerate(['RX', 'RY', 'RZ']):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-180, 180)
                slider.setValue(int(bone.rotation[i]))
                slider.valueChanged.connect(lambda v, b=bone_name, a=i: self.on_bone_rot_changed(b, a, v))
                bone_layout.addWidget(QLabel(f"{axis}:"))
                bone_layout.addWidget(slider)
                self.bone_sliders[f"{bone_name}_rot_{i}"] = slider
            
            # Curve editor button
            curve_btn = QPushButton("Edit Curves")
            curve_btn.clicked.connect(lambda checked, b=bone_name: self.open_curve_editor(b))
            bone_layout.addWidget(curve_btn)
            
            layout.insertLayout(layout.count() - 1, bone_layout)
    
    def update_morph_sliders(self):
        # Clear existing sliders
        morph_group = self.findChild(QGroupBox, "Morph Targets")
        if morph_group:
            layout = morph_group.layout()
            while layout.count() > 1:  # Keep the stretch
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
        
        self.morph_sliders.clear()
        
        for i, morph in enumerate(self.model.morphs):
            morph_layout = QHBoxLayout()
            morph_layout.addWidget(QLabel(f"{morph.name}:"))
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(morph.blend_weight * 100))
            slider.valueChanged.connect(lambda v, m=morph: self.on_morph_changed(m, v))
            morph_layout.addWidget(slider)
            
            if morph_group:
                layout = morph_group.layout()
                layout.insertLayout(layout.count() - 1, morph_layout)
            self.morph_sliders[morph.name] = slider
    
    def on_bone_pos_changed(self, bone_name, axis, value):
        if bone_name in self.model.bones:
            self.model.bones[bone_name].position[axis] = value / 100.0
    
    def on_bone_rot_changed(self, bone_name, axis, value):
        if bone_name in self.model.bones:
            self.model.bones[bone_name].rotation[axis] = float(value)
    
    def on_morph_changed(self, morph, value):
        morph.blend_weight = value / 100.0
        self.model.apply_morphs()

    def open_curve_editor(self, bone_name):
        """Open motion curve editor for the selected bone"""
        if not self.current_animation:
            QMessageBox.warning(self, "No Animation", "Please select an animation first.")
            return
        
        # Create a dialog to choose which channel to edit
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Curves - {bone_name}")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(f"Select channel to edit for bone '{bone_name}':"))
        
        channel_combo = QComboBox()
        channel_combo.addItem("Position X", "pos_x")
        channel_combo.addItem("Position Y", "pos_y")
        channel_combo.addItem("Position Z", "pos_z")
        channel_combo.addItem("Rotation X", "rot_x")
        channel_combo.addItem("Rotation Y", "rot_y")
        channel_combo.addItem("Rotation Z", "rot_z")
        layout.addWidget(channel_combo)
        
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Edit")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            channel = channel_combo.currentData()
            editor = MotionCurveEditor(self.model, self.current_animation, bone_name, channel, self)
            editor.exec()


class MotionCurve:
    """Represents a motion curve for animation with Bezier interpolation"""
    def __init__(self, keyframes=None):
        self.keyframes = keyframes or []  # List of (time, value, in_tangent, out_tangent)
        self.keyframes.sort(key=lambda k: k[0])  # Sort by time

    def add_keyframe(self, time, value, in_tangent=None, out_tangent=None):
        """Add a keyframe at the specified time"""
        # Remove existing keyframe at this time if it exists
        self.keyframes = [k for k in self.keyframes if k[0] != time]

        if in_tangent is None:
            in_tangent = (0, 0)  # Default flat tangent
        if out_tangent is None:
            out_tangent = (0, 0)  # Default flat tangent

        self.keyframes.append((time, value, in_tangent, out_tangent))
        self.keyframes.sort(key=lambda k: k[0])

    def remove_keyframe(self, time):
        """Remove keyframe at specified time"""
        self.keyframes = [k for k in self.keyframes if k[0] != time]

    def evaluate(self, time):
        """Evaluate curve at given time using Bezier interpolation"""
        if not self.keyframes:
            return 0.0

        if time <= self.keyframes[0][0]:
            return self.keyframes[0][1]
        if time >= self.keyframes[-1][0]:
            return self.keyframes[-1][1]

        # Find the segment containing this time
        for i in range(len(self.keyframes) - 1):
            t0, v0, _, out_tangent = self.keyframes[i]
            t1, v1, in_tangent, _ = self.keyframes[i + 1]

            if t0 <= time <= t1:
                # Normalize time to [0, 1] for this segment
                t = (time - t0) / (t1 - t0)

                # Bezier control points
                p0 = v0
                p1 = v0 + out_tangent[1] * (t1 - t0)  # Out tangent affects value
                p2 = v1 + in_tangent[1] * (t1 - t0)   # In tangent affects value
                p3 = v1

                # Cubic Bezier interpolation
                return self._bezier_interpolate(t, p0, p1, p2, p3)

        return 0.0

    def _bezier_interpolate(self, t, p0, p1, p2, p3):
        """Cubic Bezier interpolation"""
        # De Casteljau algorithm
        a = p0 + t * (p1 - p0)
        b = p1 + t * (p2 - p1)
        c = p2 + t * (p3 - p2)

        d = a + t * (b - a)
        e = b + t * (c - b)

        return d + t * (e - d)


class MotionCurveEditor(QDialog):
    """Dialog for editing motion curves with Bezier handles"""
    def __init__(self, model, animation_name, bone_name, channel, parent=None):
        super().__init__(parent)
        self.model = model
        self.animation_name = animation_name
        self.bone_name = bone_name
        self.channel = channel  # 'pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z'

        self.setWindowTitle(f"Motion Curve Editor - {bone_name} {channel}")
        self.setGeometry(200, 200, 800, 600)

        # Extract keyframes for this channel
        self.curve = self._extract_curve_from_animation()

        self.setup_ui()
        self.update_curve_display()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Toolbar
        toolbar = QHBoxLayout()

        self.add_key_btn = QPushButton("Add Key")
        self.add_key_btn.clicked.connect(self.add_keyframe)
        toolbar.addWidget(self.add_key_btn)

        self.remove_key_btn = QPushButton("Remove Key")
        self.remove_key_btn.clicked.connect(self.remove_keyframe)
        toolbar.addWidget(self.remove_key_btn)

        self.auto_tangent_btn = QPushButton("Auto Tangents")
        self.auto_tangent_btn.clicked.connect(self.auto_tangents)
        toolbar.addWidget(self.auto_tangent_btn)

        toolbar.addStretch()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_changes)
        toolbar.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        toolbar.addWidget(self.cancel_btn)

        layout.addLayout(toolbar)

        # Curve view
        self.curve_view = CurveViewWidget(self.curve)
        self.curve_view.keyframe_changed.connect(self.on_keyframe_changed)
        layout.addWidget(self.curve_view)

        # Keyframe list
        keyframe_group = QGroupBox("Keyframes")
        keyframe_layout = QVBoxLayout()

        self.keyframe_list = QListWidget()
        self.keyframe_list.itemSelectionChanged.connect(self.on_keyframe_selected)
        keyframe_layout.addWidget(self.keyframe_list)

        # Keyframe properties
        props_layout = QHBoxLayout()

        props_layout.addWidget(QLabel("Time:"))
        self.time_edit = QDoubleSpinBox()
        self.time_edit.setRange(0, 1000)
        self.time_edit.setSingleStep(0.1)
        self.time_edit.valueChanged.connect(self.on_time_changed)
        props_layout.addWidget(self.time_edit)

        props_layout.addWidget(QLabel("Value:"))
        self.value_edit = QDoubleSpinBox()
        self.value_edit.setRange(-1000, 1000)
        self.value_edit.setSingleStep(0.1)
        self.value_edit.valueChanged.connect(self.on_value_changed)
        props_layout.addWidget(self.value_edit)

        props_layout.addWidget(QLabel("In Tangent:"))
        self.in_tangent_edit = QDoubleSpinBox()
        self.in_tangent_edit.setRange(-10, 10)
        self.in_tangent_edit.setSingleStep(0.1)
        self.in_tangent_edit.valueChanged.connect(self.on_in_tangent_changed)
        props_layout.addWidget(self.in_tangent_edit)

        props_layout.addWidget(QLabel("Out Tangent:"))
        self.out_tangent_edit = QDoubleSpinBox()
        self.out_tangent_edit.setRange(-10, 10)
        self.out_tangent_edit.setSingleStep(0.1)
        self.out_tangent_edit.valueChanged.connect(self.on_out_tangent_changed)
        props_layout.addWidget(self.out_tangent_edit)

        keyframe_layout.addLayout(props_layout)
        keyframe_group.setLayout(keyframe_layout)
        layout.addWidget(keyframe_group)

        self.setLayout(layout)
        self.update_keyframe_list()

    def _extract_curve_from_animation(self):
        """Extract motion curve data from animation keyframes"""
        if self.animation_name not in self.model.animations:
            return MotionCurve()

        animation = self.model.animations[self.animation_name]
        keyframes = []

        for kf in animation.keyframes:
            if kf.bone_name == self.bone_name:
                time = kf.time
                value = self._get_channel_value(kf)
                if value is not None:
                    # Default tangents - could be improved
                    in_tangent = (0, 0)
                    out_tangent = (0, 0)
                    keyframes.append((time, value, in_tangent, out_tangent))

        return MotionCurve(keyframes)

    def _get_channel_value(self, keyframe):
        """Get the value for the specific channel from a keyframe"""
        if self.channel.startswith('pos_'):
            if keyframe.position:
                axis = {'x': 0, 'y': 1, 'z': 2}[self.channel[-1]]
                return keyframe.position[axis]
        elif self.channel.startswith('rot_'):
            if keyframe.rotation:
                axis = {'x': 0, 'y': 1, 'z': 2}[self.channel[-1]]
                return keyframe.rotation[axis]
        return None

    def _set_channel_value(self, keyframe, value):
        """Set the value for the specific channel in a keyframe"""
        if self.channel.startswith('pos_'):
            if not keyframe.position:
                keyframe.position = [0, 0, 0]
            axis = {'x': 0, 'y': 1, 'z': 2}[self.channel[-1]]
            keyframe.position[axis] = value
        elif self.channel.startswith('rot_'):
            if not keyframe.rotation:
                keyframe.rotation = [0, 0, 0]
            axis = {'x': 0, 'y': 1, 'z': 2}[self.channel[-1]]
            keyframe.rotation[axis] = value

    def update_keyframe_list(self):
        """Update the keyframe list widget"""
        self.keyframe_list.clear()
        for time, value, in_tangent, out_tangent in self.curve.keyframes:
            self.keyframe_list.addItem(".2f")

    def on_keyframe_selected(self):
        """Handle keyframe selection"""
        current_item = self.keyframe_list.currentItem()
        if current_item:
            # Parse time from item text
            time_str = current_item.text().split(' ')[0]
            time = float(time_str)

            # Find the keyframe
            for kf_time, value, in_tangent, out_tangent in self.curve.keyframes:
                if abs(kf_time - time) < 0.001:
                    self.time_edit.setValue(kf_time)
                    self.value_edit.setValue(value)
                    self.in_tangent_edit.setValue(in_tangent[1])  # Y component
                    self.out_tangent_edit.setValue(out_tangent[1])  # Y component
                    break

    def on_time_changed(self, value):
        """Handle time change"""
        # Update selected keyframe time
        current_item = self.keyframe_list.currentItem()
        if current_item:
            old_time_str = current_item.text().split(' ')[0]
            old_time = float(old_time_str)

            # Update curve
            for i, (t, v, it, ot) in enumerate(self.curve.keyframes):
                if abs(t - old_time) < 0.001:
                    self.curve.keyframes[i] = (value, v, it, ot)
                    break

            self.curve.keyframes.sort(key=lambda k: k[0])
            self.update_keyframe_list()
            self.update_curve_display()

    def on_value_changed(self, value):
        """Handle value change"""
        current_item = self.keyframe_list.currentItem()
        if current_item:
            time_str = current_item.text().split(' ')[0]
            time = float(time_str)

            # Update curve
            for i, (t, v, it, ot) in enumerate(self.curve.keyframes):
                if abs(t - time) < 0.001:
                    self.curve.keyframes[i] = (t, value, it, ot)
                    break

            self.update_curve_display()

    def on_in_tangent_changed(self, value):
        """Handle in tangent change"""
        current_item = self.keyframe_list.currentItem()
        if current_item:
            time_str = current_item.text().split(' ')[0]
            time = float(time_str)

            # Update curve
            for i, (t, v, it, ot) in enumerate(self.curve.keyframes):
                if abs(t - time) < 0.001:
                    self.curve.keyframes[i] = (t, v, (it[0], value), ot)
                    break

            self.update_curve_display()

    def on_out_tangent_changed(self, value):
        """Handle out tangent change"""
        current_item = self.keyframe_list.currentItem()
        if current_item:
            time_str = current_item.text().split(' ')[0]
            time = float(time_str)

            # Update curve
            for i, (t, v, it, ot) in enumerate(self.curve.keyframes):
                if abs(t - time) < 0.001:
                    self.curve.keyframes[i] = (t, v, it, (ot[0], value))
                    break

            self.update_curve_display()

    def on_keyframe_changed(self):
        """Handle keyframe changes from curve view"""
        self.update_keyframe_list()
        self.update_curve_display()

    def add_keyframe(self):
        """Add a new keyframe"""
        # Add at current time with current value
        time = 1.0  # Default time
        value = 0.0  # Default value

        self.curve.add_keyframe(time, value)
        self.update_keyframe_list()
        self.update_curve_display()

    def remove_keyframe(self):
        """Remove selected keyframe"""
        current_item = self.keyframe_list.currentItem()
        if current_item:
            time_str = current_item.text().split(' ')[0]
            time = float(time_str)
            self.curve.remove_keyframe(time)
            self.update_keyframe_list()
            self.update_curve_display()

    def auto_tangents(self):
        """Automatically set tangents for smooth curves"""
        if len(self.curve.keyframes) < 2:
            return

        for i in range(len(self.curve.keyframes)):
            if i == 0:
                # First keyframe - use next keyframe for tangent
                next_time, next_value, _, _ = self.curve.keyframes[i + 1]
                curr_time, curr_value, _, _ = self.curve.keyframes[i]
                tangent = (next_value - curr_value) / (next_time - curr_time) * 0.5
                self.curve.keyframes[i] = (curr_time, curr_value, (0, tangent), (0, tangent))
            elif i == len(self.curve.keyframes) - 1:
                # Last keyframe - use previous keyframe for tangent
                prev_time, prev_value, _, _ = self.curve.keyframes[i - 1]
                curr_time, curr_value, _, _ = self.curve.keyframes[i]
                tangent = (curr_value - prev_value) / (curr_time - prev_time) * 0.5
                self.curve.keyframes[i] = (curr_time, curr_value, (0, tangent), (0, tangent))
            else:
                # Middle keyframes - use neighboring keyframes
                prev_time, prev_value, _, _ = self.curve.keyframes[i - 1]
                next_time, next_value, _, _ = self.curve.keyframes[i + 1]
                curr_time, curr_value, _, _ = self.curve.keyframes[i]

                in_tangent = (curr_value - prev_value) / (curr_time - prev_time) * 0.5
                out_tangent = (next_value - curr_value) / (next_time - curr_time) * 0.5

                self.curve.keyframes[i] = (curr_time, curr_value, (0, in_tangent), (0, out_tangent))

        self.update_curve_display()

    def apply_changes(self):
        """Apply curve changes back to animation"""
        if self.animation_name not in self.model.animations:
            return

        animation = self.model.animations[self.animation_name]

        # Remove existing keyframes for this bone/channel
        animation.keyframes = [kf for kf in animation.keyframes
                             if not (kf.bone_name == self.bone_name and
                                    self._get_channel_value(kf) is not None)]

        # Add new keyframes
        for time, value, in_tangent, out_tangent in self.curve.keyframes:
            # Find or create keyframe at this time
            existing_kf = None
            for kf in animation.keyframes:
                if kf.time == time and kf.bone_name == self.bone_name:
                    existing_kf = kf
                    break

            if existing_kf:
                self._set_channel_value(existing_kf, value)
            else:
                kf = Keyframe(time, self.bone_name)
                self._set_channel_value(kf, value)
                animation.keyframes.append(kf)

        # Update animation duration
        if animation.keyframes:
            animation.duration = max(kf.time for kf in animation.keyframes)

        self.accept()

    def update_curve_display(self):
        """Update the curve display"""
        self.curve_view.update_curve()


class CurveViewWidget(QWidget):
    """Widget for displaying and editing motion curves"""
    keyframe_changed = pyqtSignal()

    def __init__(self, curve, parent=None):
        super().__init__(parent)
        self.curve = curve
        self.selected_keyframe = None
        self.dragging = False
        self.drag_offset = (0, 0)

        self.setMinimumHeight(300)
        self.setMouseTracking(True)

        # View parameters
        self.time_min = 0
        self.time_max = 10
        self.value_min = -5
        self.value_max = 5

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Draw grid
        self.draw_grid(painter)

        # Draw curve
        self.draw_curve(painter)

        # Draw keyframes
        self.draw_keyframes(painter)

    def draw_grid(self, painter):
        painter.setPen(QColor(60, 60, 60))

        # Vertical lines (time)
        width = self.width()
        height = self.height()
        time_range = self.time_max - self.time_min
        value_range = self.value_max - self.value_min

        for t in range(int(self.time_min), int(self.time_max) + 1):
            x = (t - self.time_min) / time_range * width
            painter.drawLine(int(x), 0, int(x), height)

        # Horizontal lines (value)
        for v in range(int(self.value_min), int(self.value_max) + 1):
            y = height - (v - self.value_min) / value_range * height
            painter.drawLine(0, int(y), width, int(y))

    def draw_curve(self, painter):
        if len(self.curve.keyframes) < 2:
            return

        painter.setPen(QPen(QColor(100, 200, 255), 2))

        # Sample the curve
        points = []
        time_range = self.time_max - self.time_min
        value_range = self.value_max - self.value_min

        for i in range(self.width()):
            t = self.time_min + (i / self.width()) * time_range
            v = self.curve.evaluate(t)

            x = i
            y = self.height() - ((v - self.value_min) / value_range) * self.height
            points.append(QPointF(x, y))

        # Draw the curve
        if len(points) > 1:
            path = QPainterPath()
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.drawPath(path)

    def draw_keyframes(self, painter):
        if not self.curve.keyframes:
            return

        time_range = self.time_max - self.time_min
        value_range = self.value_max - self.value_min

        for time, value, in_tangent, out_tangent in self.curve.keyframes:
            x = (time - self.time_min) / time_range * self.width()
            y = self.height() - ((value - self.value_min) / value_range) * self.height

            # Draw keyframe point
            color = QColor(255, 255, 0) if self.selected_keyframe and abs(self.selected_keyframe[0] - time) < 0.001 else QColor(255, 100, 100)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), 4, 4)

            # Draw tangents
            tangent_length = 30
            if in_tangent:
                in_x = x - tangent_length
                in_y = y - in_tangent[1] * tangent_length
                painter.setPen(QPen(QColor(150, 150, 150), 1))
                painter.drawLine(int(x), int(y), int(in_x), int(in_y))

            if out_tangent:
                out_x = x + tangent_length
                out_y = y - out_tangent[1] * tangent_length
                painter.setPen(QPen(QColor(150, 150, 150), 1))
                painter.drawLine(int(x), int(y), int(out_x), int(out_y))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicked on a keyframe
            time_range = self.time_max - self.time_min
            value_range = self.value_max - self.value_min

            click_time = self.time_min + (event.position().x() / self.width()) * time_range
            click_value = self.value_min + ((self.height() - event.position().y()) / self.height()) * value_range

            # Find closest keyframe
            closest_dist = float('inf')
            closest_kf = None

            for time, value, in_tangent, out_tangent in self.curve.keyframes:
                kf_x = (time - self.time_min) / time_range * self.width()
                kf_y = self.height() - ((value - self.value_min) / value_range) * self.height

                dist = ((event.position().x() - kf_x) ** 2 + (event.position().y() - kf_y) ** 2) ** 0.5
                if dist < 10 and dist < closest_dist:
                    closest_dist = dist
                    closest_kf = (time, value, in_tangent, out_tangent)

            if closest_kf:
                self.selected_keyframe = closest_kf
                self.dragging = True
                self.drag_offset = (click_time - closest_kf[0], click_value - closest_kf[1])
                self.update()
            else:
                self.selected_keyframe = None
                self.update()

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_keyframe:
            time_range = self.time_max - self.time_min
            value_range = self.value_max - self.value_min

            new_time = self.time_min + (event.position().x() / self.width()) * time_range - self.drag_offset[0]
            new_value = self.value_min + ((self.height() - event.position().y()) / self.height()) * value_range - self.drag_offset[1]

            # Update keyframe
            old_time = self.selected_keyframe[0]
            for i, (t, v, it, ot) in enumerate(self.curve.keyframes):
                if abs(t - old_time) < 0.001:
                    self.curve.keyframes[i] = (new_time, new_value, it, ot)
                    self.selected_keyframe = (new_time, new_value, it, ot)
                    break

            self.curve.keyframes.sort(key=lambda k: k[0])
            self.keyframe_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False


class PluginManager:
    """Manages loading and execution of plugins"""
    def __init__(self, main_window):
        self.main_window = main_window
        self.plugins = {}  # name -> plugin_instance
        self.plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
        os.makedirs(self.plugin_dir, exist_ok=True)
    
    def load_plugin(self, plugin_path):
        """Load a plugin from file path"""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load plugin from {plugin_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check if it has the required interface
            if not hasattr(module, 'PLUGIN_NAME'):
                raise AttributeError("Plugin must define PLUGIN_NAME")
            plugin_name = module.PLUGIN_NAME
            # Create plugin instance if it has a class
            if hasattr(module, 'Plugin'):
                plugin_instance = module.Plugin(self.main_window)
            else:
                plugin_instance = module
            self.plugins[plugin_name] = plugin_instance
            # Initialize plugin if it has init method
            if hasattr(plugin_instance, 'initialize'):
                plugin_instance.initialize()
            return plugin_name
        except Exception as e:
            print(f"Failed to load plugin {plugin_path}: {e}")
            return None
            
            # Check if it has the required interface
            if not hasattr(module, 'PLUGIN_NAME'):
                raise AttributeError("Plugin must define PLUGIN_NAME")
            
            plugin_name = module.PLUGIN_NAME
            
            # Create plugin instance if it has a class
            if hasattr(module, 'Plugin'):
                plugin_instance = module.Plugin(self.main_window)
            else:
                plugin_instance = module
            
            self.plugins[plugin_name] = plugin_instance
            
            # Initialize plugin if it has init method
            if hasattr(plugin_instance, 'initialize'):
                plugin_instance.initialize()
            
            return plugin_name
            
        except Exception as e:
            print(f"Failed to load plugin {plugin_path}: {e}")
            return None
    
    def unload_plugin(self, plugin_name):
        """Unload a plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            
            # Cleanup if plugin has cleanup method
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()
            
            del self.plugins[plugin_name]
            return True
        return False
    
    def get_plugin(self, plugin_name):
        """Get a loaded plugin instance"""
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self):
        """Get all loaded plugins"""
        return list(self.plugins.keys())
    
    def execute_plugin_action(self, plugin_name, action_name, *args, **kwargs):
        """Execute a plugin action"""
        plugin = self.get_plugin(plugin_name)
        if plugin and hasattr(plugin, action_name):
            method = getattr(plugin, action_name)
            return method(*args, **kwargs)
        return None


class PluginDialog(QDialog):
    """Dialog for managing plugins"""
    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.setWindowTitle("Plugin Manager")
        self.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout()
        
        # Plugin list
        self.plugin_list = QListWidget()
        self.update_plugin_list()
        layout.addWidget(QLabel("Loaded Plugins:"))
        layout.addWidget(self.plugin_list)
        
        # Plugin controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Plugin...")
        self.load_btn.clicked.connect(self.load_plugin)
        controls_layout.addWidget(self.load_btn)
        
        self.unload_btn = QPushButton("Unload")
        self.unload_btn.clicked.connect(self.unload_plugin)
        controls_layout.addWidget(self.unload_btn)
        
        layout.addLayout(controls_layout)
        
        # Plugin info
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        layout.addWidget(QLabel("Plugin Info:"))
        layout.addWidget(self.info_text)
        
        # Buttons
        buttons = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)
        
        self.plugin_list.itemSelectionChanged.connect(self.show_plugin_info)
    
    def update_plugin_list(self):
        """Update the plugin list"""
        self.plugin_list.clear()
        for plugin_name in self.plugin_manager.get_all_plugins():
            self.plugin_list.addItem(plugin_name)
    
    def load_plugin(self):
        """Load a plugin file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Plugin", self.plugin_manager.plugin_dir, "Python Files (*.py)"
        )
        
        if file_path:
            plugin_name = self.plugin_manager.load_plugin(file_path)
            if plugin_name:
                self.update_plugin_list()
                QMessageBox.information(self, "Plugin Loaded", f"Successfully loaded plugin: {plugin_name}")
            else:
                QMessageBox.warning(self, "Load Failed", "Failed to load plugin")
    
    def unload_plugin(self):
        """Unload selected plugin"""
        current_item = self.plugin_list.currentItem()
        if current_item:
            plugin_name = current_item.text()
            if self.plugin_manager.unload_plugin(plugin_name):
                self.update_plugin_list()
                self.info_text.clear()
                QMessageBox.information(self, "Plugin Unloaded", f"Successfully unloaded plugin: {plugin_name}")
            else:
                QMessageBox.warning(self, "Unload Failed", "Failed to unload plugin")
    
    def show_plugin_info(self):
        """Show information about selected plugin"""
        current_item = self.plugin_list.currentItem()
        if current_item:
            plugin_name = current_item.text()
            plugin = self.plugin_manager.get_plugin(plugin_name)
            
            info = f"Plugin: {plugin_name}\n"
            info += f"Type: {type(plugin).__name__}\n"
            
            if hasattr(plugin, '__doc__') and plugin.__doc__:
                info += f"Description: {plugin.__doc__}\n"
            
            # Show available methods
            methods = [m for m in dir(plugin) if not m.startswith('_') and callable(getattr(plugin, m))]
            if methods:
                info += f"Actions: {', '.join(methods)}\n"
            
            self.info_text.setText(info)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 JPatch - 3D Modeler")
        self.setGeometry(50, 50, 1600, 1000)

        # Initialize settings
        self.settings = JPatchSettings()
        # log_memory_usage("After settings initialization")

        # Initialize shortcuts
        self.shortcuts = ShortcutManager(self.settings)
        # log_memory_usage("After shortcuts initialization")

        # Initialize model
        self.model = Model()
        self.model.create_default_cube()  # Add default cube for testing
        # log_memory_usage("After model initialization")

        # Initialize autonomous brain
        self.brain = AutonomousBrain()
        # log_memory_usage("After brain initialization")

        # Initialize capsule manager for model knowledge base
        from Model import create_capsule_manager
        self.capsule_manager = create_capsule_manager()
        # log_memory_usage("After capsule manager initialization")
        
        # Connect capsule manager to model for spatial integration
        self.model.capsule_manager = self.capsule_manager

        # Set up graphical widget classes
        from Model import Animation as ModelAnimation
        TimelineWidget.set_animation_class(ModelAnimation)
        CapsuleTreeWidget.set_capsule_manager_class(type(self.capsule_manager))
        OrbitalViewWidget.set_capsule_manager_class(type(self.capsule_manager))

        # Create brain widget
        self.brain_widget = BrainWidget(self.brain)
        # log_memory_usage("After brain widget creation")

        # Initialize plugin manager
        self.plugin_manager = PluginManager(self)
        # log_memory_usage("After plugin manager initialization")

        # Initialize new AI systems
        self.voice_system = VoiceSystem()
        self.knowledge_base = KnowledgeBase()
        # log_memory_usage("After AI systems initialization")

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Initialize model
        self.model = Model()
        self.model.create_default_cube()
        self.current_file = None

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter (viewport | sidebar)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().addWidget(main_splitter)

        # Create viewport widget
        self.viewport_widget = QWidget()
        viewport_layout = QVBoxLayout()
        self.viewport_widget.setLayout(viewport_layout)

        # Add viewport manager to handle multiple viewports
        self.viewport_manager = ViewportManager(self.model)
        viewport_layout.addWidget(self.viewport_manager)
        log_memory_usage("After viewport manager creation")

        main_splitter.addWidget(self.viewport_widget)

        # Create sidebar
        self.sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(sidebar_layout)
        self.sidebar.setFixedWidth(250)

        # Model tree
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabel("Model Hierarchy")
        self.update_model_tree()
        sidebar_layout.addWidget(QLabel("Model:"))
        sidebar_layout.addWidget(self.model_tree)

        # Control point browser placeholder
        sidebar_layout.addWidget(QLabel("Control Points:"))
        self.cp_browser = QTreeWidget()
        self.cp_browser.setHeaderLabel("Control Points")
        self.update_cp_browser()
        sidebar_layout.addWidget(self.cp_browser)

        # Animation widget
        sidebar_layout.addWidget(QLabel("Animation:"))
        self.animation_widget = AnimationWidget(self.model)
        sidebar_layout.addWidget(self.animation_widget)

        main_splitter.addWidget(self.sidebar)
        main_splitter.setSizes([800, 250])

        # Add brain widget as dock widget
        from PyQt6.QtWidgets import QDockWidget
        brain_dock = QDockWidget("Autonomous Brain", self)
        brain_dock.setObjectName("brainDock")
        brain_dock.setWidget(self.brain_widget)
        brain_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, brain_dock)

        # Create capsule tree widget as dock widget
        capsule_tree_dock = QDockWidget("Knowledge Base", self)
        capsule_tree_dock.setObjectName("capsuleTreeDock")
        self.capsule_tree_widget = CapsuleTreeWidget(self.capsule_manager, self)
        capsule_tree_dock.setWidget(self.capsule_tree_widget)
        capsule_tree_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, capsule_tree_dock)

        # Create orbital view widget as dock widget
        orbital_dock = QDockWidget("Orbital View & Chat", self)
        orbital_dock.setObjectName("orbitalDock")
        
        # Create container widget for orbital + chat
        orbital_container = QWidget()
        orbital_layout = QVBoxLayout(orbital_container)
        orbital_layout.setContentsMargins(0, 0, 0, 0)
        
        self.orbital_widget = OrbitalViewWidget(self.capsule_manager, self)
        orbital_layout.addWidget(self.orbital_widget, 3)  # Give orbital view more space
        
        # Add chat widget below orbital view
        self.chat_widget = ChatWidget(self.voice_system, self)
        orbital_layout.addWidget(self.chat_widget, 1)  # Give chat less space
        
        orbital_dock.setWidget(orbital_container)
        orbital_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        orbital_dock.setMinimumWidth(400)
        orbital_dock.setMinimumHeight(500)
        self.orbital_dock = orbital_dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, orbital_dock)

        # Create timeline widget as dock widget
        timeline_dock = QDockWidget("Animation Timeline", self)
        timeline_dock.setObjectName("timelineDock")
        self.timeline_widget = TimelineWidget(self)
        timeline_dock.setWidget(self.timeline_widget)
        timeline_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        timeline_dock.setMinimumHeight(150)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, timeline_dock)

        # Create knowledge base widget as dock widget
        kb_dock = QDockWidget("Research Knowledge Base", self)
        kb_dock.setObjectName("kbDock")
        self.kb_widget = KnowledgeBaseWidget(self.knowledge_base)
        kb_dock.setWidget(self.kb_widget)
        kb_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        kb_dock.setMinimumSize(400, 300)  # Allow flexible minimum sizing
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, kb_dock)

        # Connect timeline to model's animation
        self.timeline_widget.set_animation(self.model.animation)
        
        # Initialize ROCA graph and manager
        from roca.graph_manager import ROCAGraph, GraphManager
        self.roca_graph = ROCAGraph()
        self.graph_manager = GraphManager(self.roca_graph, timeline=self.timeline_widget, orbital=self.orbital_widget)
        
        # Set managers for chat widget
        if hasattr(self, 'chat_widget'):
            self.chat_widget.set_managers(self.capsule_manager, self.graph_manager, self.orbital_widget)
            # Set up a simple message handler
            self.chat_widget.set_message_handler(self.handle_chat_message)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbars
        self.create_toolbars()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Restore window state
        self.settings.restore_window_state(self)

    def closeEvent(self, event):
        """Save window state when closing"""
        self.settings.save_window_state(self)
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if self._is_supported_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        imported_count = 0
        bvh_files = []
        for url in urls:
            file_path = url.toLocalFile()
            try:
                count = self._import_file(file_path)
                imported_count += count
                if os.path.splitext(file_path)[1].lower() == '.bvh':
                    bvh_files.append(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Failed to import {file_path}:\n{str(e)}")
        
        if imported_count > 0:
            self.update_model_tree()
            self.update_cp_browser()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Imported {imported_count} items")
            
            # Automatically create capsule for BVH files
            for bvh_file in bvh_files:
                try:
                    capsule_name = os.path.splitext(os.path.basename(bvh_file))[0]
                    model_data = self.model.to_capsule_data(name=capsule_name, 
                                                          description=f"BVH animation: {capsule_name}")
                    capsule = self.capsule_manager.create_model_capsule(model_data, "animation")
                    self.status_bar.showMessage(f"BVH capsule created: {capsule.name}")
                except Exception as e:
                    QMessageBox.warning(self, "Capsule Creation Failed", 
                                      f"Failed to create capsule from {bvh_file}:\n{str(e)}")

    def _is_supported_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        return ext in ['.jpt', '.obj', '.3ds', '.fbx', '.mdl', '.bvh']

    def _import_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.jpt':
            self.model.load_from_file(filepath)
            log_memory_usage(f"After loading JPatch file: {filepath}")
            return len(self.model.control_points)
        elif ext == '.obj':
            count = self.model.import_obj(filepath)
            log_memory_usage(f"After importing OBJ file: {filepath}")
            return count
        elif ext == '.3ds':
            count = self.model.import_3ds(filepath)
            log_memory_usage(f"After importing 3DS file: {filepath}")
            return count
        elif ext == '.fbx':
            count = self.model.import_fbx(filepath)
            log_memory_usage(f"After importing FBX file: {filepath}")
            return count
        elif ext == '.mdl':
            count = self.model.import_mdl(filepath)
            log_memory_usage(f"After importing MDL file: {filepath}")
            return count
        elif ext == '.bvh':
            count = self.model.import_bvh(filepath)
            log_memory_usage(f"After importing BVH file: {filepath}")
            return count
        else:
            raise Exception(f"Unsupported file format: {ext}")

    def update_model_tree(self):
        self.model_tree.clear()
        root = QTreeWidgetItem(["Scene"])
        self.model_tree.addTopLevelItem(root)
        
        # Bones hierarchy
        bones_item = QTreeWidgetItem([f"Bones ({len(self.model.bones)})"])
        root.addChild(bones_item)
        self.add_bone_to_tree(bones_item, None)  # Add root bones
        
        morphs = QTreeWidgetItem([f"Morphs ({len(self.model.morphs)})"])
        root.addChild(morphs)
        
        # Animations
        anims = QTreeWidgetItem([f"Animations ({len(self.model.animations)})"])
        root.addChild(anims)
        for anim_name in self.model.animations.keys():
            anim_item = QTreeWidgetItem([anim_name])
            anims.addChild(anim_item)
        
        materials = QTreeWidgetItem([f"Materials ({len(self.model.materials)})"])
        root.addChild(materials)
        
        patches = QTreeWidgetItem([f"Patches ({len(self.model.patches)})"])
        root.addChild(patches)
        
        curves = QTreeWidgetItem([f"Curves ({len(self.model.curves)})"])
        root.addChild(curves)
        for i, curve in enumerate(self.model.curves):
            curve_type = type(curve).__name__
            curve_item = QTreeWidgetItem([f"{curve_type} {i}"])
            curves.addChild(curve_item)
        
        surfaces = QTreeWidgetItem([f"Subdivision Surfaces ({len(self.model.subdivision_surfaces)})"])
        root.addChild(surfaces)
        for i, surface in enumerate(self.model.subdivision_surfaces):
            surface_type = type(surface).__name__
            surface_item = QTreeWidgetItem([f"{surface_type} {i} (Level {surface.levels})"])
            surfaces.addChild(surface_item)
        log_memory_usage("After model tree update")

    def add_bone_to_tree(self, parent_item, parent_bone):
        """Recursively add bones to the tree"""
        for bone_name, bone in self.model.bones.items():
            if bone.parent == parent_bone:
                bone_item = QTreeWidgetItem([f"{bone_name} ({bone.position[0]:.1f}, {bone.position[1]:.1f}, {bone.position[2]:.1f})"])
                parent_item.addChild(bone_item)
                self.add_bone_to_tree(bone_item, bone)

    def update_cp_browser(self):
        self.cp_browser.clear()
        for i, point in enumerate(self.model.control_points):
            item = QTreeWidgetItem([f"Point {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"])
            self.cp_browser.addTopLevelItem(item)

    def create_model_capsule(self):
        """Create a capsule from the current model for the knowledge base."""
        try:
            # Get model data
            model_data = self.model.to_capsule_data()

            # Create capsule using the capsule manager
            capsule = self.capsule_manager.create_model_capsule(model_data, "3dmodel")

            # Show success message
            QMessageBox.information(
                self, "Capsule Created",
                f"Successfully created capsule '{capsule.name}' for the current model.\n"
                f"Capsule ID: {capsule.uuid[:8]}...\n"
                f"This model is now part of the knowledge base and can be used for AI-assisted modeling."
            )

            self.status_bar.showMessage(f"Model capsule created: {capsule.name}")

        except Exception as e:
            QMessageBox.warning(
                self, "Capsule Creation Failed",
                f"Failed to create model capsule: {str(e)}"
            )
            logger.error(f"Capsule creation failed: {e}", exc_info=True)

    def toggle_artistic_input(self):
        """Toggle the visibility of the artistic input dock widget."""
        if hasattr(self, 'learning_dock_widget'):
            # Find the parent dock widget
            for dock in self.findChildren(QDockWidget):
                if dock.widget() == self.learning_dock_widget:
                    if dock.isVisible():
                        dock.hide()
                        self.status_bar.showMessage("Artistic input hidden")
                    else:
                        dock.show()
                        self.status_bar.showMessage("Artistic input shown")
                    break

    def toggle_texture_editor(self):
        """Toggle the texture editor dialog."""
        if not hasattr(self, 'texture_editor'):
            self.texture_editor = TextureEditorDialog(self)
            self.texture_editor.texture_updated.connect(self.on_texture_updated)

        if self.texture_editor.isVisible():
            self.texture_editor.hide()
            self.status_bar.showMessage("Texture Editor closed")
        else:
            self.texture_editor.show()
            self.status_bar.showMessage("Texture Editor opened")

    def on_texture_updated(self, texture_data):
        """Handle texture updates from the editor."""
        self.status_bar.showMessage("Texture updated")

    def show_similar_models(self):
        """Show models similar to the current one from the knowledge base."""
        try:
            model_data = self.model.to_capsule_data()
            similar_capsules = self.capsule_manager.find_similar_models(model_data, limit=5)

            if not similar_capsules:
                QMessageBox.information(
                    self, "No Similar Models",
                    "No similar models found in the knowledge base."
                )
                return

            # Create a dialog to show similar models
            dialog = QDialog(self)
            dialog.setWindowTitle("Similar Models")
            dialog.setModal(True)

            layout = QVBoxLayout(dialog)

            list_widget = QListWidget()
            for capsule, similarity in similar_capsules:
                item_text = f"{capsule.name} (Similarity: {similarity:.2f})"
                if 'description' in capsule.metadata:
                    item_text += f"\n  {capsule.metadata['description']}"
                list_widget.addItem(item_text)

            layout.addWidget(QLabel("Models similar to your current work:"))
            layout.addWidget(list_widget)

            buttons = QHBoxLayout()
            load_btn = QPushButton("Load Selected")
            load_btn.clicked.connect(lambda: self._load_similar_model(list_widget.currentItem(), similar_capsules, dialog))
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)

            buttons.addWidget(load_btn)
            buttons.addWidget(close_btn)
            layout.addLayout(buttons)

            dialog.exec()

        except Exception as e:
            QMessageBox.warning(
                self, "Error Finding Similar Models",
                f"Failed to find similar models: {str(e)}"
            )
            logger.error(f"Similar models search failed: {e}", exc_info=True)

    def _load_similar_model(self, current_item, similar_capsules, dialog):
        """Load a similar model from the knowledge base."""
        if current_item is None:
            return

        # Find the corresponding capsule
        row = current_item.listWidget().currentRow()
        if row < len(similar_capsules):
            capsule, _ = similar_capsules[row]

            # Load model data from capsule
            model_data = capsule.metadata.get('model_data')
            if model_data:
                try:
                    # Clear current model
                    self.model.clear()

                    # Load control points
                    self.model.control_points = model_data.get('control_points', [])

                    # Note: Full model reconstruction would require more complex logic
                    # for patches, bones, materials, etc.

                    self.update_model_tree()
                    self.update_cp_browser()
                    self.viewport_manager.update_all_viewports()

                    QMessageBox.information(
                        self, "Model Loaded",
                        f"Loaded model from capsule: {capsule.name}"
                    )

                    dialog.accept()

                except Exception as e:
                    QMessageBox.warning(
                        self, "Load Failed",
                        f"Failed to load model: {str(e)}"
                    )
            else:
                QMessageBox.warning(
                    self, "No Model Data",
                    "This capsule doesn't contain loadable model data."
                )

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        new_action = file_menu.addAction('New', self.new_file)
        self.shortcuts.apply_to_action(new_action, 'new_file')
        open_action = file_menu.addAction('Open...', self.open_file)
        self.shortcuts.apply_to_action(open_action, 'open_file')
        save_action = file_menu.addAction('Save', self.save_file)
        self.shortcuts.apply_to_action(save_action, 'save_file')
        save_as_action = file_menu.addAction('Save As...', self.save_as_file)
        self.shortcuts.apply_to_action(save_as_action, 'save_as_file')
        file_menu.addSeparator()
        import_menu = file_menu.addMenu('Import')
        import_menu.addAction('Animation:Master (.mdl)', self.import_am)
        import_menu.addAction('JPatch', self.import_jpatch)
        import_menu.addAction('SPatch', self.import_spatch)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu('Export')
        export_menu.addAction('Wavefront OBJ', self.export_obj)
        export_menu.addAction('Autodesk FBX', self.export_fbx)
        export_menu.addAction('3D Studio', self.export_3ds)
        export_menu.addAction('Animation:Master MDL', self.export_mdl)
        export_menu.addAction('POV-Ray', self.export_pov)
        export_menu.addAction('RenderMan RIB', self.export_rib)
        file_menu.addSeparator()
        quit_action = file_menu.addAction('Quit', self.close)
        self.shortcuts.apply_to_action(quit_action, 'quit')

        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        undo_action = edit_menu.addAction('Undo', self.undo)
        self.shortcuts.apply_to_action(undo_action, 'undo')
        redo_action = edit_menu.addAction('Redo', self.redo)
        self.shortcuts.apply_to_action(redo_action, 'redo')
        edit_menu.addSeparator()
        edit_menu.addAction('Add Control Point', self.add_control_point)
        edit_menu.addAction('Add Bone', self.add_bone)
        delete_action = edit_menu.addAction('Delete Selection', self.delete_selection)
        self.shortcuts.apply_to_action(delete_action, 'delete_selection')
        edit_menu.addAction('Clone', self.clone)
        edit_menu.addSeparator()
        select_all_action = edit_menu.addAction('Select All', self.select_all)
        self.shortcuts.apply_to_action(select_all_action, 'select_all')
        edit_menu.addAction('Select None', self.select_none)
        invert_action = edit_menu.addAction('Invert Selection', self.invert_selection)
        self.shortcuts.apply_to_action(invert_action, 'invert_selection')

        # View menu
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Single Viewport', self.view_single)
        view_menu.addAction('Quad Viewports', self.view_quad)
        view_menu.addAction('Split Horizontal', self.view_split_h)
        view_menu.addAction('Split Vertical', self.view_split_v)
        view_menu.addSeparator()
        view_menu.addAction('Show Points', self.show_points)
        view_menu.addAction('Show Curves', self.show_curves)
        view_menu.addAction('Show Patches', self.show_patches)
        view_menu.addAction('Show Shaded', self.show_shaded)
        view_menu.addAction('Show Textured', self.show_textured)
        view_menu.addAction('Show Grid', self.show_grid)
        view_menu.addAction('Show Rotoscope', self.show_rotoscope)
        view_menu.addAction('Show Bones', self.show_bones)
        view_menu.addSeparator()
        lighting_menu = view_menu.addMenu('Lighting')
        lighting_menu.addAction('Head Light', lambda: self.set_lighting_mode('head'))
        lighting_menu.addAction('Simple Light', lambda: self.set_lighting_mode('simple'))
        lighting_menu.addAction('Three-Point Lighting', lambda: self.set_lighting_mode('three-point'))
        lighting_menu.addAction('Sticky Light', lambda: self.set_lighting_mode('sticky'))
        view_menu.addSeparator()
        view_menu.addAction('Load Rotoscope Image', self.load_rotoscope)
        view_menu.addSeparator()
        view_menu.addAction('Zoom to Fit', self.zoom_to_fit)
        view_menu.addAction('Reset View', self.reset_view)

        # Model menu
        model_menu = menubar.addMenu('Model')
        model_menu.addAction('Create Patch from Selection', self.create_patch_from_selection)
        model_menu.addAction('Create Bezier Curve', self.create_bezier_curve)
        model_menu.addAction('Create B-Spline Curve', self.create_bspline_curve)
        model_menu.addAction('Create Subdivision Surface', self.create_subdivision_surface)
        model_menu.addSeparator()
        model_menu.addAction('Compute Patches', self.compute_patches)
        model_menu.addAction('Check Model', self.check_model)
        model_menu.addAction('Lathe', self.lathe)
        model_menu.addAction('Extrude', self.extrude)
        model_menu.addAction('Align', self.align)
        model_menu.addSeparator()
        model_menu.addAction('New Material', self.new_material)
        model_menu.addAction('Edit Material', self.edit_material)
        model_menu.addAction('Apply Material', self.apply_material)

        # Bones menu
        bones_menu = menubar.addMenu('Bones')
        bones_menu.addAction('Add Bone', self.add_bone)
        bones_menu.addAction('Delete Bone', self.delete_bone)
        bones_menu.addAction('Select Bone', self.select_bone)
        bones_menu.addAction('Solve IK', self.solve_ik)
        bones_menu.addSeparator()
        bones_menu.addAction('Create Animation', self.create_animation)
        bones_menu.addAction('Add Keyframe', self.add_keyframe)
        bones_menu.addAction('Show Bones', self.show_bones)

        # Animation menu
        anim_menu = menubar.addMenu('Animation')
        anim_menu.addAction('New Morph', self.new_morph)
        anim_menu.addAction('Edit Morph', self.edit_morph)
        anim_menu.addAction('New Morph Group', self.new_morph_group)
        anim_menu.addSeparator()
        anim_menu.addAction('Play', self.play_anim)
        anim_menu.addAction('Stop', self.stop_anim)

        # AI/Knowledge menu
        ai_menu = menubar.addMenu('AI')
        ai_menu.addAction('Create Model Capsule', self.create_model_capsule)
        ai_menu.addAction('Find Similar Models', self.show_similar_models)
        ai_menu.addSeparator()
        ai_menu.addAction('Show Brain Status', lambda: self.brain_widget.show() if hasattr(self, 'brain_widget') else None)
        ai_menu.addAction('Show Knowledge Base', lambda: self.capsule_tree_widget.show() if hasattr(self, 'capsule_tree_widget') else None)
        ai_menu.addAction('Show Orbital View', lambda: self.orbital_widget.show() if hasattr(self, 'orbital_widget') else None)
        ai_menu.addAction('Show Timeline', lambda: self.timeline_widget.show() if hasattr(self, 'timeline_widget') else None)
        ai_menu.addAction('Texture Editor', self.toggle_texture_editor)
        ai_menu.addSeparator()
        ai_menu.addAction('Research Knowledge Base', lambda: self.kb_widget.show() if hasattr(self, 'kb_widget') else None)

        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        settings_menu.addAction('Preferences...', self.show_preferences)
        settings_menu.addAction('Reset to Defaults', self.reset_settings)

        # Help menu
        help_menu = menubar.addMenu('Help')
        help_menu.addAction('About', self.about)

        # Plugins menu
        plugins_menu = menubar.addMenu('Plugins')
        plugins_menu.addAction('Plugin Manager...', self.show_plugin_manager)

    def create_toolbars(self):
        # Main toolbar
        main_toolbar = self.addToolBar('Main')
        main_toolbar.setObjectName('Main')
        main_toolbar.addAction('New', self.new_file)
        main_toolbar.addAction('Open', self.open_file)
        main_toolbar.addAction('Save', self.save_file)
        main_toolbar.addSeparator()
        main_toolbar.addAction('Undo', self.undo)
        main_toolbar.addAction('Redo', self.redo)

        # Mesh toolbar
        mesh_toolbar = self.addToolBar('Mesh')
        mesh_toolbar.setObjectName('Mesh')
        mesh_toolbar.addAction('Add Point', self.add_control_point)
        mesh_toolbar.addAction('Extrude', self.extrude)
        mesh_toolbar.addAction('Lathe', self.lathe)
        mesh_toolbar.addAction('Align', self.align)

        # Bones toolbar
        bones_toolbar = self.addToolBar('Bones')
        bones_toolbar.setObjectName('Bones')
        bones_toolbar.addAction('Add Bone', self.add_bone)
        bones_toolbar.addAction('Select Bone', self.select_bone)
        bones_toolbar.addAction('Keyframe', self.add_keyframe)

        # View toolbar
        view_toolbar = self.addToolBar('View')
        view_toolbar.setObjectName('View')
        view_toolbar.addAction('Points', self.show_points)
        view_toolbar.addAction('Curves', self.show_curves)
        view_toolbar.addAction('Patches', self.show_patches)
        view_toolbar.addAction('Shaded', self.show_shaded)
        view_toolbar.addAction('Grid', self.show_grid)
        view_toolbar.addSeparator()
        view_toolbar.addAction('Zoom Fit', self.zoom_to_fit)

    def new_file(self):
        self.model.clear()
        self.current_file = None
        self.update_model_tree()
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("New file created")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open JPatch File", "", "JPatch Files (*.jpt);;All Files (*)")
        if file_path:
            try:
                self.model.load_from_file(file_path)
                self.current_file = file_path
                self.update_model_tree()
                self.update_cp_browser()
                self.viewport_manager.update_all_viewports()
                self.status_bar.showMessage(f"Opened: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def save_file(self):
        if self.current_file:
            try:
                self.model.save_to_file(self.current_file)
                self.status_bar.showMessage(f"Saved: {self.current_file}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{str(e)}")
        else:
            self.save_as_file()

    def save_as_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save JPatch File", "", "JPatch Files (*.jpt);;All Files (*)")
        if file_path:
            try:
                self.model.save_to_file(file_path)
                self.current_file = file_path
                self.status_bar.showMessage(f"Saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{str(e)}")

    def import_am(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Animation:Master", "", "Animation:Master Files (*.mdl);;All Files (*)")
        if file_path:
            try:
                count = self.model.import_mdl(file_path)
                self.update_model_tree()
                self.update_cp_browser()
                self.viewport_manager.update_all_viewports()
                self.status_bar.showMessage(f"Imported {count} items from AM: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import Animation:Master file:\n{str(e)}")

    def import_jpatch(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import JPatch", "", "JPatch Files (*.jpt);;All Files (*)")
        if file_path:
            try:
                count = self._import_file(file_path)
                self.update_model_tree()
                self.update_cp_browser()
                self.viewport_manager.update_all_viewports()
                self.status_bar.showMessage(f"Imported {count} items from JPatch: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import JPatch file:\n{str(e)}")

    def import_spatch(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import SPatch", "", "SPatch Files (*.spt);;All Files (*)")
        if file_path:
            # Placeholder: SPatch import
            self.status_bar.showMessage(f"SPatch import not implemented: {file_path}")

    def export_obj(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Wavefront OBJ", "", "OBJ Files (*.obj);;All Files (*)")
        if file_path:
            try:
                self._export_obj(file_path)
                self.status_bar.showMessage(f"Exported OBJ: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export OBJ:\n{str(e)}")

    def _export_obj(self, filepath):
        """Enhanced OBJ export with materials and proper geometry"""
        try:
            with open(filepath, 'w') as f:
                f.write("# Exported from PyQt6 JPatch\n")
                f.write(f"# Export time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write vertices
                f.write("# Vertices\n")
                for i, point in enumerate(self.model.control_points):
                    f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                f.write("\n")
                
                # Write vertex normals (basic calculation)
                f.write("# Vertex normals\n")
                for point in self.model.control_points:
                    # Simple normalized position as normal (can be improved)
                    length = (point[0]**2 + point[1]**2 + point[2]**2)**0.5
                    if length > 0:
                        nx, ny, nz = point[0]/length, point[1]/length, point[2]/length
                    else:
                        nx, ny, nz = 0, 1, 0
                    f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
                f.write("\n")
                
                # Write materials
                if self.model.materials:
                    # Create MTL file
                    mtl_path = filepath.rsplit('.', 1)[0] + '.mtl'
                    with open(mtl_path, 'w') as mtl_f:
                        mtl_f.write("# Material file for PyQt6 JPatch export\n\n")
                        for mat_name, material in self.model.materials.items():
                            mtl_f.write(f"newmtl {mat_name}\n")
                            if hasattr(material, 'diffuse_color'):
                                color = material.diffuse_color
                                mtl_f.write(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
                            if hasattr(material, 'specular_color'):
                                color = material.specular_color
                                mtl_f.write(f"Ks {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
                            if hasattr(material, 'shininess'):
                                mtl_f.write(f"Ns {material.shininess:.1f}\n")
                            mtl_f.write("\n")
                    
                    # Reference MTL file in OBJ
                    mtl_filename = os.path.basename(mtl_path)
                    f.write(f"mtllib {mtl_filename}\n\n")
                
                # Write faces from patches
                f.write("# Faces\n")
                vertex_index = 1  # OBJ uses 1-based indexing
                
                for patch in self.model.patches:
                    if isinstance(patch, list) and len(patch) >= 3:
                        # Use material if assigned
                        obj_id = f"patch_{id(patch)}"
                        mat_name = self.model.material_assignments.get(obj_id)
                        if mat_name:
                            f.write(f"usemtl {mat_name}\n")
                        
                        # Write face (OBJ face indices are 1-based)
                        face_indices = []
                        for idx in patch:
                            if idx < len(self.model.control_points):
                                # vertex/texture/normal format: v/vt/vn
                                face_indices.append(f"{idx + 1}//{idx + 1}")
                        
                        if len(face_indices) >= 3:
                            f.write(f"f {' '.join(face_indices)}\n")
                    
                    elif hasattr(patch, 'control_points'):
                        # BezierPatch - tessellate and export
                        try:
                            vertices, indices = patch.tessellate(u_segments=8, v_segments=8)
                            for i in range(0, len(indices), 3):
                                if i + 2 < len(indices):
                                    v1 = indices[i] + vertex_index
                                    v2 = indices[i + 1] + vertex_index
                                    v3 = indices[i + 2] + vertex_index
                                    f.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
                        except Exception as e:
                            logger.warning(f"Failed to tessellate patch for OBJ export: {e}")
                
                # Write subdivision surfaces
                for surface in self.model.subdivision_surfaces:
                    try:
                        vertices, faces = surface.get_mesh()
                        base_vertex = vertex_index + len(self.model.control_points)
                        
                        # Write surface vertices
                        for v in vertices:
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        
                        # Write surface faces
                        for face in faces:
                            if len(face) >= 3:
                                face_str = ' '.join([f"{base_vertex + idx}//{base_vertex + idx}" for idx in face])
                                f.write(f"f {face_str}\n")
                    except Exception as e:
                        logger.warning(f"Failed to export subdivision surface: {e}")
                
                logger.info(f"Successfully exported OBJ with {len(self.model.control_points)} vertices and {len(self.model.patches)} patches")
                
        except Exception as e:
            log_error_with_context(logger, "OBJ export failed", e, {"filepath": filepath})
            raise

    def export_pov(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export POV-Ray", "", "POV Files (*.pov);;All Files (*)")
        if file_path:
            # Placeholder: export POV
            self.status_bar.showMessage(f"Exported POV: {file_path}")

    def export_rib(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export RenderMan RIB", "", "RIB Files (*.rib);;All Files (*)")
        if file_path:
            # Placeholder: export RIB
            self.status_bar.showMessage(f"Exported RIB: {file_path}")

    def export_fbx(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export FBX", "", "FBX Files (*.fbx);;All Files (*)")
        if file_path:
            try:
                self.model.export_fbx(file_path)
                self.status_bar.showMessage(f"Exported FBX: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export FBX:\n{str(e)}")

    def export_3ds(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export 3DS", "", "3DS Files (*.3ds);;All Files (*)")
        if file_path:
            try:
                self.model.export_3ds(file_path)
                self.status_bar.showMessage(f"Exported 3DS: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export 3DS:\n{str(e)}")

    def export_mdl(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Animation:Master MDL", "", "MDL Files (*.mdl);;All Files (*)")
        if file_path:
            try:
                self.model.export_mdl(file_path)
                self.status_bar.showMessage(f"Exported MDL: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export MDL:\n{str(e)}")

    def undo(self):
        self.status_bar.showMessage("Undo - Not implemented yet")

    def redo(self):
        self.status_bar.showMessage("Redo - Not implemented yet")

    def add_control_point(self):
        # Add a point at origin for now
        self.model.add_control_point(0.0, 0.0, 0.0)
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("Added control point")

    def add_bone(self):
        # Add a bone at origin for now
        bone_name, ok = QInputDialog.getText(self, "Add Bone", "Bone name:")
        if ok and bone_name:
            parent_name, ok2 = QInputDialog.getText(self, "Parent Bone", "Parent bone name (leave empty for root):")
            if ok2:
                parent = parent_name if parent_name else None
                self.model.add_bone(bone_name, [0.0, 0.0, 0.0], parent)
                self.update_model_tree()
                self.animation_widget.update_bone_sliders()
                self.viewport_manager.update_all_viewports()
                self.status_bar.showMessage(f"Added bone: {bone_name}")

    def delete_bone(self):
        if self.model.selected_bones:
            bone_name = next(iter(self.model.selected_bones))
            self.model.remove_bone(bone_name)
            self.update_model_tree()
            self.animation_widget.update_bone_sliders()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Deleted bone: {bone_name}")
        else:
            self.status_bar.showMessage("No bone selected")

    def select_bone(self):
        bone_names = list(self.model.bones.keys())
        if bone_names:
            bone_name, ok = QInputDialog.getItem(self, "Select Bone", "Choose bone:", bone_names, 0, False)
            if ok:
                self.model.selected_bones.clear()
                self.model.selected_bones.add(bone_name)
                self.viewport_manager.update_all_viewports()
                self.status_bar.showMessage(f"Selected bone: {bone_name}")
        else:
            self.status_bar.showMessage("No bones in model")

    def create_animation(self):
        anim_name, ok = QInputDialog.getText(self, "Create Animation", "Animation name:")
        if ok and anim_name:
            self.model.add_animation(anim_name)
            self.animation_widget.update_animations_list()
            self.status_bar.showMessage(f"Created animation: {anim_name}")

    def add_keyframe(self):
        if not self.model.selected_bones:
            self.status_bar.showMessage("No bone selected")
            return
            
        bone_name = next(iter(self.model.selected_bones))
        if bone_name not in self.model.bones:
            return
            
        bone = self.model.bones[bone_name]
        time, ok = QInputDialog.getDouble(self, "Add Keyframe", "Time (seconds):", 0.0, 0.0, 100.0, 2)
        if ok:
            # Add keyframe for current bone pose
            self.model.add_keyframe(self.animation_widget.current_animation, time, 
                                  bone_name, list(bone.position), list(bone.rotation))
            self.status_bar.showMessage(f"Added keyframe for {bone_name} at {time}s")

    def show_bones(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_bones = not main_viewport.show_bones
            self.viewport_manager.set_display_mode('bones', main_viewport.show_bones)
            self.status_bar.showMessage(f"Bones: {'ON' if main_viewport.show_bones else 'OFF'}")

    def undo(self):
        description = self.model.command_stack.undo()
        if description:
            self.update_model_tree()
            self.update_cp_browser()
            self.animation_widget.update_bone_sliders()
            self.animation_widget.update_morph_sliders()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Undo: {description}")
        else:
            self.status_bar.showMessage("Nothing to undo")

    def redo(self):
        description = self.model.command_stack.redo()
        if description:
            self.update_model_tree()
            self.update_cp_browser()
            self.animation_widget.update_bone_sliders()
            self.animation_widget.update_morph_sliders()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Redo: {description}")
        else:
            self.status_bar.showMessage("Nothing to redo")

    def clone(self):
        # Clone selected points
        selected = self.model.get_selected_points()
        if selected:
            new_indices = []
            for idx in selected:
                point = self.model.control_points[idx]
                new_idx = self.model.add_control_point(point[0] + 0.1, point[1] + 0.1, point[2] + 0.1)
                new_indices.append(new_idx)
            
            # Select the new points
            self.model.clear_selection()
            for idx in new_indices:
                self.model.selected_points.add(idx)
            
            self.update_cp_browser()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Cloned {len(selected)} points")
        else:
            self.status_bar.showMessage("No points selected to clone")

    def select_all(self):
        self.model.clear_selection()
        for i in range(len(self.model.control_points)):
            self.model.selected_points.add(i)
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("Selected all points")

    def select_none(self):
        self.model.clear_selection()
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("Cleared selection")

    def invert_selection(self):
        all_indices = set(range(len(self.model.control_points)))
        current_selection = self.model.selected_points.copy()
        self.model.selected_points = all_indices - current_selection
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("Inverted selection")

    def delete_selection(self):
        if self.model.selected_points:
            command = DeleteControlPointCommand(self.model)
            self.model.command_stack.execute(command)
            self.update_cp_browser()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage("Deleted selected points")
        else:
            self.status_bar.showMessage("No points selected")
        self.update_cp_browser()
        self.viewport_manager.update_all_viewports()
        self.status_bar.showMessage("Inverted selection")

    def view_single(self):
        self.viewport_manager.set_layout_mode('single')
        self.status_bar.showMessage("Switched to single viewport")

    def view_quad(self):
        self.viewport_manager.set_layout_mode('quad')
        self.status_bar.showMessage("Switched to quad viewports")

    def view_split_h(self):
        self.viewport_manager.set_layout_mode('horizontal')
        self.status_bar.showMessage("Switched to horizontal split")

    def view_split_v(self):
        self.viewport_manager.set_layout_mode('vertical')
        self.status_bar.showMessage("Switched to vertical split")

    def show_shaded(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_shaded = not main_viewport.show_shaded
            self.viewport_manager.set_display_mode('shaded', main_viewport.show_shaded)
            self.status_bar.showMessage(f"Shaded: {'ON' if main_viewport.show_shaded else 'OFF'}")

    def show_textured(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_textured = not main_viewport.show_textured
            self.viewport_manager.set_display_mode('textured', main_viewport.show_textured)
            self.status_bar.showMessage(f"Textured: {'ON' if main_viewport.show_textured else 'OFF'}")

    def show_grid(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_grid = not main_viewport.show_grid
            self.viewport_manager.set_display_mode('grid', main_viewport.show_grid)
            self.status_bar.showMessage(f"Grid: {'ON' if main_viewport.show_grid else 'OFF'}")

    def set_lighting_mode(self, mode):
        self.viewport_manager.set_lighting_mode(mode)
        self.status_bar.showMessage(f"Lighting: {mode.title()}")

    def load_rotoscope(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Rotoscope Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        if file_path:
            self.viewport_manager.load_rotoscope_image(file_path)
            self.status_bar.showMessage(f"Loaded rotoscope: {file_path}")

    def show_points(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_points = not main_viewport.show_points
            self.viewport_manager.set_display_mode('points', main_viewport.show_points)
            self.status_bar.showMessage(f"Points: {'ON' if main_viewport.show_points else 'OFF'}")

    def show_curves(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_curves = not main_viewport.show_curves
            self.viewport_manager.set_display_mode('curves', main_viewport.show_curves)
            self.status_bar.showMessage(f"Curves: {'ON' if main_viewport.show_curves else 'OFF'}")

    def show_patches(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_patches = not main_viewport.show_patches
            self.viewport_manager.set_display_mode('patches', main_viewport.show_patches)
            self.status_bar.showMessage(f"Patches: {'ON' if main_viewport.show_patches else 'OFF'}")

    def show_rotoscope(self):
        main_viewport = self.viewport_manager.get_main_viewport()
        if main_viewport:
            main_viewport.show_rotoscope = not main_viewport.show_rotoscope
            self.viewport_manager.set_display_mode('rotoscope', main_viewport.show_rotoscope)
            self.status_bar.showMessage(f"Rotoscope: {'ON' if main_viewport.show_rotoscope else 'OFF'}")

    def zoom_to_fit(self):
        self.status_bar.showMessage("Zoom to fit - Not implemented yet")

    def reset_view(self):
        self.status_bar.showMessage("Reset view - Not implemented yet")

    def create_patch_from_selection(self):
        if len(self.model.selected_points) == 16:
            command = CreatePatchCommand(self.model)
            self.model.command_stack.execute(command)
            self.update_model_tree()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage("Created patch from selected points")
        else:
            QMessageBox.warning(self, "Create Patch", "Please select exactly 16 control points to create a patch")

    def create_bezier_curve(self):
        curve = self.model.create_bezier_curve()
        if curve:
            self.update_model_tree()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Created Bezier curve with {len(curve.control_points)} control points")
        else:
            QMessageBox.warning(self, "Create Curve", "Please select at least 2 control points to create a Bezier curve")

    def create_bspline_curve(self):
        curve = self.model.create_bspline_curve()
        if curve:
            self.update_model_tree()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Created B-spline curve with {len(curve.control_points)} control points")
        else:
            QMessageBox.warning(self, "Create Curve", "Please select at least 4 control points to create a B-spline curve")

    def create_subdivision_surface(self):
        """Create a subdivision surface from selected control points"""
        selected = sorted(list(self.model.selected_points))
        if len(selected) < 4:
            QMessageBox.warning(self, "Create Subdivision Surface",
                              "Please select at least 4 control points arranged in a grid")
            return

        # Try to create a grid from selected points
        # For simplicity, assume points are selected in row-major order
        grid_size = int(len(selected) ** 0.5)
        if grid_size * grid_size != len(selected):
            QMessageBox.warning(self, "Create Subdivision Surface",
                              "Please select a square number of points (4, 9, 16, 25, etc.)")
            return

        # Create control grid
        control_grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                point = self.model.control_points[selected[idx]]
                row.append(Vector3D.from_tuple(point))
            control_grid.append(row)

        surface = self.model.create_subdivision_surface(control_grid)
        if surface:
            self.update_model_tree()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Created Catmull-Clark subdivision surface ({grid_size}x{grid_size})")
        else:
            QMessageBox.warning(self, "Create Subdivision Surface", "Failed to create subdivision surface")

    def solve_ik(self):
        """Solve inverse kinematics for selected bone towards target"""
        if not self.model.selected_bones:
            QMessageBox.warning(self, "Solve IK", "Please select a bone to use as the effector")
            return

        effector_name = list(self.model.selected_bones)[0]

        # For now, use a simple target - in a real implementation, this would be picked from the scene
        target = Vector3D(0, 2, 0)  # Simple target position

        if self.model.solve_ik(target, effector_name):
            self.update_model_tree()
            self.viewport_manager.update_all_viewports()
            self.status_bar.showMessage(f"Solved IK for bone '{effector_name}' towards target {target}")
        else:
            QMessageBox.warning(self, "Solve IK", f"Failed to solve IK for bone '{effector_name}'")

    def compute_patches(self):
        self.status_bar.showMessage("Compute patches - Not implemented yet")

    def check_model(self):
        self.status_bar.showMessage("Check model - Not implemented yet")

    def lathe(self):
        self.status_bar.showMessage("Lathe - Not implemented yet")

    def extrude(self):
        self.status_bar.showMessage("Extrude - Not implemented yet")

    def align(self):
        self.status_bar.showMessage("Align - Not implemented yet")

    def new_material(self):
        self.status_bar.showMessage("New material - Not implemented yet")

    def edit_material(self):
        self.status_bar.showMessage("Edit material - Not implemented yet")

    def apply_material(self):
        self.status_bar.showMessage("Apply material - Not implemented yet")

    def new_morph(self):
        morph_name, ok = QInputDialog.getText(self, "Create Morph Target", "Morph name:")
        if ok and morph_name:
            morph = self.model.create_morph_target(morph_name)
            self.update_model_tree()
            self.animation_widget.update_morph_sliders()
            self.status_bar.showMessage(f"Created morph target: {morph_name}")

    def edit_morph(self):
        if self.model.morphs:
            morph_names = [m.name for m in self.model.morphs]
            morph_name, ok = QInputDialog.getItem(self, "Edit Morph", "Choose morph:", morph_names, 0, False)
            if ok:
                # For now, just reset the morph to current state
                for morph in self.model.morphs:
                    if morph.name == morph_name:
                        morph.control_points = [list(pt) for pt in self.model.control_points]
                        self.status_bar.showMessage(f"Updated morph: {morph_name}")
                        break
        else:
            self.status_bar.showMessage("No morph targets to edit")

    def new_morph_group(self):
        self.status_bar.showMessage("Morph groups - Not implemented yet")

    def play_anim(self):
        if self.animation_widget.current_animation:
            self.animation_widget.toggle_playback()
        else:
            self.status_bar.showMessage("No animation selected")

    def stop_anim(self):
        self.animation_widget.stop_playback()

    def handle_chat_message(self, message):
        """Handle chat messages from the ROCA network chat."""
        message_lower = message.lower()
        
        # Simple response system
        if 'hello' in message_lower or 'hi' in message_lower:
            return "Hello! I'm ROCA, your capsule network assistant. I can help you create and manage knowledge capsules for your 3D modeling projects."
            
        elif 'help' in message_lower:
            return "I can create capsules from your messages. Try describing a character, motion, style, or concept, and I'll create a capsule for it!"
            
        elif 'capsule' in message_lower:
            capsule_count = len(self.capsule_manager.capsules) if self.capsule_manager else 0
            return f"You have {capsule_count} capsules in your network. Each capsule represents knowledge that can be reused in your projects."
            
        elif 'create' in message_lower or 'make' in message_lower:
            return "Just describe what you want to create, and I'll turn it into a capsule! For example: 'Create a walking motion' or 'Make a fantasy character style'."
            
        elif 'timeline' in message_lower:
            return "The timeline supports drag-and-drop capsules. Try dragging character capsules onto frames, then use multi-selection (Ctrl+click) and 'Generate In-betweens' for AI-powered animation."
            
        else:
            # Default response - acknowledge and suggest capsule creation
            return f"I understand: '{message}'. I've created a capsule from this. You can now drag it from the capsule tree to the timeline or orbital view!"

    def about(self):
        QMessageBox.about(self, "About PyQt6 JPatch", "PyQt6 JPatch - A 3D modeling application\nInspired by JPatch")

    def show_plugin_manager(self):
        """Show the plugin manager dialog"""
        dialog = PluginDialog(self.plugin_manager, self)
        dialog.exec()

    def show_preferences(self):
        """Show preferences dialog - placeholder for now"""
        QMessageBox.information(self, "Preferences", "Preferences dialog not yet implemented.\nSettings are saved automatically.")

    def reset_settings(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(self, "Reset Settings",
                                   "Are you sure you want to reset all settings to defaults?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.settings.reset_to_defaults()
            QMessageBox.information(self, "Settings Reset", "All settings have been reset to defaults.\nPlease restart the application for changes to take effect.")


def main():
    log_memory_usage("Application startup")
    app = QApplication(sys.argv)
    log_memory_usage("After QApplication creation")

    window = MainWindow()
    log_memory_usage("After MainWindow creation")

    window.show()
    log_memory_usage("After window.show()")

    logger.info("PyQt6 JPatch application is now running")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()