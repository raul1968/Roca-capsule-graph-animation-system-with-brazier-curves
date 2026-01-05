# PyQt6 JPatch - 3D Modeling Application

## Overview

PyQt6 JPatch is a modern 3D modeling and animation application built with PyQt6 and OpenGL, inspired by the classic JPatch 3D modeler. The application provides a comprehensive suite of 3D modeling tools with an intuitive graphical user interface and integrates ROCA (Relational Orbital Capsule Architecture) for AI-assisted modeling and animation through capsule-based knowledge representation.

## Architecture

### Core Components

- **MainWindow**: The primary application window containing menus, toolbars, and layout management
- **GLWidget**: OpenGL-based 3D viewport widget for rendering 3D scenes
- **ViewportManager**: Manages multiple viewport configurations (single, quad, split views)
- **Model**: Core data model containing control points, patches, curves, bones, and animation data
- **PluginManager**: Extensible plugin system for additional functionality

### GUI Structure

```
MainWindow (1600x1000)
├── Menu Bar (File, Edit, View, Model, Bones, Animation, Settings, Help, Plugins)
├── Toolbars (Main, Mesh, Bones, View)
├── Central Widget
│   ├── Main Splitter
│   │   ├── Viewport Widget (ViewportManager with GLWidget instances)
│   │   └── Sidebar
│   │       ├── Model Tree (QTreeWidget)
│   │       ├── Control Points Browser (QTreeWidget)
│   │       ├── Capsule Tree Browser (QTreeWidget)
│   │       └── Orbital View Widget
├── Orbital View & Chat (Dock Widget - Right, 400x500 min)
│   ├── Orbital View Widget (70% height)
│   └── Chat Widget (30% height)
└── Animation Timeline (Dock Widget - Bottom, 150px min height)
```

## Key Features

### 3D Rendering & Viewport
- **OpenGL-based rendering** with multiple viewport modes (perspective, top, front, side)
- **Real-time 3D display** with lighting, shading, and texture support
- **Camera controls** including orbit, pan, zoom, and orthographic views
- **Grid system** with configurable size and subdivisions
- **Rotoscoping support** with image overlay and opacity control

### Modeling Tools
- **Control points** for creating 3D geometry
- **Bezier patches** and **B-Spline curves** for smooth surfaces
- **Subdivision surfaces** (Catmull-Clark, Loop subdivision)
- **Bone system** for skeletal animation and rigging
- **Morph targets** for blend shape animation

### File I/O
- **Native JPatch format** (.jpt) for saving/loading projects
- **Import support**: Wavefront OBJ, 3D Studio (.3ds), FBX, Animation:Master (.mdl), **BVH motion capture (.bvh)**
- **Export support**: Wavefront OBJ, FBX, 3D Studio, Animation:Master MDL, POV-Ray, RenderMan RIB

### Animation System
- **Keyframe animation** with bone transformations
- **Inverse kinematics (IK)** solver for pose manipulation
- **Morph target blending** for facial animation
- **Advanced Timeline widget** with multi-selection and ROCA-powered in-between generation
- **Capsule-based animation** with drag-and-drop character and motion capsules
- **Pose capture** for creating motion capsules from current frames
- **Automatic capsule merging** when character and animation capsules are combined in timeline

### User Interface
- **Comprehensive menu system** with keyboard shortcuts
- **Contextual toolbars** for quick access to common functions
- **Model hierarchy browser** showing bones, morphs, materials, and geometry
- **Control point inspector** for precise coordinate editing
- **Capsule tree browser** for managing ROCA knowledge capsules
- **Orbital view widget** for exploring capsule relationships
- **Network chat interface** for interactive capsule creation and AI assistance
- **Advanced animation timeline** with multi-selection and AI assistance
- **Plugin manager dialog** for extending functionality

## Technical Implementation

### Dependencies
- **PyQt6**: GUI framework and OpenGL widget integration
- **PyOpenGL**: OpenGL bindings for 3D rendering
- **NumPy**: Mathematical operations and vector calculations
- **Custom math3d module**: 3D math utilities (vectors, matrices, transformations)

### OpenGL Rendering Pipeline
1. **Initialization** (`initializeGL`): Set up OpenGL context, lighting, depth testing
2. **Scene rendering** (`paintGL`): Clear buffers, set camera, render geometry
3. **Viewport updates** (`resizeGL`): Handle window resizing and aspect ratio

### Drawing Methods
- `draw_grid()`: Coordinate system visualization
- `draw_rotoscope()`: Background image overlay
- `draw_shaded()`: Lit surface rendering
- `draw_patches()`: Bezier patch tessellation and display
- `draw_curves()`: Curve geometry rendering
- `draw_points()`: Control point visualization
- `draw_bones()`: Skeletal structure display
- `draw_subdivision_surfaces()`: Subdivision surface rendering

### Lighting Modes
- **Head Light**: Directional lighting from camera position
- **Simple Light**: Basic directional illumination
- **Three-Point Lighting**: Key, fill, and rim lights
- **Sticky Light**: Light attached to camera position

## Code Structure

### Main Classes

#### GLWidget (QOpenGLWidget)
```python
class GLWidget(QOpenGLWidget):
    def __init__(self, model, parent=None):
        # Initialize viewport settings, camera, lighting

    def initializeGL(self):
        # OpenGL context setup

    def paintGL(self):
        # Main rendering loop

    def resizeGL(self, w, h):
        # Viewport resizing

    # Camera and drawing methods...
```

#### MainWindow (QMainWindow)
```python
class MainWindow(QMainWindow):
    def __init__(self):
        # GUI setup, model initialization, plugin loading

    def create_menu_bar(self):
        # Menu construction

    def create_toolbars(self):
        # Toolbar creation with object names

    # Event handlers and business logic...
```

#### Model Class
- Manages control points, patches, curves, bones, materials
- Handles file I/O operations
- Provides import/export functionality

#### ViewportManager
- Manages multiple GLWidget instances
- Handles viewport layout configurations
- Coordinates updates across viewports

### Plugin System
- **PluginManager**: Loads and manages Python plugin modules
- **PluginDialog**: GUI for plugin management
- Extensible architecture for custom tools and importers

## Development Status

### Completed Features
- ✅ **Core 3D viewport** with OpenGL rendering
- ✅ **Model import/export** (OBJ, 3DS, FBX, MDL)
- ✅ **BVH motion capture import** with automatic animation capsule creation
- ✅ **GUI framework** with menus and toolbars
- ✅ **Bone animation system** with IK solver
- ✅ **Subdivision surfaces** and Bezier patches
- ✅ **Plugin architecture** for extensibility
- ✅ **Settings persistence** and window state management
- ✅ **Advanced animation timeline** with ROCA integration
- ✅ **Capsule-based AI assistance** for modeling and animation
- ✅ **Automatic capsule merging** for character+animation combinations in timeline
- ✅ **ROCA Functional Lanes** with 7 orbital bands for capsule organization
- ✅ **Dynamic Gravity System** with time-based orbital drift and smooth animations
- ✅ **Auto-Proposal System** for automatic capsule creation from usage patterns
- ✅ **Shadow Identity Merging** with reversible capsule coalescing and lineage preservation

### Known Limitations
- Some menu actions are placeholders ("Not implemented yet")
- Limited texture and material support
- Undo/redo system not fully implemented

## Usage

### Running the Application
```bash
cd pyqt6_jpatch
python main.py
```

The application opens in a 1600x1000 window with all components properly sized and visible.

### Basic Workflow
1. **Create new model** or **import existing** 3D file
2. **Add control points** to define geometry
3. **Create patches/curves** from control points
4. **Set up bones** for animation rigging
5. **Create keyframes** for animation in the timeline
6. **Drag character/motion capsules** onto timeline frames
7. **Use multi-selection** to choose keyframes for in-between generation
8. **Generate AI-powered in-betweens** using ROCA capsule interpolation
9. **Capture poses** as reusable motion capsules
10. **Combine character and animation capsules** in timeline for automatic merged capsules (e.g., "Batman_walk")
11. **Chat with ROCA** to create new capsules: "Create a walking motion" or "Make a fantasy character"
12. **Export** to desired format

### Viewport Controls
- **Mouse drag**: Orbit camera (perspective) or pan (orthographic)
- **Mouse wheel**: Zoom in/out
- **View menu**: Switch between viewport modes
- **Display toggles**: Show/hide points, curves, patches, grid, bones

## Future Development

### Planned Enhancements
- Complete undo/redo system implementation
- Advanced material and texture system
- Python scripting interface
- More import/export formats
- Performance optimizations for large models
- Enhanced ROCA capsule types and AI features

### Potential Extensions
- NURBS surface support
- Particle systems
- Physics simulation
- Rendering integration
- Collaborative editing features

## ROCA Integration

PyQt6 JPatch includes integration with the ROCA (Relational Orbital Capsule Architecture) system, providing AI-assisted 3D modeling and animation capabilities through capsule-based knowledge representation.

### ROCA Components

- **Capsules**: Knowledge units representing concepts, characters, skills, styles, 3D models, and other entities with vector embeddings for similarity matching
- **Graph Manager**: Symbolic graph structure for managing capsule relationships and similarity edges
- **Migration Manager**: Version control system for ROCA graphs with explicit schema migrations
- **Orbital Visualization**: Interactive orbital view for exploring capsule relationships and properties

### Capsule Types

- **Character**: Representing animated characters with metadata and references
- **Skill**: Animation techniques and motion patterns
- **Style**: Visual and animation styles that can be applied to models
- **3D Model**: Stored 3D model data with similarity search capabilities
- **2D Image**: Image assets with vector embeddings
- **Animation**: Motion capture data and animation sequences (from BVH import)
- **Character_Animation**: Merged capsules combining character and animation data (e.g., "Batman_walk")
- **Personality**: Character personality traits for procedural animation
- **Unassigned**: Generic capsules for custom use

### ROCA Features in JPatch

#### Capsule Management
- **Capsule Tree Browser**: Hierarchical view of all capsules organized by type
- **Orbital View Widget**: Interactive orbital visualization showing capsule relationships with **functional lanes**
- **Functional Lanes**: Capsules organized in 7 distinct orbital bands (Core, Characters, Styles, Skills, Memories, Workflows, Experimental)
- **Dynamic Gravity System**: Capsules drift inward/outward based on usage frequency with smooth sigmoid-based positioning
- **Auto-Proposal System**: Automatic creation of new capsules from repeated usage patterns (≥3 repeats)
- **Shadow Identity Merging**: Reversible capsule coalescing that preserves original identities as shadows
- **Drag-and-Drop Integration**: Capsules can be dragged from the tree to the timeline
- **Similarity Search**: Find similar capsules using vector embeddings
- **Capsule Creation**: Automatic creation of capsules from 3D models and other assets

#### Timeline Integration
- **Character References**: Drop character capsules onto timeline frames to associate characters
- **Skill Application**: Apply animation skills and techniques to timeline segments
- **Style Application**: Apply visual styles to animations
- **Motion Capsules**: Attach motion capture data to frames for procedural animation
- **Animation Capsules**: Drop BVH-derived animation capsules onto timeline frames
- **Automatic Capsule Merging**: Combine character and animation capsules in same frame to create merged capsules (e.g., "Batman_walk")
- **Multi-Selection**: Select multiple keyframes for batch operations
- **ROCA In-between Generation**: AI-powered creation of intermediate frames between keyframes
- **Pose Capture**: Create motion capsules from current frame poses
- **Keyframe Enhancement**: Use capsule metadata to enhance animation keyframes

#### Graph Operations
- **Capsule Addition**: Add new capsules to the ROCA graph
- **Similarity Linking**: Create similarity edges between related capsules
- **In-between Creation**: Generate intermediate capsules by merging properties
- **Capsule Merging**: Automatically combine character and animation capsules into composite capsules
- **Frame Attachment**: Attach capsules to specific timeline frames
- **Pose Capture**: Create motion capsules from animation frames
- **Graph Migration**: Handle version upgrades of ROCA graph schemas

#### Orbital Visualization
- **Interactive Orbits**: Capsules orbit at distances based on their properties within functional lanes
- **Functional Lanes**: 7 distinct orbital bands for different capsule types with visual ring separators
- **Dynamic Positioning**: Capsules positioned within lanes based on gravity (usage frequency)
- **Zoom and Pan**: Navigate the orbital space
- **Selection and Hover**: Interact with individual capsules
- **Network Overlays**: Visualize active connections and network activity
- **Real-time Updates**: Orbital positions update dynamically

#### Network Chat Interface
- **Integrated Chat Widget**: Text-based interaction with ROCA system below orbital view
- **Capsule Creation**: Convert chat messages into knowledge capsules automatically
- **Smart Type Inference**: Automatically determines capsule type (character, motion, style, etc.) from message content
- **Real-time Integration**: New capsules appear immediately in orbital view and capsule tree
- **Contextual Responses**: AI assistant provides guidance and acknowledges capsule creation

### ROCA Orbital Chat (Standalone)
- **Chat Interface**: Text-based interaction with ROCA system
- **Orbital Visualization**: Real-time orbital view of conversation capsules
- **Capsule Generation**: Convert chat messages into capsules
- **Threaded Processing**: Background message handling to avoid UI blocking

### Advanced Timeline Features

The JPatch timeline now provides sophisticated animation tools with ROCA AI integration:

#### Multi-Keyframe Operations
- **Multi-Selection**: Ctrl+click to select multiple keyframes
- **Batch In-between Generation**: Generate smooth transitions between selected keyframes
- **ROCA-Powered Interpolation**: AI creates natural motion capsules for intermediate frames

#### Capsule-Driven Animation
- **Character Capsules**: Drag characters onto frames to associate with keyframes
- **Motion Capsules**: Attach motion capture data for procedural animation
- **Pose Capture**: "Create Pose" button captures current frame as reusable motion capsule
- **Visual Feedback**: Timeline shows capsule attachments with colored indicators

#### AI-Assisted Workflow
- **Smart In-betweens**: ROCA analyzes keyframe capsules and generates intermediate motion
- **Capsule Reuse**: Generated in-betweens become new capsules for future use
- **Graph Integration**: All timeline operations update the ROCA knowledge graph
- **Similarity Matching**: Find and reuse similar motion patterns from capsule database

### Technical Integration

#### Capsule Data Structure
```python
class Capsule:
    name: str
    type: str
    vector: np.ndarray  # 32D embedding
    metadata: Dict
    orbit_distance: float
    usage_count: int
    orbit_score: float
```

#### Graph Management
```python
class ROCAGraph:
    capsules: Dict[str, Capsule]
    similarity_edges: Dict[str, List[str]]

class GraphManager:
    def execute(command: Command) -> Result
```

#### Commands
- `ADD_CAPSULE`: Add new capsule to graph
- `RENAME_CAPSULE`: Rename existing capsule
- `INSERT_INBETWEEN`: Create merged intermediate capsule
- `MERGE_CAPSULES`: Combine multiple capsules into composite capsule (e.g., character + animation)
- `ATTACH_TO_FRAME`: Link capsule to timeline frame

### Future ROCA Enhancements

- **GNN Integration**: Graph Neural Network processing for capsule relationships
- **Procedural Generation**: Generate 3D content from capsule combinations
- **AI-Assisted Modeling**: Use capsule embeddings for intelligent suggestions
- **Multi-modal Capsules**: Support for audio, video, and other media types
- **Collaborative Features**: Shared capsule databases and graph synchronization

---

*Project Status: Advanced 3D Modeler with AI Integration*
*Last Updated: January 4, 2026*
*Technology: PyQt6 + OpenGL + ROCA AI System*

*ROCA Integration Status: Active Development*
*Graph Version: 0.1.0*
*Capsule Types: 8 Supported*