
# ============================================================================
# IMPORTS
# ============================================================================

import time
import uuid
import numpy as np
import json
import logging
import os
import math
from typing import Dict, List, Optional, Any, Tuple
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont
from PyQt6.QtCore import QSize, Qt, QObject, pyqtSignal, QPointF, QRectF, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

# Initialize logger
logger = logging.getLogger("Model")

# ============================================================================
# CONSTANTS
# ============================================================================

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# ============================================================================
# DATA MODELS
# ============================================================================

class Capsule:
    """Enhanced Capsule class with Qt compatibility."""

    def __init__(self, name: str, capsule_type: str, vector_dim: int = 32, 
                 metadata: Dict = None, uuid_str: str = None, 
                 created_timestamp: float = None):
        self.name = name
        self.type = capsule_type
        self.vector = np.random.randn(vector_dim)
        self.metadata = metadata.copy() if metadata else {}
        self.orbit_distance = 1.0
        self.usage_count = 0
        self.orbit_score = 0
        self.last_used_time = time.time()
        self.pinned_to_core = False
        self.uuid = uuid_str or str(uuid.uuid4())
        self.created_timestamp = created_timestamp or time.time()
        self.angle = 0.0  # For orbital visualization

        # Ensure metadata consistency
        self.metadata['uuid'] = self.uuid
        self.metadata['created_timestamp'] = self.created_timestamp
        self.metadata['name'] = self.name
        self.metadata['type'] = self.type

        # Thumbnail cache for UI
        self._thumbnail = None
        self._thumbnail_path = None

    def clear_thumbnail_cache(self):
        """Explicitly clear the thumbnail cache to free memory."""
        self._thumbnail = None
        
    def pin_to_core(self):
        self.pinned_to_core = True
        self.orbit_distance = 0.0
        
    def set_orbit_distance(self, distance: float):
        if self.pinned_to_core:
            self.orbit_distance = 0.0
            return
        self.orbit_distance = float(distance)
        
    def get_thumbnail(self, size: QSize = None) -> Optional[QPixmap]:
        """Get cached thumbnail, loading if necessary."""
        if self._thumbnail is None and self._thumbnail_path:
            try:
                pixmap = QPixmap(self._thumbnail_path)
                if not pixmap.isNull():
                    if size:
                        pixmap = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                    self._thumbnail = pixmap
            except:
                pass
        return self._thumbnail
    
    def set_thumbnail_path(self, path: str):
        self._thumbnail_path = path
        self._thumbnail = None  # Clear cache
        
    def to_dict(self) -> Dict:
        """Serialize capsule to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'vector': self.vector.tolist(),
            'metadata': self.metadata,
            'orbit_distance': self.orbit_distance,
            'orbit_score': self.orbit_score,
            'usage_count': self.usage_count,
            'last_used_time': self.last_used_time,
            'pinned_to_core': self.pinned_to_core,
            'uuid': self.uuid,
            'created_timestamp': self.created_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Capsule':
        """Create capsule from dictionary."""
        # Handle both 'type' and 'capsule_type' keys for compatibility
        capsule_type = data.get('type') or data.get('capsule_type', 'unknown')
        
        capsule = cls(
            name=data['name'],
            capsule_type=capsule_type,
            vector_dim=len(data['vector']),
            metadata=data.get('metadata', {}),
            uuid_str=data.get('uuid'),
            created_timestamp=data.get('created_timestamp')
        )
        capsule.vector = np.array(data['vector'])
        capsule.orbit_distance = data.get('orbit_distance', 1.0)
        capsule.orbit_score = data.get('orbit_score', 0)
        capsule.usage_count = data.get('usage_count', 0)
        capsule.last_used_time = data.get('last_used_time', time.time())
        capsule.pinned_to_core = data.get('pinned_to_core', False)
        if capsule.pinned_to_core:
            capsule.pin_to_core()
        return capsule
    
    @property
    def id(self):
        """Alias for uuid for compatibility."""
        return self.uuid



class CapsuleManager(QObject):
    """Manages all capsules with Qt signals for UI updates."""

    capsules_changed = pyqtSignal()
    capsule_added = pyqtSignal(object)  # Capsule object
    capsule_removed = pyqtSignal(object)  # Capsule object
    capsule_updated = pyqtSignal(object)  # Capsule object

    def __init__(self, brain=None):
        super().__init__()
        super().__init__()
        self.capsules = []  # type: List[Capsule]
        self.capsule_map = {}  # type: Dict[str, Capsule]  # UUID -> Capsule
        self.coactivation_counts = {}  # type: Dict[str, int]  # "uuid1|uuid2" -> count

        # Auto-save settings
        self.auto_save_enabled = True
        self.capsule_store_path = os.path.join(os.path.dirname(__file__), "capsule_store", "capsules.json")

        # Integrate creative consciousness systems (optional)
        try:
            from Creative_conciousness import MemoryConsolidationEngine, TeachingEngine, StoryGenerator
            self.memory_engine = MemoryConsolidationEngine(self)
            self.teaching_engine = TeachingEngine(brain)
            self.story_generator = StoryGenerator(self)
            self.advanced_features_available = True
        except ImportError:
            # Fallback to basic functionality
            self.memory_engine = None
            self.teaching_engine = None
            self.story_generator = None
            self.advanced_features_available = False
            logger.warning("Advanced AI features not available - Creative_conciousness module not found")

        # Create default capsules (only if no auto-save exists)
        if not os.path.exists(self.capsule_store_path):
            self.create_default_capsules()
        
        # Try to load auto-saved capsules
        self._load_auto_save()

        # Dynamic gravity system
        self.gravity_alpha = 0.12  # Controls sigmoid steepness
        self.decay_interval_days = 7  # Decay every 7 days
        self.last_decay_time = time.time()
        self.gravity_decay_timer = QTimer(self)
        self.gravity_decay_timer.timeout.connect(self._decay_tick)
        self.gravity_decay_timer.start(24 * 60 * 60 * 1000)  # 24 hours in milliseconds

        # Auto-proposal system
        self.pattern_window_days = 30  # Look back 30 days for patterns
        self.min_pattern_repeats = 3   # Need at least 3 repeats to propose
        self.dedupe_threshold = 0.8    # Similarity threshold for deduplication
        self.usage_patterns = {}       # Track patterns: pattern_key -> {'count': int, 'last_seen': float, 'capsules': set}
        self.auto_proposal_timer = QTimer(self)
        self.auto_proposal_timer.timeout.connect(self._check_auto_proposals)
        self.auto_proposal_timer.start(60 * 60 * 1000)  # Check every hour

        # Initialize merge system
        self.__init_merge_system()

    def calculate_gravity(self, capsule: Capsule) -> float:
        """Calculate gravity score for a capsule using sigmoid function."""
        # g(c) = σ(α * orbitScore_c) where σ is sigmoid
        orbit_score = getattr(capsule, 'orbit_score', 0)
        gravity = 1.0 / (1.0 + math.exp(-self.gravity_alpha * float(orbit_score)))
        return gravity

    def update_capsule_gravity(self, capsule: Capsule):
        """Update a capsule's gravity and trigger UI updates."""
        old_gravity = getattr(capsule, 'current_gravity', 0.5)
        new_gravity = self.calculate_gravity(capsule)
        capsule.current_gravity = new_gravity

        # Trigger UI update if gravity changed significantly
        if abs(new_gravity - old_gravity) > 0.01:
            self.capsule_updated.emit(capsule)

    def _decay_tick(self):
        """Perform decay tick on all capsules (reduce orbit scores over time)."""
        current_time = time.time()
        days_since_last_decay = (current_time - self.last_decay_time) / (24 * 60 * 60)

        if days_since_last_decay >= self.decay_interval_days:
            logger.info(f"Performing gravity decay tick ({len(self.capsules)} capsules)")

            for capsule in self.capsules:
                if not capsule.pinned_to_core:  # Don't decay pinned capsules
                    # Reduce orbit score (but don't go below -10 to prevent complete decay)
                    old_score = getattr(capsule, 'orbit_score', 0)
                    new_score = max(-10, old_score - 1)
                    capsule.orbit_score = new_score

                    # Update gravity
                    self.update_capsule_gravity(capsule)

            self.last_decay_time = current_time
            logger.info("Gravity decay tick completed")

    def record_usage(self, capsule: Capsule, amount: int = 1):
        """Record usage of a capsule and update gravity."""
        capsule.usage_count += amount
        capsule.orbit_score += amount
        capsule.last_used_time = time.time()

        # Update gravity immediately
        self.update_capsule_gravity(capsule)

        # Track usage patterns for auto-proposal
        self._track_usage_pattern(capsule)

        self.capsule_updated.emit(capsule)

        # Auto-save if enabled
        if self.auto_save_enabled:
            self._auto_save()
        if not self.advanced_features_available or not self.memory_engine:
            return {"status": "basic_mode", "message": "Advanced memory consolidation not available"}
        
        if deep_sleep:
            return self.memory_engine.deep_sleep_processing()
        else:
            # Regular consolidation - could be lighter processing
            return self.memory_engine.deep_sleep_processing()

    def get_teaching_moment(self):
        """Get a teaching moment if available."""
        if not self.advanced_features_available or not self.teaching_engine:
            return None
        return self.teaching_engine.analyze_for_teaching_moments()

    def generate_story(self, capsule_ids=None, genre=None):
        """Generate a story using the story generator."""
        if not self.advanced_features_available or not self.story_generator:
            return {"error": "Story generation not available - advanced AI features not loaded"}
        return self.story_generator.generate_story(capsule_ids, genre)

    def clear_all_thumbnail_caches(self):
        """Clear thumbnail caches for all capsules."""
        for capsule in self.capsules:
            capsule.clear_thumbnail_cache()
        logger.info("All capsule thumbnail caches cleared.")
        # cleanup_memory("After clearing all capsule thumbnails")

    def create_default_capsules(self):
        """Create initial capsules."""
        # Core personality
        personality = Capsule(
            name="CorePersonality",
            capsule_type="personality",
            vector_dim=64,
            metadata={
                "traits": ["creative", "helpful", "artistic"],
                "description": "Central personality for guiding interactions",
                "color": "#1E90FF"
            }
        )
        personality.pin_to_core()
        self.add_capsule(personality)
        
        # Example character
        batman = Capsule(
            name="Batman",
            capsule_type="character",
            vector_dim=32,
            metadata={
                "description": "Dark knight detective",
                "style": "comic book",
                "skeleton": {
                    "head": [400, 100],
                    "shoulder": [400, 200],
                    "hip": [400, 350]
                }
            }
        )
        batman.set_orbit_distance(0.5)
        self.add_capsule(batman)
        
        # Example skill
        walk_cycle = Capsule(
            name="WalkCycle",
            capsule_type="skill",
            vector_dim=16,
            metadata={
                "description": "Basic walking animation",
                "tags": ["motion", "cycle", "locomotion"]
            }
        )
        walk_cycle.set_orbit_distance(1.0)
        self.add_capsule(walk_cycle)
        
    def add_capsule(self, capsule: Capsule):
        """Add a capsule to the manager."""
        self.capsules.append(capsule)
        self.capsule_map[capsule.uuid] = capsule
        logger.info(f"Capsule added: {capsule.name} (type: {capsule.type}, uuid: {capsule.uuid})")
        try:
            log_memory_usage(f"After adding capsule {capsule.name}")
        except NameError:
            # log_memory_usage not available in this context
            pass
        self.capsules_changed.emit()
        self.capsule_added.emit(capsule)
        
        # Auto-save if enabled
        if self.auto_save_enabled:
            self._auto_save()
        
    def remove_capsule(self, capsule: Capsule):
        """Remove a capsule from the manager."""
        if capsule in self.capsules:
            self.capsules.remove(capsule)
            if capsule.uuid in self.capsule_map:
                del self.capsule_map[capsule.uuid]
            capsule.clear_thumbnail_cache()
            logger.info(f"Capsule removed: {capsule.name} (uuid: {capsule.uuid})")
            # cleanup_memory(f"After removing capsule {capsule.name}")
            self.capsules_changed.emit()
            self.capsule_removed.emit(capsule)
            
            # Auto-save if enabled
            if self.auto_save_enabled:
                self._auto_save()
            
    def get_capsule_by_uuid(self, uuid_str: str) -> Optional[Capsule]:
        """Get capsule by UUID."""
        return self.capsule_map.get(uuid_str)
    
    def get_capsules_by_type(self, capsule_type: str) -> List[Capsule]:
        """Get all capsules of a specific type."""
        return [c for c in self.capsules if c.type.lower() == capsule_type.lower()]
        
    def update_orbit(self, capsule: Capsule):
        """Update capsule orbit distance based on score."""
        if capsule.pinned_to_core:
            capsule.pin_to_core()
            return
            
        min_r = 0.5
        max_r = 2.0
        alpha = 0.12
        
        score = capsule.orbit_score
        g = 1.0 / (1.0 + math.exp(-alpha * float(score)))
        dist = float(min_r + (1.0 - g) * (max_r - min_r))
        capsule.set_orbit_distance(dist)
        
    def record_coactivation(self, capsules: List[Capsule]):
        """Record coactivation between capsules."""
        ids = [c.uuid for c in capsules]
        n = len(ids)
        for i in range(n):
            for j in range(i+1, n):
                key = f"{ids[i]}|{ids[j]}" if ids[i] < ids[j] else f"{ids[j]}|{ids[i]}"
                self.coactivation_counts[key] = self.coactivation_counts.get(key, 0) + 1
                
    def save_to_file(self, filepath: str):
        """Save all capsules to JSON file."""
        data = {
            'capsules': [c.to_dict() for c in self.capsules],
            'coactivations': self.coactivation_counts,
            'version': '1.0',
            'saved_at': time.time()
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Capsules saved to file: {filepath}")
            try:
                log_memory_usage(f"After saving capsules to {filepath}")
            except NameError:
                # log_memory_usage not available in this context
                pass
            return True
        except Exception as e:
            logger.error(f"Error saving capsules: {e}")
            return False
    
    def _auto_save(self):
        """Auto-save capsules to default location."""
        try:
            # Ensure capsule_store directory exists
            os.makedirs(os.path.dirname(self.capsule_store_path), exist_ok=True)
            self.save_to_file(self.capsule_store_path)
            logger.debug("Auto-saved capsules to default location")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    def _load_auto_save(self):
        """Load auto-saved capsules on startup."""
        if os.path.exists(self.capsule_store_path):
            try:
                self.load_from_file(self.capsule_store_path)
                logger.info("Loaded auto-saved capsules")
            except Exception as e:
                logger.error(f"Failed to load auto-saved capsules: {e}")
            
    def load_from_file(self, filepath: str):
        """Load capsules from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Clear existing capsules (except core personality)
            core_personality = next((c for c in self.capsules if c.name == "CorePersonality"), None)
            self.capsules = [core_personality] if core_personality else []
            self.capsule_map = {c.uuid: c for c in self.capsules}
            # Load capsules
            for cap_data in data.get('capsules', []):
                if cap_data.get('name') == "CorePersonality" and core_personality:
                    # Update existing core personality
                    core_personality.metadata.update(cap_data.get('metadata', {}))
                    core_personality.orbit_score = cap_data.get('orbit_score', 0)
                    core_personality.usage_count = cap_data.get('usage_count', 0)
                else:
                    capsule = Capsule.from_dict(cap_data)
                    self.add_capsule(capsule)
            # Load coactivations
            self.coactivation_counts = data.get('coactivations', {})
            self.capsules_changed.emit()
            logger.info(f"Capsules loaded from file: {filepath}")
            # cleanup_memory(f"After loading capsules from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading capsules: {e}")
            return False

    # ============================================================================
    # AUTO-PROPOSAL SYSTEM
    # ============================================================================

    def _track_usage_pattern(self, capsule: Capsule):
        """Track usage patterns for auto-proposal system."""
        current_time = time.time()

        # Create pattern key based on capsule type and recent context
        pattern_key = f"{capsule.type}:{capsule.name}"

        if pattern_key not in self.usage_patterns:
            self.usage_patterns[pattern_key] = {
                'count': 0,
                'last_seen': current_time,
                'capsules': set([capsule.uuid])
            }

        pattern = self.usage_patterns[pattern_key]
        pattern['count'] += 1
        pattern['last_seen'] = current_time
        pattern['capsules'].add(capsule.uuid)

        # Clean old patterns (older than pattern_window_days)
        cutoff_time = current_time - (self.pattern_window_days * 24 * 60 * 60)
        old_patterns = [k for k, v in self.usage_patterns.items() if v['last_seen'] < cutoff_time]
        for k in old_patterns:
            del self.usage_patterns[k]

    def _check_auto_proposals(self):
        """Check for patterns that should trigger auto-proposal of new capsules."""
        current_time = time.time()
        cutoff_time = current_time - (self.pattern_window_days * 24 * 60 * 60)

        # Look for patterns that meet the threshold (create a copy to avoid modification during iteration)
        patterns_to_check = list(self.usage_patterns.items())

        for pattern_key, pattern_data in patterns_to_check:
            if (pattern_data['count'] >= self.min_pattern_repeats and
                pattern_data['last_seen'] >= cutoff_time):

                # Check if we should propose a new capsule for this pattern
                if self._should_propose_capsule(pattern_key, pattern_data):
                    self._propose_capsule(pattern_key, pattern_data)

    def _should_propose_capsule(self, pattern_key: str, pattern_data: Dict) -> bool:
        """Determine if a pattern warrants proposing a new capsule."""
        pattern_type, pattern_name = pattern_key.split(':', 1)

        # Get existing capsules of this type
        existing_capsules = self.get_capsules_by_type(pattern_type)

        # Check similarity to existing capsules
        for existing in existing_capsules:
            similarity = self._calculate_pattern_similarity(pattern_data, existing)
            if similarity >= self.dedupe_threshold:
                # Too similar to existing capsule, don't propose
                return False

        return True

    def _calculate_pattern_similarity(self, pattern_data: Dict, existing_capsule: Capsule) -> float:
        """Calculate similarity between a usage pattern and existing capsule."""
        # Simple similarity based on name overlap and usage patterns
        pattern_key = list(pattern_data['capsules'])[0]  # Get the pattern key from capsules
        pattern_type, pattern_name = pattern_key.split(':', 1) if ':' in pattern_key else (pattern_key, pattern_key)

        # Name similarity (basic string matching)
        name_similarity = 1.0 if pattern_name.lower() in existing_capsule.name.lower() else 0.0

        # Usage pattern similarity
        usage_ratio = min(pattern_data['count'], existing_capsule.usage_count) / max(pattern_data['count'], existing_capsule.usage_count)

        # Combine factors (weighted average)
        return 0.6 * name_similarity + 0.4 * usage_ratio

    def _propose_capsule(self, pattern_key: str, pattern_data: Dict):
        """Propose and create a new capsule based on detected pattern."""
        pattern_type, base_name = pattern_key.split(':', 1)

        # Generate a unique name for the new capsule
        existing_names = {c.name for c in self.get_capsules_by_type(pattern_type)}
        new_name = base_name
        counter = 1
        while new_name in existing_names:
            new_name = f"{base_name}_{counter}"
            counter += 1

        # Determine vector dimensions based on capsule type
        vector_dims = {
            'character': 32,
            'style': 16,
            'skill': 16,
            'memory': 32,
            'workflow': 32,
            'experimental': 16
        }.get(pattern_type, 16)

        # Create metadata based on pattern
        metadata = {
            'auto_proposed': True,
            'pattern_repeats': pattern_data['count'],
            'source_capsules': list(pattern_data['capsules']),
            'proposed_at': time.time(),
            'description': f"Auto-proposed {pattern_type} capsule from repeated usage pattern"
        }

        # Create the new capsule
        new_capsule = Capsule(
            name=new_name,
            capsule_type=pattern_type,
            vector_dim=vector_dims,
            metadata=metadata
        )

        # Set initial orbit score low (outer orbit)
        new_capsule.orbit_score = -5  # Start in outer orbit
        new_capsule.usage_count = 1   # Initial usage

        # Add the capsule
        self.add_capsule(new_capsule)

        logger.info(f"Auto-proposed new capsule: {new_name} (type: {pattern_type}) from pattern '{pattern_key}'")

        # Remove the pattern so we don't keep proposing the same thing
        if pattern_key in self.usage_patterns:
            del self.usage_patterns[pattern_key]

    def __init_merge_system(self):
        """Initialize the merge system parameters."""
        self.merge_similarity_threshold = 0.85  # Similarity threshold for merge candidates
        self.merge_coactivation_threshold = 3   # Minimum co-activations to consider merge
        self.merge_conflict_threshold = 0.3     # Maximum conflict ratio allowed
        self.merge_check_interval_hours = 24    # Check for merges daily
        self.merge_timer = QTimer(self)
        self.merge_timer.timeout.connect(self._check_for_merges)
        self.merge_timer.start(self.merge_check_interval_hours * 60 * 60 * 1000)  # Daily checks

    # ============================================================================
    # COALESCING SYSTEM (MERGE WITH SHADOW IDENTITIES)
    # ============================================================================

    def __init_merge_system(self):
        """Initialize the merge system parameters."""
        self.merge_similarity_threshold = 0.85  # Similarity threshold for merge candidates
        self.merge_coactivation_threshold = 3   # Minimum co-activations to consider merge
        self.merge_conflict_threshold = 0.3     # Maximum conflict ratio allowed
        self.merge_check_interval_hours = 24    # Check for merges daily
        self.merge_timer = QTimer(self)
        self.merge_timer.timeout.connect(self._check_for_merges)
        self.merge_timer.start(self.merge_check_interval_hours * 60 * 60 * 1000)  # Daily checks

    def _check_for_merges(self):
        """Check for capsules that should be merged based on similarity and co-activation."""
        logger.info("Checking for capsule merge candidates...")

        # Group capsules by type for lane-local merging
        capsules_by_type = {}
        for capsule in self.capsules:
            if capsule.type not in capsules_by_type:
                capsules_by_type[capsule.type] = []
            capsules_by_type[capsule.type].append(capsule)

        merge_candidates_found = 0

        # Check each type separately (lane-local merging)
        for capsule_type, type_capsules in capsules_by_type.items():
            if len(type_capsules) < 2:
                continue

            # Find merge candidates within this type
            candidates = self._find_merge_candidates(type_capsules)
            for candidate_pair in candidates:
                self._execute_merge(candidate_pair[0], candidate_pair[1])
                merge_candidates_found += 1

        if merge_candidates_found > 0:
            logger.info(f"Executed {merge_candidates_found} capsule merges")
        else:
            logger.debug("No merge candidates found")

    def _find_merge_candidates(self, capsules: List[Capsule]) -> List[Tuple[Capsule, Capsule]]:
        """Find capsule pairs that are candidates for merging."""
        candidates = []

        for i, cap1 in enumerate(capsules):
            for cap2 in capsules[i+1:]:
                if self._should_merge_capsules(cap1, cap2):
                    candidates.append((cap1, cap2))

        return candidates

    def _should_merge_capsules(self, cap1: Capsule, cap2: Capsule) -> bool:
        """Determine if two capsules should be merged."""
        # Skip if either is already a merged capsule or a shadow
        if (cap1.metadata.get('is_merged_proxy', False) or
            cap2.metadata.get('is_merged_proxy', False) or
            cap1.metadata.get('is_shadow', False) or
            cap2.metadata.get('is_shadow', False)):
            return False

        # Skip core personality
        if cap1.name == "CorePersonality" or cap2.name == "CorePersonality":
            return False

        # Check similarity
        similarity = self._calculate_capsule_similarity(cap1, cap2)
        if similarity < self.merge_similarity_threshold:
            return False

        # Check co-activation history
        coactivation_count = self.coactivation_counts.get(f"{cap1.uuid}|{cap2.uuid}", 0) + \
                           self.coactivation_counts.get(f"{cap2.uuid}|{cap1.uuid}", 0)
        if coactivation_count < self.merge_coactivation_threshold:
            return False

        # Check for conflicts (if they were used in mutually exclusive contexts)
        conflict_ratio = self._calculate_conflict_ratio(cap1, cap2)
        if conflict_ratio > self.merge_conflict_threshold:
            return False

        return True

    def _calculate_capsule_similarity(self, cap1: Capsule, cap2: Capsule) -> float:
        """Calculate similarity between two capsules."""
        # Vector similarity (cosine similarity) - handle different dimensions
        if cap1.vector is not None and cap2.vector is not None:
            try:
                # If dimensions match, use cosine similarity
                if len(cap1.vector) == len(cap2.vector):
                    dot_product = np.dot(cap1.vector, cap2.vector)
                    norm1 = np.linalg.norm(cap1.vector)
                    norm2 = np.linalg.norm(cap2.vector)
                    if norm1 > 0 and norm2 > 0:
                        vector_similarity = dot_product / (norm1 * norm2)
                    else:
                        vector_similarity = 0.0
                else:
                    # Different dimensions - use a simpler similarity measure
                    # For now, just use 0.5 as a neutral similarity
                    vector_similarity = 0.5
            except:
                vector_similarity = 0.0
        else:
            vector_similarity = 0.0

        # Name similarity
        name1, name2 = cap1.name.lower(), cap2.name.lower()
        name_similarity = 1.0 if name1 in name2 or name2 in name1 else 0.0

        # Usage pattern similarity
        usage_similarity = min(cap1.usage_count, cap2.usage_count) / max(cap1.usage_count, cap2.usage_count) if max(cap1.usage_count, cap2.usage_count) > 0 else 0.0

        # Weighted combination
        return 0.4 * vector_similarity + 0.4 * name_similarity + 0.2 * usage_similarity

    def _calculate_conflict_ratio(self, cap1: Capsule, cap2: Capsule) -> float:
        """Calculate conflict ratio between two capsules."""
        # For now, use a simple heuristic: if they have very different orbit scores,
        # they might be in conflict (one popular, one not)
        score_diff = abs(cap1.orbit_score - cap2.orbit_score)
        max_score = max(abs(cap1.orbit_score), abs(cap2.orbit_score))
        if max_score > 0:
            return score_diff / max_score
        return 0.0

    def _execute_merge(self, cap1: Capsule, cap2: Capsule):
        """Execute the merge of two capsules, preserving shadows."""
        logger.info(f"Merging capsules: '{cap1.name}' and '{cap2.name}'")

        # Create merged proxy capsule
        merged_name = self._generate_merged_name(cap1, cap2)
        merged_vector = (cap1.vector + cap2.vector) / 2  # Average the vectors

        # Determine vector dimension
        vector_dims = {
            'character': 32, 'style': 16, 'skill': 16,
            'memory': 32, 'workflow': 32, 'experimental': 16
        }.get(cap1.type, 16)

        merged_capsule = Capsule(
            name=merged_name,
            capsule_type=cap1.type,
            vector_dim=vector_dims,
            metadata={
                'is_merged_proxy': True,
                'merged_from': [cap1.uuid, cap2.uuid],
                'merge_timestamp': time.time(),
                'merge_confidence': 0.8,  # Initial high confidence
                'description': f"Merged capsule from '{cap1.name}' and '{cap2.name}'"
            }
        )
        merged_capsule.vector = merged_vector

        # Inherit the higher orbit score and usage count
        merged_capsule.orbit_score = max(cap1.orbit_score, cap2.orbit_score)
        merged_capsule.usage_count = cap1.usage_count + cap2.usage_count

        # Mark originals as shadows
        cap1.metadata['is_shadow'] = True
        cap1.metadata['merged_into'] = merged_capsule.uuid
        cap1.metadata['merge_timestamp'] = time.time()

        cap2.metadata['is_shadow'] = True
        cap2.metadata['merged_into'] = merged_capsule.uuid
        cap2.metadata['merge_timestamp'] = time.time()

        # Update merged capsule metadata with shadow references
        merged_capsule.metadata['shadows'] = [cap1.uuid, cap2.uuid]

        # Add the merged capsule
        self.add_capsule(merged_capsule)

        # Update coactivation counts to point to merged capsule
        self._update_coactivations_for_merge(cap1, cap2, merged_capsule)

        logger.info(f"Created merged capsule: '{merged_name}' with shadows '{cap1.name}' and '{cap2.name}'")

    def _generate_merged_name(self, cap1: Capsule, cap2: Capsule) -> str:
        """Generate a name for the merged capsule."""
        # Try to find common prefix
        name1, name2 = cap1.name, cap2.name
        common_prefix = ""
        for i in range(min(len(name1), len(name2))):
            if name1[i] == name2[i]:
                common_prefix += name1[i]
            else:
                break

        if len(common_prefix) > 3:
            return common_prefix.rstrip('_')
        else:
            # Use first capsule's name as base
            return f"{name1}_{name2}"

    def _update_coactivations_for_merge(self, cap1: Capsule, cap2: Capsule, merged: Capsule):
        """Update coactivation counts to reference the merged capsule."""
        # This is a simplified version - in practice, you'd want to migrate
        # all coactivation references to the merged capsule
        pass

    def check_merge_divergence(self, merged_capsule: Capsule):
        """Check if a merged capsule should be split due to divergence."""
        if not merged_capsule.metadata.get('is_merged_proxy', False):
            return False

        shadows = merged_capsule.metadata.get('shadows', [])
        if len(shadows) < 2:
            return False

        # Find shadow capsules
        shadow_capsules = []
        for shadow_uuid in shadows:
            shadow = self.get_capsule_by_uuid(shadow_uuid)
            if shadow:
                shadow_capsules.append(shadow)

        if len(shadow_capsules) < 2:
            return False

        # Check if shadows have diverged significantly
        current_similarity = self._calculate_capsule_similarity(shadow_capsules[0], shadow_capsules[1])
        original_similarity = merged_capsule.metadata.get('original_similarity', self.merge_similarity_threshold)

        # If similarity has dropped significantly, consider splitting
        if current_similarity < original_similarity * 0.7:  # 30% drop
            self._split_merged_capsule(merged_capsule)
            return True

        return False

    def _split_merged_capsule(self, merged_capsule: Capsule):
        """Split a merged capsule back into its shadows."""
        logger.info(f"Splitting merged capsule: '{merged_capsule.name}'")

        shadows = merged_capsule.metadata.get('shadows', [])
        shadow_capsules = []

        # Find and restore shadow capsules
        for shadow_uuid in shadows:
            shadow = self.get_capsule_by_uuid(shadow_uuid)
            if shadow:
                # Remove shadow status
                shadow.metadata.pop('is_shadow', None)
                shadow.metadata.pop('merged_into', None)
                shadow.metadata.pop('merge_timestamp', None)
                shadow_capsules.append(shadow)

        # Remove the merged capsule
        self.remove_capsule(merged_capsule)

        logger.info(f"Restored {len(shadow_capsules)} shadow capsules from merged capsule")


class AnimationFrame:
    """Represents a single frame in an animation."""
    
    def __init__(self, index: int, image: QImage = None):
        self.index = index
        self.image = image or QImage(CANVAS_WIDTH, CANVAS_HEIGHT, QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.transparent)
        self.capsule_refs: List[str] = []  # UUIDs of related capsules
        self.timestamp = time.time()
        self.notes = ""
        
    def get_thumbnail(self, size: QSize) -> QPixmap:
        """Get thumbnail of frame."""
        pixmap = QPixmap.fromImage(self.image)
        return pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, 
                            Qt.TransformationMode.SmoothTransformation)

class Animation(QObject):
    """Manages animation frames and timeline."""

    frames_changed = pyqtSignal()
    current_frame_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.frames: List[AnimationFrame] = [AnimationFrame(0)]
        self.current_frame_index = 0
        self.frame_rate = 24  # FPS
        self.name = "Untitled Animation"
        
    @property
    def current_frame(self) -> AnimationFrame:
        return self.frames[self.current_frame_index]
        
    def add_frame(self, image: QImage = None, at_index: int = None):
        """Add a new frame."""
        if at_index is None:
            at_index = self.current_frame_index + 1
        index = min(max(0, at_index), len(self.frames))
        frame = AnimationFrame(index, image)
        self.frames.insert(index, frame)
        self.renumber_frames()
        self.frames_changed.emit()
        
    def remove_frame(self, index: int):
        """Remove a frame."""
        if 0 <= index < len(self.frames) and len(self.frames) > 1:
            del self.frames[index]
            self.renumber_frames()
            if self.current_frame_index >= len(self.frames):
                self.current_frame_index = len(self.frames) - 1
            self.frames_changed.emit()
            self.current_frame_changed.emit(self.current_frame_index)
            
    def duplicate_frame(self, index: int):
        """Duplicate a frame."""
        if 0 <= index < len(self.frames):
            image_copy = QImage(self.frames[index].image)
            self.add_frame(image_copy, index + 1)
            
    def set_current_frame(self, index: int):
        """Set current frame index."""
        if 0 <= index < len(self.frames):
            self.current_frame_index = index
            self.current_frame_changed.emit(index)
            
    def renumber_frames(self):
        """Renumber frames after insert/delete."""
        for i, frame in enumerate(self.frames):
            frame.index = i
            
    def get_frame_count(self) -> int:
        return len(self.frames)
    
    def clear(self):
        """Clear all frames."""
        self.frames = [AnimationFrame(0)]
        self.current_frame_index = 0
        self.frames_changed.emit()
        self.current_frame_changed.emit(0)

# ============================================================================
# MISSING WIDGET CLASSES (PLACEHOLDERS)
# ============================================================================

class OrbitalViewWidget(QWidget):
    """Interactive orbital visualization of capsules with animated orbiting and network overlays."""

    capsule_selected = pyqtSignal(object)  # Emits selected capsule

    def __init__(self, capsule_manager, parent=None):
        super().__init__(parent)
        self.capsule_manager = capsule_manager
        self._angle = 0.0
        self._speed = 30.0
        self._radius = 80
        self._capsule_size = 14
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(16)
        self.setMinimumSize(320, 320)
        self.capsules = []
        self.zoom_level = 1.0
        self.selected_capsule = None
        self.hovered_capsule = None
        self.show_info = True
        self.bg_color = QColor(20, 20, 30)
        self.panel_color = QColor(30, 30, 40)
        self.border_color = QColor(60, 60, 80)
        self.highlight_color = QColor(100, 200, 255)
        self.text_color = QColor(255, 255, 255)
        # Overlay data driven by the knowledge network / GNN
        self.network_activity = 0.0
        self.active_connections = []

        # Connect to capsule manager changes
        self.capsule_manager.capsules_changed.connect(self.refresh_capsules)

        self.refresh_capsules()

    def set_speed(self, deg_per_sec: float):
        self._speed = deg_per_sec

    def _step(self):
        dt = self._timer.interval() / 1000.0
        self._angle = (self._angle + self._speed * dt) % 360.0
        self.update()

    def refresh_capsules(self):
        """Refresh the capsule list from the manager."""
        self.capsules = self.capsule_manager.capsules
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
                orbit_radius = cap.orbit_distance * 50 * self.zoom_level
                # Use a deterministic angle based on capsule UUID
                angle = (int(cap.uuid.replace('-', ''), 16) % 360) * (3.14159 / 180)
                x = cx + orbit_radius * math.cos(angle)
                y = cy + orbit_radius * math.sin(angle)
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
            orbit_radius = cap.orbit_distance * 50 * self.zoom_level
            angle = (int(cap.uuid.replace('-', ''), 16) % 360) * (3.14159 / 180)
            x = cx + orbit_radius * math.cos(angle)
            y = cy + orbit_radius * math.sin(angle)
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
        self.draw_starfield(p)
        for cap in self.capsules:
            # Update angle for animation
            angle = (int(cap.uuid.replace('-', ''), 16) % 360 + self._angle) * (3.14159 / 180)
            cap.angle = angle  # Store for consistency
        orbit_radii = sorted(set(int(cap.orbit_distance * 100) / 100.0 for cap in self.capsules))
        p.setPen(QPen(QColor(60, 80, 120), 1))
        p.setOpacity(0.35)
        for radius in orbit_radii:
            orbit_radius = radius * 50 * self.zoom_level
            p.drawEllipse(int(cx - orbit_radius), int(cy - orbit_radius), int(orbit_radius * 2), int(orbit_radius * 2))
        p.setOpacity(1.0)
        for cap in self.capsules:
            orbit_radius = cap.orbit_distance * 50 * self.zoom_level
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

    def draw_starfield(self, p: QPainter):
        p.fillRect(self.rect(), self.bg_color)
        random.seed(42)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        num_stars = 160
        w = self.width()
        h = self.height()
        for _ in range(num_stars):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(1, 3)
            brightness = random.randint(100, 255)
            p.setBrush(QBrush(QColor(brightness, brightness, brightness)))
            p.drawEllipse(x, y, size, size)
        random.seed()

    def draw_central_nucleus(self, p: QPainter, cx: int, cy: int):
        # Heartbeat intensity modulated by network_activity
        for i in range(30, 0, -5):
            glow_color = QColor(255, 200, 0)
            base = 60 + int(40 * self.network_activity)
            glow_color.setAlpha(int(base * (30 - i) / 30))
            p.setBrush(QBrush(glow_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(cx - i), int(cy - i), i * 2, i * 2)
        gradient_color = QColor(255, 220, 100)
        p.setBrush(QBrush(gradient_color))
        p.setPen(QPen(QColor(255, 240, 150), 2))
        p.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)
        p.setBrush(QBrush(QColor(255, 255, 200)))
        p.setPen(Qt.PenStyle.NoPen)
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

    # ============================================================================
    # UTILITY METHODS FOR MAIN.PY INTEGRATION
    # ============================================================================

    def create_model_capsule(self, model_data: Dict[str, Any], model_type: str = "3dmodel") -> Capsule:
        """Create a capsule from 3D model data for the main application.

        Args:
            model_data: Dictionary containing model information
            model_type: Type of model ('character', 'pose', 'skill', etc.)

        Returns:
            New Capsule object representing the model
        """
        name = model_data.get('name', f"Model_{len(self.capsules)}")
        description = model_data.get('description', '3D model capsule')

        # Create vector representation from model data
        vector = self._create_vector_from_model_data(model_data)

        capsule = Capsule(
            name=name,
            capsule_type=model_type,
            vector_dim=len(vector),
            metadata={
                'description': description,
                'model_data': model_data,
                'created_from': 'main_application',
                'color': CAPSULE_COLORS.get(model_type, CAPSULE_COLORS['unassigned'])
            }
        )
        capsule.vector = vector
        self.add_capsule(capsule)
        return capsule

    def _create_vector_from_model_data(self, model_data: Dict[str, Any]) -> np.ndarray:
        """Create a vector representation from model data."""
        # Extract features from model data
        features = []

        # Control points count
        control_points = model_data.get('control_points', [])
        features.append(len(control_points))

        # Patches count
        patches = model_data.get('patches', [])
        features.append(len(patches))

        # Bones count
        bones = model_data.get('bones', {})
        features.append(len(bones))

        # Curves count
        curves = model_data.get('curves', [])
        features.append(len(curves))

        # Materials count
        materials = model_data.get('materials', {})
        features.append(len(materials))

        # Animations count
        animations = model_data.get('animations', {})
        features.append(len(animations))

        # Total keyframes across all animations
        total_keyframes = sum(len(anim.get('keyframes', [])) for anim in animations.values())
        features.append(total_keyframes)

        # Bounding box dimensions (if available)
        bbox = model_data.get('bounding_box')
        if bbox:
            features.extend([bbox['width'], bbox['height'], bbox['depth']])

        # Convert to numpy array and pad/truncate to standard size
        vector = np.array(features, dtype=np.float32)
        target_dim = 32  # Standard capsule dimension

        if len(vector) < target_dim:
            # Pad with zeros
            padding = np.zeros(target_dim - len(vector))
            vector = np.concatenate([vector, padding])
        elif len(vector) > target_dim:
            # Truncate
            vector = vector[:target_dim]

        return vector

    def get_model_capsules(self, model_type: str = None) -> List[Capsule]:
        """Get all capsules created from 3D models.

        Args:
            model_type: Filter by specific model type, or None for all model capsules

        Returns:
            List of Capsule objects created from models
        """
        model_capsules = []
        for capsule in self.capsules:
            if capsule.metadata.get('created_from') == 'main_application':
                if model_type is None or capsule.type == model_type:
                    model_capsules.append(capsule)
        return model_capsules

    def find_similar_models(self, model_data: Dict[str, Any], limit: int = 5) -> List[Tuple[Capsule, float]]:
        """Find similar models based on vector similarity.

        Args:
            model_data: Model data to compare against
            limit: Maximum number of similar models to return

        Returns:
            List of (capsule, similarity_score) tuples
        """
        query_vector = self._create_vector_from_model_data(model_data)
        similarities = []

        for capsule in self.get_model_capsules():
            similarity = self._cosine_similarity(query_vector, capsule.vector)
            similarities.append((capsule, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def create_pose_capsule(self, pose_data: Dict[str, Any]) -> Capsule:
        """Create a capsule from pose data."""
        return self.create_model_capsule(pose_data, 'pose')

    def create_character_capsule(self, character_data: Dict[str, Any]) -> Capsule:
        """Create a capsule from character data."""
        return self.create_model_capsule(character_data, 'character')

    def create_animation_capsule(self, animation_data: Dict[str, Any]) -> Capsule:
        """Create a capsule from animation data."""
        return self.create_model_capsule(animation_data, 'bvh_motion')

    def get_capsule_stats(self) -> Dict[str, Any]:
        """Get statistics about the capsule collection."""
        stats = {
            'total_capsules': len(self.capsules),
            'capsule_types': {},
            'model_capsules': len(self.get_model_capsules()),
            'memory_capsules': len(self.get_capsules_by_type('memory')),
            'personality_capsules': len(self.get_capsules_by_type('personality')),
            'character_capsules': len(self.get_capsules_by_type('character')),
            'pose_capsules': len(self.get_capsules_by_type('pose')),
            'skill_capsules': len(self.get_capsules_by_type('skill')),
            'style_capsules': len(self.get_capsules_by_type('style'))
        }

        # Count by type
        for capsule in self.capsules:
            stats['capsule_types'][capsule.type] = stats['capsule_types'].get(capsule.type, 0) + 1

        return stats

    def export_model_capsules(self, filepath: str):
        """Export all model-related capsules to a file."""
        model_capsules = self.get_model_capsules()
        data = {
            'capsules': [capsule.to_dict() for capsule in model_capsules],
            'exported_at': time.time(),
            'stats': self.get_capsule_stats()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def import_model_capsules(self, filepath: str) -> int:
        """Import model capsules from a file. Returns number imported."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported_count = 0
            for cap_data in data.get('capsules', []):
                # Only import if it doesn't already exist
                existing = self.get_capsule_by_uuid(cap_data.get('uuid'))
                if not existing:
                    capsule = Capsule.from_dict(cap_data)
                    self.add_capsule(capsule)
                    imported_count += 1

            return imported_count
        except Exception as e:
            logger.error(f"Failed to import model capsules: {e}")
            return 0


# ============================================================================
# CONSTANTS FOR MAIN.PY INTEGRATION
# ============================================================================

# Capsule type colors (matching main.py)
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


# ============================================================================
# CONVENIENCE FUNCTIONS FOR MAIN.PY
# ============================================================================

def create_capsule_manager(store_path: str = "capsule_store/capsules.json") -> CapsuleManager:
    """Create and return a new CapsuleManager instance.

    Args:
        store_path: Path to the capsule storage file

    Returns:
        Initialized CapsuleManager
    """
    return CapsuleManager(store_path)


def create_model_capsule_from_data(model_data: Dict[str, Any], capsule_manager: CapsuleManager,
                                  model_type: str = "3dmodel") -> Capsule:
    """Convenience function to create a model capsule.

    Args:
        model_data: Dictionary with model information
        capsule_manager: CapsuleManager instance
        model_type: Type of model capsule

    Returns:
        Created Capsule object
    """
    return capsule_manager.create_model_capsule(model_data, model_type)

