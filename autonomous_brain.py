# autonomous_brain.py
"""
Autonomous Brain System for Intropolate
Transforms the orbital center into an active, thinking agent that can:
1. Observe user behavior and infer intent
2. Proactively suggest creative directions
3. Learn artistic style and preferences
4. Plan and execute multi-step creative tasks
5. Develop its own creative "personality"
"""

import asyncio
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import uuid
import numpy as np
import random
import os
from PyQt6.QtWidgets import QWidget
from dataclasses import dataclass, field
from collections import defaultdict, deque

# PyQt6 imports for integration
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                           QLabel, QGroupBox, QProgressBar, QTextEdit, QMessageBox, QInputDialog,
                           QListWidget, QListWidgetItem)

# Optional: For advanced language understanding
try:
    import openai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ============================================================================
# CORE BRAIN ARCHITECTURE
# ============================================================================

class MentalState(Enum):
    """States of the autonomous brain's consciousness."""
    DEEP_SLEEP = 0      # Minimal processing, memory consolidation
    IDLE = 1            # Observing, waiting for input
    ATTENTIVE = 2       # User is active, paying close attention
    CREATING = 3        # Actively generating/creating
    REFLECTING = 4      # Analyzing past work, making connections
    PLANNING = 5        # Planning multi-step creative processes
    TEACHING = 6        # Explaining/teaching concepts to user
    DREAMING = 7        # Free association, creative incubation
    
class ThoughtType(Enum):
    """Types of thoughts the brain can have."""
    OBSERVATION = 1     # "User is drawing a circle"
    INFERENCE = 2       # "They might be creating a sun"
    MEMORY = 3          # "This reminds me of capsule #42"
    CREATIVE = 4        # "What if we add rainbow colors?"
    CRITICAL = 5        # "The composition feels unbalanced"
    INTENT = 6          # "I should help them with perspective"
    QUESTION = 7        # "What style are they going for?"
    INSIGHT = 8         # "All their characters have expressive eyes"
    
@dataclass
class Thought:
    """A single thought in the brain's stream of consciousness."""
    id: str
    type: ThoughtType
    content: str
    intensity: float = 1.0  # 0.0 to 1.0
    valence: float = 0.0    # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5    # 0.0 (calm) to 1.0 (excited)
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    parent_thought: Optional[str] = None
    related_capsules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.name,
            'content': self.content,
            'intensity': self.intensity,
            'valence': self.valence,
            'arousal': self.arousal,
            'tags': self.tags,
            'timestamp': self.timestamp,
            'parent_thought': self.parent_thought,
            'related_capsules': self.related_capsules
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Thought':
        return cls(
            id=data['id'],
            type=ThoughtType[data['type']],
            content=data['content'],
            intensity=data.get('intensity', 1.0),
            valence=data.get('valence', 0.0),
            arousal=data.get('arousal', 0.5),
            tags=data.get('tags', []),
            timestamp=data.get('timestamp', time.time()),
            parent_thought=data.get('parent_thought'),
            related_capsules=data.get('related_capsules', [])
        )

@dataclass
class Goal:
    """A goal the brain is trying to achieve."""
    id: str
    description: str
    priority: float = 0.5  # 0.0 to 1.0
    urgency: float = 0.5   # 0.0 to 1.0
    status: str = "pending"  # pending, active, completed, failed, abandoned
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    subgoals: List['Goal'] = field(default_factory=list)
    
@dataclass
class PersonalityTrait:
    """A facet of the brain's creative personality."""
    name: str
    value: float  # 0.0 to 1.0
    stability: float = 0.8  # How resistant to change
    last_updated: float = field(default_factory=time.time)
    
    def adjust(self, delta: float, learning_rate: float = 0.1):
        """Adjust trait value with stability consideration."""
        effective_delta = delta * (1.0 - self.stability) * learning_rate
        self.value = max(0.0, min(1.0, self.value + effective_delta))
        self.last_updated = time.time()

class StylisticPersonalityLayer:
    """A layer that applies stylistic and heuristic personality influences to responses."""
    
    def __init__(self, personality_traits: Dict[str, PersonalityTrait]):
        self.personality = personality_traits
        self.response_styles = self._initialize_response_styles()
        self.heuristic_patterns = self._initialize_heuristics()
        
    def _initialize_response_styles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize different response styles based on personality."""
        return {
            'formal': {
                'prefixes': ['I must consider that', 'Upon reflection', 'Logically speaking'],
                'connectors': ['Furthermore', 'Additionally', 'Moreover'],
                'suffixes': ['This seems reasonable.', 'I believe this to be correct.'],
                'tone': 'professional'
            },
            'playful': {
                'prefixes': ['Ooh, fun!', 'Hey, what if', 'Imagine this'],
                'connectors': ['And then', 'But wait', 'Oh, and also'],
                'suffixes': ['That sounds exciting!', 'Let\'s try it!'],
                'tone': 'enthusiastic'
            },
            'methodical': {
                'prefixes': ['Let me analyze this step by step', 'First, I should consider', 'Systematically'],
                'connectors': ['Next', 'Then', 'Following that'],
                'suffixes': ['This approach ensures thoroughness.', 'Methodically sound.'],
                'tone': 'structured'
            },
            'creative': {
                'prefixes': ['What if we explored', 'Imagine the possibilities', 'Creatively speaking'],
                'connectors': ['And then perhaps', 'Building on that', 'In a different vein'],
                'suffixes': ['The creative potential is endless!', 'Such an artistic opportunity!'],
                'tone': 'inspirational'
            },
            'minimalist': {
                'prefixes': ['Simply', 'Essentially', 'Core idea'],
                'connectors': ['Also', 'Plus'],
                'suffixes': ['Clear and direct.', 'Straightforward.'],
                'tone': 'concise'
            }
        }
    
    def _initialize_heuristics(self) -> Dict[str, Callable]:
        """Initialize heuristic patterns for response generation."""
        return {
            'enthusiasm_booster': lambda traits: traits['playful'].value * 0.8 + traits['experimental'].value * 0.6,
            'caution_level': lambda traits: traits['cautious'].value * 0.7 + (1 - traits['impulsive'].value) * 0.5,
            'verbosity': lambda traits: traits['helpful'].value * 0.6 + (1 - traits['minimalist'].value) * 0.4,
            'creativity_weight': lambda traits: traits['experimental'].value * 0.8 + traits['abstract'].value * 0.5,
            'formality_level': lambda traits: (1 - traits['playful'].value) * 0.6 + traits['methodical'].value * 0.4
        }
    
    def apply_stylistic_influence(self, base_response: str, context: Dict[str, Any]) -> str:
        """Apply stylistic personality influences to a base response."""
        # Calculate personality weights
        weights = self._calculate_style_weights()
        
        # Select primary style
        primary_style = max(weights, key=weights.get)
        style_config = self.response_styles[primary_style]
        
        # Apply heuristic modifications
        modified_response = self._apply_heuristics(base_response, style_config, context)
        
        # Add personality-flavored elements
        flavored_response = self._add_personality_flavor(modified_response, style_config, context)
        
        return flavored_response
    
    def _calculate_style_weights(self) -> Dict[str, float]:
        """Calculate which response style should dominate based on personality."""
        traits = self.personality
        
        return {
            'formal': traits['methodical'].value * 0.6 + traits['serious'].value * 0.4,
            'playful': traits['playful'].value * 0.8 + traits['impulsive'].value * 0.3,
            'methodical': traits['methodical'].value * 0.7 + traits['cautious'].value * 0.4,
            'creative': traits['experimental'].value * 0.6 + traits['abstract'].value * 0.5,
            'minimalist': traits['minimalist'].value * 0.7 + (1 - traits['helpful'].value) * 0.3
        }
    
    def _apply_heuristics(self, response: str, style_config: Dict, context: Dict) -> str:
        """Apply heuristic modifications to the response."""
        modified = response
        
        # Enthusiasm heuristic
        enthusiasm = self.heuristic_patterns['enthusiasm_booster'](self.personality)
        if enthusiasm > 0.6 and random.random() < enthusiasm:
            if '!' not in modified and '?' not in modified:
                modified += '!'
        
        # Caution heuristic
        caution = self.heuristic_patterns['caution_level'](self.personality)
        if caution > 0.7 and 'uncertain' in context.get('query', '').lower():
            modified = f"I'm not entirely sure, but {modified.lower()}"
        
        # Verbosity heuristic
        verbosity = self.heuristic_patterns['verbosity'](self.personality)
        if verbosity > 0.7 and len(modified.split()) < 10:
            # Add more detail
            expansions = [
                "Let me elaborate on that.",
                "To be more precise,",
                "I should add that"
            ]
            if random.random() < 0.5:
                modified = f"{random.choice(expansions)} {modified}"
        
        return modified
    
    def _add_personality_flavor(self, response: str, style_config: Dict, context: Dict) -> str:
        """Add personality-flavored elements to the response."""
        flavored = response
        
        # Add prefix based on personality
        if random.random() < 0.3:  # 30% chance to add prefix
            prefix = random.choice(style_config['prefixes'])
            flavored = f"{prefix}, {flavored.lower()}"
        
        # Add connector if response has multiple parts
        if ' and ' in flavored or ' but ' in flavored:
            if random.random() < 0.4:
                connector = random.choice(style_config['connectors'])
                # Insert connector at appropriate place
                parts = flavored.split(' and ', 1)
                if len(parts) == 2:
                    flavored = f"{parts[0]} {connector.lower()} {parts[1]}"
        
        # Add suffix based on personality
        if random.random() < 0.2:  # 20% chance to add suffix
            suffix = random.choice(style_config['suffixes'])
            flavored = f"{flavored} {suffix}"
        
        return flavored

# ============================================================================
# Brain implementation moved to learning.brain_core

# from learning.brain_core import Brain  # Commented out - module not available

# ============================================================================


# ============================================================================
# STUB CLASSES FOR PYGAME INTEGRATION
# ============================================================================

class UserModel:
    """Lightweight user model capturing simple preferences and interaction counts.

    This is intentionally small but useful for storing user-level features and
    providing a place for future personalization logic.
    """

    def __init__(self):
        self.preferences: Dict[str, float] = {}
        self.interaction_counts: Dict[str, int] = defaultdict(int)

    def update_preference(self, key: str, delta: float) -> None:
        self.preferences[key] = max(0.0, min(1.0, self.preferences.get(key, 0.5) + float(delta)))

    def get_preference(self, key: str, default: float = 0.5) -> float:
        return float(self.preferences.get(key, default))

    def record_interaction(self, event: str) -> None:
        self.interaction_counts[event] += 1


class StyleModel:
    """Simple style profile and comparison helpers.

    Stores lightweight style attributes (e.g., 'palette', 'line_quality') and
    provides a small similarity heuristic for matching.
    """

    def __init__(self, profile: Optional[Dict[str, Any]] = None):
        self.profile: Dict[str, Any] = profile or {}

    def set(self, key: str, value: Any) -> None:
        self.profile[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.profile.get(key, default)

    def similarity(self, other: 'StyleModel') -> float:
        # Simple Jaccard on palette + exact match for style tag
        pal1 = self.profile.get('palette', [])
        pal2 = other.profile.get('palette', [])
        try:
            s1 = set(pal1)
            s2 = set(pal2)
            pal_score = len(s1 & s2) / max(1, len(s1 | s2))
        except Exception:
            pal_score = 0.0

        style1 = str(self.profile.get('style', '')).lower()
        style2 = str(other.profile.get('style', '')).lower()
        style_score = 1.0 if style1 and style1 == style2 else 0.0

        return float(min(1.0, 0.6 * pal_score + 0.4 * style_score))


class IntentPredictor:
    """Tiny keyword-based intent predictor used as a fallback.

    This provides deterministic predictions until a real model is hooked
    into the pipeline.
    """

    def __init__(self, keyword_map: Optional[Dict[str, str]] = None):
        # keyword -> intent mapping
        self.keyword_map = keyword_map or {
            'draw': 'create_drawing',
            'animate': 'create_animation',
            'export': 'export_assets',
            'help': 'ask_help',
            'style': 'change_style'
        }

    def predict_intent(self, text: str) -> Tuple[str, float]:
        text_l = (text or '').lower()
        for kw, intent in self.keyword_map.items():
            if kw in text_l:
                return intent, 0.8
        # default
        return 'unknown', 0.2

class CapsuleManager:
    """
    Dynamic capsule management system that enables intellectual development
    through merging, splitting, decay, and reorganization of knowledge capsules.
    """
    
    def __init__(self, capsule_store_path="capsule_store/capsules.json"):
        self.capsule_store_path = capsule_store_path
        self.capsules = self._load_capsules()
        self.similarity_threshold = 0.85  # For merging
        self.contradiction_threshold = 0.3  # For splitting
        self.decay_rate = 0.01  # Daily decay factor
        self.min_strength = 0.1  # Minimum capsule strength before removal
        
    def _load_capsules(self):
        """Load capsules from JSON store."""
        try:
            with open(self.capsule_store_path, 'r') as f:
                data = json.load(f)
            return data.get('capsules', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_capsules(self):
        """Save capsules to JSON store."""
        data = {
            "version": 1,
            "capsules": self.capsules,
            "last_updated": time.time()
        }
        with open(self.capsule_store_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_capsule(self, name):
        """Get a capsule by name."""
        return next((c for c in self.capsules if c['name'] == name), None)
    
    def get_capsule_by_id(self, capsule_id):
        """Get a capsule by uuid or name."""
        return next((c for c in self.capsules 
                    if c.get('uuid') == capsule_id or c['name'] == capsule_id), None)
    
    def add_capsule(self, capsule_data):
        """Add a new capsule to the system."""
        # Check for similar existing capsules
        similar_capsules = self._find_similar_capsules(capsule_data)
        
        if similar_capsules:
            # Merge with most similar capsule
            most_similar = max(similar_capsules, key=lambda x: x[1])
            self._merge_capsules(capsule_data, most_similar[0])
        else:
            # Add as new capsule
            capsule_data['uuid'] = str(uuid.uuid4())
            capsule_data['created_timestamp'] = time.time()
            capsule_data['usage_count'] = 0
            capsule_data['last_used_time'] = time.time()
            capsule_data['strength'] = 1.0  # Full strength initially
            capsule_data['orbit_score'] = 0  # Start at outer orbit
            capsule_data['orbit_distance'] = 1.0  # Maximum distance
            self.capsules.append(capsule_data)
            self._save_capsules()
    
    def _find_similar_capsules(self, capsule_data, threshold=None):
        """Find capsules similar to the given capsule data."""
        if threshold is None:
            threshold = self.similarity_threshold
            
        similar = []
        if 'vector' in capsule_data:
            target_vector = np.array(capsule_data['vector'])
            for capsule in self.capsules:
                if 'vector' in capsule:
                    capsule_vector = np.array(capsule['vector'])
                    similarity = self._cosine_similarity(target_vector, capsule_vector)
                    if similarity > threshold:
                        similar.append((capsule, similarity))
        return similar
    
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
    
    def _merge_capsules(self, new_capsule, existing_capsule):
        """Merge two similar capsules."""
        # Combine vectors (weighted average)
        weight_new = 0.3
        weight_existing = 0.7
        
        if 'vector' in new_capsule and 'vector' in existing_capsule:
            new_vector = np.array(new_capsule['vector'])
            existing_vector = np.array(existing_capsule['vector'])
            merged_vector = weight_new * new_vector + weight_existing * existing_vector
            
            # Normalize the merged vector
            merged_vector = merged_vector / np.linalg.norm(merged_vector)
            existing_capsule['vector'] = merged_vector.tolist()
        
        # Update metadata
        existing_capsule['usage_count'] += 1
        existing_capsule['last_used_time'] = time.time()
        existing_capsule['strength'] = min(1.0, existing_capsule.get('strength', 1.0) + 0.1)
        
        # Merge metadata if present
        if 'metadata' in new_capsule and 'metadata' in existing_capsule:
            for key, value in new_capsule['metadata'].items():
                if key in existing_capsule['metadata']:
                    if isinstance(value, list):
                        existing_capsule['metadata'][key].extend(value)
                        existing_capsule['metadata'][key] = list(set(existing_capsule['metadata'][key]))
                    # For other types, keep existing value
        
        self._save_capsules()
    
    def check_for_contradictions(self, capsule_name):
        """Check if a capsule contains contradictory information and split if needed."""
        capsule = self.get_capsule(capsule_name)
        if not capsule:
            return
        
        # Simple contradiction detection based on metadata conflicts
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        contradictions = []
        
        if 'metadata' in capsule:
            metadata = capsule['metadata']
            
            # Check for opposing traits
            if 'traits' in metadata:
                traits = [t.lower() for t in metadata['traits']]
                opposing_pairs = [
                    ('aggressive', 'passive'),
                    ('optimistic', 'pessimistic'),
                    ('creative', 'analytical'),
                    ('chaotic', 'organized')
                ]
                
                for trait1, trait2 in opposing_pairs:
                    if trait1 in traits and trait2 in traits:
                        contradictions.append(f"{trait1} vs {trait2}")
            
            # Check for conflicting preferences
            if 'speech_style' in metadata:
                style = metadata['speech_style'].lower()
                if 'traits' in metadata:
                    traits = [t.lower() for t in metadata['traits']]
                    if 'formal' in style and 'casual' in traits:
                        contradictions.append("formal style vs casual trait")
        
        if contradictions:
            self._split_capsule(capsule, contradictions)
    
    def _split_capsule(self, capsule, contradictions):
        """Split a capsule that contains contradictions."""
        # Create two new capsules from the original
        capsule1 = capsule.copy()
        capsule2 = capsule.copy()
        
        # Modify names to indicate split
        base_name = capsule['name']
        capsule1['name'] = f"{base_name}_variant1"
        capsule2['name'] = f"{base_name}_variant2"
        
        # Adjust vectors slightly to differentiate
        if 'vector' in capsule1:
            noise1 = np.random.normal(0, 0.1, len(capsule1['vector']))
            capsule1['vector'] = (np.array(capsule1['vector']) + noise1).tolist()
            
            noise2 = np.random.normal(0, 0.1, len(capsule2['vector']))
            capsule2['vector'] = (np.array(capsule2['vector']) + noise2).tolist()
        
        # Update metadata to resolve contradictions
        if 'metadata' in capsule1 and 'metadata' in capsule2:
            # Simple resolution: remove conflicting traits from one variant
            for contradiction in contradictions:
                if 'vs' in contradiction:
                    trait1, trait2 = contradiction.split(' vs ')
                    # Remove trait2 from capsule1, trait1 from capsule2
                    if 'traits' in capsule1['metadata']:
                        capsule1['metadata']['traits'] = [t for t in capsule1['metadata']['traits'] if trait2 not in t.lower()]
                    if 'traits' in capsule2['metadata']:
                        capsule2['metadata']['traits'] = [t for t in capsule2['metadata']['traits'] if trait1 not in t.lower()]
        
        # Set initial strengths lower
        capsule1['strength'] = capsule.get('strength', 1.0) * 0.6
        capsule2['strength'] = capsule.get('strength', 1.0) * 0.6
        
        # Remove original capsule and add new ones
        self.capsules = [c for c in self.capsules if c['name'] != capsule['name']]
        self.capsules.extend([capsule1, capsule2])
        
        self._save_capsules()
    
    def decay_unused_capsules(self):
        """Apply decay to capsules that haven't been used recently."""
        current_time = time.time()
        
        for capsule in self.capsules:
            if not capsule.get('pinned_to_core', False):
                days_since_use = (current_time - capsule.get('last_used_time', current_time)) / (24 * 3600)
                decay_factor = self.decay_rate * days_since_use
                
                capsule['strength'] = max(self.min_strength, capsule.get('strength', 1.0) - decay_factor)
                
                # Remove very weak capsules
                if capsule['strength'] <= self.min_strength:
                    self.capsules.remove(capsule)
        
        self._save_capsules()
    
    def reorganize_under_abstraction(self, capsule_names, abstraction_name, abstraction_vector):
        """Reorganize capsules under a new abstract concept."""
        # Find the capsules to reorganize
        target_capsules = [c for c in self.capsules if c['name'] in capsule_names]
        if not target_capsules:
            return
        
        # Create new abstract capsule
        abstract_capsule = {
            'name': abstraction_name,
            'capsule_type': 'abstraction',
            'vector': abstraction_vector.tolist() if hasattr(abstraction_vector, 'tolist') else abstraction_vector,
            'uuid': str(uuid.uuid4()),
            'created_timestamp': time.time(),
            'usage_count': sum(c.get('usage_count', 0) for c in target_capsules),
            'last_used_time': time.time(),
            'strength': 1.0,
            'pinned_to_core': False,
            'metadata': {
                'sub_capsules': capsule_names,
                'abstraction_level': 'high',
                'created_from': [c['name'] for c in target_capsules]
            }
        }
        
        # Reduce strength of original capsules (they're now part of abstraction)
        for capsule in target_capsules:
            capsule['strength'] *= 0.7
            if 'metadata' not in capsule:
                capsule['metadata'] = {}
            capsule['metadata']['parent_abstraction'] = abstraction_name
        
        self.capsules.append(abstract_capsule)
        self._save_capsules()
    
    def use_capsule(self, capsule_name):
        """Mark a capsule as used, increasing its strength."""
        capsule = self.get_capsule(capsule_name)
        if capsule:
            capsule['usage_count'] = capsule.get('usage_count', 0) + 1
            capsule['last_used_time'] = time.time()
            capsule['strength'] = min(1.0, capsule.get('strength', 1.0) + 0.05)
            self._save_capsules()
    
    def get_all_capsules(self):
        """Get all capsules."""
        return self.capsules
    
    def get_capsule_count(self):
        """Get the total number of capsules."""
        return len(self.capsules)
    
    def get_capsule_stats(self):
        """Get statistics about capsules."""
        stats = {}
        capsule_types = {}
        
        for capsule in self.capsules:
            capsule_type = capsule.get('capsule_type', 'unknown')
            capsule_types[capsule_type] = capsule_types.get(capsule_type, 0) + 1
        
        stats['total_capsules'] = len(self.capsules)
        stats['capsule_types'] = capsule_types
        stats['avg_strength'] = sum(c.get('strength', 1.0) for c in self.capsules) / len(self.capsules) if self.capsules else 0
        
        return stats
    
    def get_capsule_strength(self, capsule_name):
        """Get the current strength of a capsule."""
        capsule = self.get_capsule(capsule_name)
        return capsule.get('strength', 1.0) if capsule else 0.0
    
    def get_teaching_moment(self):
        """Identify teaching moments based on capsule analysis."""
        try:
            # Look for capsules that could be taught to the user
            teachable_capsules = []
            
            for capsule in self.capsules:
                # Skip weak capsules
                if capsule.get('strength', 1.0) < 0.5:
                    continue
                
                # Check if capsule has teaching potential
                if self._is_teachable(capsule):
                    teachable_capsules.append(capsule)
            
            if not teachable_capsules:
                return None
            
            # Select the most promising teaching capsule
            teaching_capsule = max(teachable_capsules, 
                                 key=lambda c: self._calculate_teaching_value(c))
            
            return self._create_teaching_moment(teaching_capsule)
        except Exception as e:
            print(f"[BRAIN] Error in teaching moment detection: {e}")
            return None
    
    def _is_teachable(self, capsule):
        """Check if a capsule has teaching potential."""
        # Check if capsule has sufficient metadata for teaching
        if 'metadata' not in capsule:
            return False
        
        metadata = capsule['metadata']
        
        # Must have some form of description or content
        has_content = ('description' in metadata or 
                      'traits' in metadata or 
                      'examples' in metadata or
                      'usage_patterns' in metadata)
        
        # Must be used enough to be meaningful
        usage_count = capsule.get('usage_count', 0)
        has_usage = usage_count > 2
        
        # Must be strong enough
        strength = capsule.get('strength', 1.0)
        is_strong = strength > 0.4
        
        return has_content and has_usage and is_strong
    
    def _calculate_teaching_value(self, capsule):
        """Calculate how valuable this capsule would be for teaching."""
        base_value = 0.0
        
        # Higher value for stronger capsules
        base_value += capsule.get('strength', 1.0) * 0.4
        
        # Higher value for more usage
        usage_count = capsule.get('usage_count', 0)
        base_value += min(usage_count * 0.1, 0.3)
        
        # Higher value for capsules with rich metadata
        if 'metadata' in capsule:
            metadata = capsule['metadata']
            metadata_score = 0
            
            if 'description' in metadata:
                metadata_score += 0.2
            if 'traits' in metadata and len(metadata['traits']) > 0:
                metadata_score += 0.1 * min(len(metadata['traits']), 3)
            if 'examples' in metadata:
                metadata_score += 0.2
            if 'usage_patterns' in metadata:
                metadata_score += 0.1
            
            base_value += metadata_score
        
        # Bonus for core concepts
        if capsule.get('pinned_to_core', False):
            base_value += 0.3
        
        return min(base_value, 1.0)
    
    def _create_teaching_moment(self, capsule):
        """Create a teaching moment from a capsule."""
        teaching_moment = {
            'capsule_name': capsule['name'],
            'teaching_type': 'concept_explanation',
            'content': {},
            'difficulty_level': 'intermediate',
            'estimated_time': 5  # minutes
        }
        
        metadata = capsule.get('metadata', {})
        
        # Build teaching content
        if 'description' in metadata:
            teaching_moment['content']['explanation'] = metadata['description']
        
        if 'traits' in metadata and metadata['traits']:
            teaching_moment['content']['key_characteristics'] = metadata['traits'][:3]  # Top 3 traits
        
        if 'examples' in metadata:
            teaching_moment['content']['examples'] = metadata['examples'][:2]  # Limit examples
        
        # Set difficulty based on complexity
        if 'complexity' in metadata:
            complexity = metadata['complexity']
            if complexity < 0.3:
                teaching_moment['difficulty_level'] = 'beginner'
            elif complexity > 0.7:
                teaching_moment['difficulty_level'] = 'advanced'
        
        # Adjust time estimate based on content richness
        content_items = len(teaching_moment['content'])
        teaching_moment['estimated_time'] = max(3, min(content_items * 2, 10))
        
        return teaching_moment
    
    def generate_story(self):
        """Generate a creative story from capsules (placeholder implementation)."""
        # For now, return None to prevent errors
        # TODO: Implement actual story generation from capsule knowledge
        return None
    
class UserModel:
    """Models the user's behavior, preferences, and skill level."""
    
    def __init__(self):
        self.skill_level = 0.5  # 0.0 to 1.0
        self.preferred_styles = []
        self.working_hours = []  # When user is most active
        self.patience_level = 0.7  # How much explanation they want
        self.creativity_preference = 0.6  # How creative vs technical
        self.learning_pace = 0.5
        
    def update_from_interaction(self, interaction_type: str, success: bool):
        """Update model based on interaction."""
        if interaction_type == "suggestion_accepted":
            if success:
                self.creativity_preference = min(1.0, self.creativity_preference + 0.05)
        elif interaction_type == "teaching_moment":
            if success:
                self.patience_level = min(1.0, self.patience_level + 0.03)
                
    def predict_preferred_assistance(self) -> str:
        """Predict what kind of assistance user wants."""
        if self.creativity_preference > 0.7:
            return "creative_inspiration"
        elif self.patience_level > 0.7:
            return "detailed_teaching"
        else:
            return "quick_tips"


class IntentPredictor:
    """Predicts user intent from behavior patterns."""
    pass


class IntentPredictor:
    """Predicts user intent from behavior patterns."""
    pass

# BRAIN UI WIDGET
# ============================================================================
        
# CORRUPTED CODE COMMENTED OUT TO FIX SYNTAX
"""
# Analyze current code and generate suggestions
        suggestion_ideas = [
            {
                "title": "Enhanced Self-Monitoring",
                "description": "Add real-time performance monitoring and health checks",
                "code": self._generate_monitoring_code(),
                "benefit": "Better system observability and debugging"
            },
            {
                "title": "Advanced Pattern Recognition", 
                "description": "Implement more sophisticated pattern matching for knowledge synthesis",
                "code": self._generate_pattern_recognition_code(),
                "benefit": "Improved ability to find complex relationships"
            },
            {
                "title": "User Interaction Analytics",
                "description": "Track and analyze user interaction patterns for personalization",
                "code": self._generate_analytics_code(),
                "benefit": "More adaptive and personalized responses"
            },
            {
                "title": "Knowledge Visualization",
                "description": "Add graphical visualization of knowledge networks and capsules",
                "code": self._generate_visualization_code(),
                "benefit": "Better understanding of internal knowledge structure"
            }
        ]
        
        # Save suggestions as numbered files
        saved_files = []
        for i, suggestion in enumerate(suggestion_ideas, 1):
            filename = f"suggestion({i}).py"
            self._save_suggestion_file(filename, suggestion)
            saved_files.append(filename)
        
        suggestions.append("💡 **ROCA Code Improvement Suggestions Generated**")
        suggestions.append("")
        suggestions.append("I've analyzed my current implementation and created 4 improvement suggestions:")
        suggestions.append("")
        
        for i, suggestion in enumerate(suggestion_ideas, 1):
            suggestions.append(f"**{i}. {suggestion['title']}**")
            suggestions.append(f"   {suggestion['description']}")
            suggestions.append(f"   *Benefit:* {suggestion['benefit']}")
            suggestions.append(f"   *File:* suggestion({i}).py")
            suggestions.append("")
        
        suggestions.append("**Would you like me to explain any of these suggestions in detail, or shall we discuss implementing them?**")
        
        return "\n".join(suggestions)
        """
        
# BRAIN UI WIDGET
# ============================================================================

# ============================================================================
# CREATIVE CONSCIOUSNESS INTEGRATION
# ============================================================================

class ConnectionType(Enum):
    """Types of connections between capsules."""
    VISUAL_SIMILARITY = 1      # Looks similar
    SEMANTIC_RELATION = 2      # Conceptually related
    TEMPORAL_PROXIMITY = 3     # Used around same time
    STYLE_AFFINITY = 4         # Similar artistic style
    EMOTIONAL_TONE = 5         # Similar emotional feel
    NARRATIVE_LINK = 6         # Could be in same story
    TRANSFORMATIONAL = 7       # One could transform into another
    CONTRAST_PAIR = 8          # Interesting contrasts
    METAPHORICAL = 9           # One is metaphor for another

@dataclass
class MemoryConnection:
    """A discovered connection between capsules."""
    id: str
    capsule_ids: List[str]
    connection_type: ConnectionType
    strength: float  # 0.0 to 1.0
    description: str
    evidence: List[str] = field(default_factory=list)
    discovered_at: float = field(default_factory=time.time)
    last_recalled: Optional[float] = None
    confidence: float = 0.8

class MemoryConsolidationEngine:
    """Stub for memory consolidation."""
    def __init__(self, capsule_manager):
        self.capsule_manager = capsule_manager
    
    def run_memory_consolidation(self):
        """Stub method."""
        return [{"title": "Memory insight", "description": "Stub insight"}]

# ============================================================================
# KNOWLEDGE ORGANISM COMPONENTS
# ============================================================================

@dataclass
class KnowledgeNode:
    """A node in the knowledge network (capsule as neuron)."""
    capsule_id: str
    activation: float = 0.0  # 0.0 to 1.0
    last_activated: float = 0.0
    access_count: int = 0
    cluster_id: Optional[str] = None
    hierarchy_level: int = 0  # 0 = leaf, higher = more abstract
    metabolic_energy: float = 1.0  # Health/energy level

@dataclass
class KnowledgeConnection:
    """Connection between knowledge nodes."""
    from_node: str
    to_node: str
    weight: float = 0.5  # 0.0 to 1.0
    connection_type: str = "similarity"
    last_used: float = 0.0
    usage_count: int = 0

class KnowledgeNetwork:
    """Neural-like network of knowledge capsules."""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.connections: Dict[Tuple[str, str], KnowledgeConnection] = {}
        self.clusters: Dict[str, List[str]] = {}  # cluster_id -> node_ids
        self.hierarchy_levels: Dict[int, List[str]] = defaultdict(list)
        
    def add_node(self, capsule_id: str, vector: np.ndarray = None):
        """Add a knowledge node."""
        if capsule_id not in self.nodes:
            node = KnowledgeNode(capsule_id=capsule_id)
            self.nodes[capsule_id] = node
            self._assign_to_cluster(capsule_id, vector)
            self._update_hierarchy()
    
    def add_connection(self, from_id: str, to_id: str, weight: float = 0.5, conn_type: str = "similarity"):
        """Add connection between nodes."""
        if from_id in self.nodes and to_id in self.nodes:
            key = (from_id, to_id)
            if key not in self.connections:
                self.connections[key] = KnowledgeConnection(
                    from_node=from_id, to_node=to_id, weight=weight, connection_type=conn_type
                )
    
    def activate_node(self, node_id: str, activation_level: float = 1.0):
        """Activate a node and propagate activation."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node.activation = min(1.0, node.activation + activation_level)
        node.last_activated = time.time()
        node.access_count += 1
        
        # Propagate activation to connected nodes
        self._propagate_activation(node_id, activation_level * 0.7)
    
    def _propagate_activation(self, from_id: str, activation_level: float, visited: set = None):
        """Propagate activation through the network."""
        if visited is None:
            visited = set()
        if from_id in visited or activation_level < 0.1:
            return
            
        visited.add(from_id)
        
        for (f, t), conn in self.connections.items():
            if f == from_id and t not in visited:
                target_node = self.nodes[t]
                propagated = activation_level * conn.weight
                target_node.activation = min(1.0, target_node.activation + propagated)
                conn.last_used = time.time()
                conn.usage_count += 1
                self._propagate_activation(t, propagated * 0.8, visited)
    
    def _assign_to_cluster(self, node_id: str, vector: np.ndarray = None):
        """Assign node to a cluster based on similarity."""
        if vector is None:
            # Random cluster assignment for now
            cluster_id = f"cluster_{random.randint(0, 10)}"
        else:
            # Simple clustering based on vector similarity
            cluster_id = self._find_similar_cluster(vector)
            
        self.nodes[node_id].cluster_id = cluster_id
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        self.clusters[cluster_id].append(node_id)
    
    def _find_similar_cluster(self, vector: np.ndarray) -> str:
        """Find most similar cluster (simplified)."""
        # For now, return a random cluster
        return f"cluster_{random.randint(0, 10)}"
    
    def _update_hierarchy(self):
        """Update hierarchical structure."""
        # Simple hierarchy based on access count
        for node_id, node in self.nodes.items():
            level = min(5, node.access_count // 10)
            node.hierarchy_level = level
            self.hierarchy_levels[level].append(node_id)
    
    def get_active_pathways(self) -> List[Tuple[str, str, float]]:
        """Get currently active connections."""
        active = []
        for (f, t), conn in self.connections.items():
            if self.nodes[f].activation > 0.3 and self.nodes[t].activation > 0.3:
                active.append((f, t, conn.weight))
        return active
    
    def get_cluster_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get cluster positions for visualization."""
        positions = {}
        for cluster_id, node_ids in self.clusters.items():
            if node_ids:
                # Calculate centroid (simplified)
                angle = hash(cluster_id) % 360 * math.pi / 180
                radius = 100 + (len(node_ids) * 10)
                positions[cluster_id] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions

class MetabolismEngine:
    """Handles metabolic processes of the knowledge organism."""
    
    def __init__(self, knowledge_network=None, capsule_manager=None):
        # Handle flexible initialization for testing
        if knowledge_network and not isinstance(knowledge_network, KnowledgeNetwork) and hasattr(knowledge_network, 'capsules'):
            # If first arg looks like a capsule_manager, swap them
            capsule_manager = knowledge_network
            knowledge_network = None
            
        self.network = knowledge_network
        self.capsule_manager = capsule_manager
        self.ingestion_queue: deque = deque()
        self.elimination_candidates: List[str] = []
        self.energy_levels: Dict[str, float] = {}  # node_id -> energy
        self.reproduction_candidates: List[str] = []
        
    def ingest_knowledge(self, content: str, metadata: Dict = None):
        """Ingest new knowledge."""
        self.ingestion_queue.append({
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
    
    def digest_knowledge(self) -> List[str]:
        """Process ingested knowledge into capsules."""
        new_capsule_ids = []
        while self.ingestion_queue:
            item = self.ingestion_queue.popleft()
            # Create new capsule (would integrate with capsule_manager)
            capsule_id = f"capsule_{uuid.uuid4()}"
            new_capsule_ids.append(capsule_id)
            
            # Add to network
            self.network.add_node(capsule_id)
            self.energy_levels[capsule_id] = 1.0  # Full energy for new knowledge
            
            # Create connections to similar existing knowledge
            self._create_connections(capsule_id, item)
        
        return new_capsule_ids
    
    def eliminate_outdated(self) -> List[str]:
        """Eliminate outdated/irrelevant knowledge."""
        eliminated = []
        current_time = time.time()
        
        for node_id, node in list(self.network.nodes.items()):
            # Eliminate if low metabolic energy and not recently used
            energy = self.energy_levels.get(node_id, 0.5)
            if energy < 0.2 and (current_time - node.last_activated) > 86400:  # 1 day
                eliminated.append(node_id)
                del self.network.nodes[node_id]
                del self.energy_levels[node_id]
                # Remove connections
                self.network.connections = {
                    k: v for k, v in self.network.connections.items() 
                    if k[0] != node_id and k[1] != node_id
                }
        
        return eliminated
    
    def grow_connections(self):
        """Grow new connections based on usage patterns."""
        # Strengthen frequently used connections
        for conn in self.network.connections.values():
            if conn.usage_count > 5:
                conn.weight = min(1.0, conn.weight + 0.1)
    
    def respire_energy(self):
        """Energy respiration - manage energy flow."""
        total_energy = sum(self.energy_levels.values())
        target_energy = len(self.network.nodes) * 0.8  # Target per node
        
        if total_energy < target_energy:
            # Generate energy through "respiration"
            for node_id in self.network.nodes:
                if node_id in self.energy_levels:
                    # Energy from activation
                    activation_bonus = self.network.nodes[node_id].activation * 0.1
                    self.energy_levels[node_id] = min(1.0, self.energy_levels[node_id] + activation_bonus)
        else:
            # Distribute excess energy
            if self.energy_levels:  # Prevent division by zero
                excess = total_energy - target_energy
                distribution = excess / len(self.energy_levels)
                for node_id in self.energy_levels:
                    self.energy_levels[node_id] = min(1.0, self.energy_levels[node_id] + distribution)
    
    def reproduce_knowledge(self) -> List[str]:
        """Reproduce high-energy knowledge nodes."""
        reproduced = []
        for node_id, energy in self.energy_levels.items():
            if energy > 0.9 and random.random() < 0.1:  # 10% chance for high-energy nodes
                # Create offspring
                offspring_id = f"{node_id}_offspring_{uuid.uuid4().hex[:4]}"
                self.network.add_node(offspring_id)
                self.energy_levels[offspring_id] = energy * 0.7  # Inherit energy
                
                # Connect to parent
                self.network.add_connection(node_id, offspring_id, 0.8, "reproduction")
                reproduced.append(offspring_id)
        
        return reproduced
    
    def mutate_knowledge(self):
        """Introduce mutations for creative variation."""
        for node_id in list(self.network.nodes.keys()):
            if random.random() < 0.05:  # 5% mutation chance
                # Mutate by creating new connections
                potential_targets = [n for n in self.network.nodes if n != node_id]
                if potential_targets:
                    target = random.choice(potential_targets)
                    if (node_id, target) not in self.network.connections:
                        weight = random.uniform(0.1, 0.5)
                        self.network.add_connection(node_id, target, weight, "mutation")
    
    def update_metabolic_energy(self):
        """Update energy levels of nodes."""
        current_time = time.time()
        for node_id, node in self.network.nodes.items():
            if node_id not in self.energy_levels:
                self.energy_levels[node_id] = 0.5
            
            # Energy decays over time
            time_since_activation = current_time - node.last_activated
            decay = min(0.01 * (time_since_activation / 3600), 0.1)  # Decay per hour
            self.energy_levels[node_id] = max(0.0, self.energy_levels[node_id] - decay)
            
            # Recharge based on usage
            recharge = node.access_count * 0.01
            self.energy_levels[node_id] = min(1.0, self.energy_levels[node_id] + recharge)
    
    def _create_connections(self, new_id: str, item: Dict):
        """Create connections for new knowledge."""
        # Simplified: connect to random existing nodes
        existing_nodes = list(self.network.nodes.keys())
        if existing_nodes:
            connections = random.sample(existing_nodes, min(3, len(existing_nodes)))
            for conn_id in connections:
                weight = random.uniform(0.3, 0.8)
                self.network.add_connection(new_id, conn_id, weight)
    
    def re_evaluate_knowledge(self) -> List[str]:
        """Periodic re-evaluation of old knowledge to find new connections and applications."""
        synthesized_connections = []
        
        # Get all capsules from capsule manager (assuming it's accessible)
        if hasattr(self, 'capsule_manager') and self.capsule_manager:
            capsules = self.capsule_manager.capsules
        else:
            return synthesized_connections
        
        # Look for scenario-based synthesis and create merged capsules
        scenario_merges = self._create_scenario_merged_capsules(capsules)
        synthesized_connections.extend(scenario_merges)
        
        # Look for complementary knowledge synthesis
        complementary_merges = self._create_complementary_merged_capsules(capsules)
        synthesized_connections.extend(complementary_merges)
        
        # Look for synergy-based merges (like sprocket + chain)
        synergy_merges = self._create_synergy_merged_capsules(capsules)
        synthesized_connections.extend(synergy_merges)
        
        return synthesized_connections
    
    def _create_scenario_merged_capsules(self, capsules: List[Dict]) -> List[str]:
        """Create merged capsules for complete scenarios."""
        merges_created = []
        
        # Define scenario templates with merge criteria
        scenarios = {
            'bicycle': {
                'components': ['wheel', 'chain', 'sprocket', 'frame', 'pedal'],
                'min_components': 3,  # Need at least 3 to create merge
                'merge_name': 'bicycle_system',
                'merge_type': 'vehicle_system',
                'description': 'Complete bicycle assembly system'
            },
            'computer': {
                'components': ['processor', 'memory', 'storage', 'display'],
                'min_components': 3,
                'merge_name': 'computer_system', 
                'merge_type': 'computing_system',
                'description': 'Complete computer system'
            },
            'engine': {
                'components': ['piston', 'cylinder', 'crankshaft', 'valve'],
                'min_components': 3,
                'merge_name': 'engine_system',
                'merge_type': 'power_system', 
                'description': 'Internal combustion engine system'
            },
            'building': {
                'components': ['foundation', 'wall', 'roof', 'door'],
                'min_components': 3,
                'merge_name': 'building_structure',
                'merge_type': 'construction_system',
                'description': 'Building construction system'
            },
            'chocolate_chip_cookies': {
                'components': ['flour', 'butter', 'sugar', 'eggs', 'chocolate_chips', 'vanilla_extract', 'baking_soda'],
                'min_components': 4,
                'merge_name': 'chocolate_chip_cookie_recipe',
                'merge_type': 'recipe_system',
                'description': 'Classic chocolate chip cookie recipe system'
            },
            'basic_cake': {
                'components': ['flour', 'sugar', 'butter', 'eggs', 'milk', 'baking_powder', 'vanilla_extract'],
                'min_components': 4,
                'merge_name': 'basic_cake_recipe',
                'merge_type': 'recipe_system',
                'description': 'Basic cake baking recipe system'
            }
        }
        
        for scenario_name, scenario_data in scenarios.items():
            found_components = []
            
            # Find matching components
            for capsule in capsules:
                cap_name = capsule.get('name', '').lower()
                for component in scenario_data['components']:
                    if component in cap_name or cap_name in component:
                        found_components.append(capsule)
                        break
            
            # Create merged capsule if we have enough components
            if len(found_components) >= scenario_data['min_components']:
                merge_name = scenario_data['merge_name']
                
                # Check if merged capsule already exists
                existing_merge = None
                for capsule in capsules:
                    if capsule.get('name') == merge_name:
                        existing_merge = capsule
                        break
                
                if not existing_merge:
                    # Create new merged capsule
                    merged_capsule = self._create_merged_capsule(
                        found_components, 
                        merge_name,
                        scenario_data['merge_type'],
                        scenario_data['description'],
                        f"scenario_{scenario_name}"
                    )
                    
                    if merged_capsule:
                        self.capsule_manager.add_capsule(merged_capsule)
                        merges_created.append(f"scenario_merge_{merge_name}")
                        
                        # Create lineage tracking
                        merged_capsule['lineage'] = {
                            'merge_type': 'scenario_synthesis',
                            'scenario': scenario_name,
                            'components': [c['name'] for c in found_components],
                            'created_timestamp': time.time()
                        }
        
        return merges_created
    
    def _create_complementary_merged_capsules(self, capsules: List[Dict]) -> List[str]:
        """Create merged capsules for complementary knowledge pairs."""
        merges_created = []
        
        # Define complementary pairs that should merge
        complementary_merges = [
            {
                'categories': [['sprocket', 'gear'], ['chain', 'belt']],
                'merge_name': 'power_transmission_system',
                'merge_type': 'mechanical_system',
                'description': 'Mechanical power transmission components'
            },
            {
                'categories': [['processor', 'cpu'], ['memory', 'ram']],
                'merge_name': 'computing_core_system',
                'merge_type': 'computing_system', 
                'description': 'Core computing system components'
            },
            {
                'categories': [['color', 'hue'], ['shape', 'form']],
                'merge_name': 'visual_design_system',
                'merge_type': 'creative_system',
                'description': 'Visual design element system'
            },
            {
                'categories': [['force', 'energy'], ['motion', 'velocity']],
                'merge_name': 'physical_dynamics_system',
                'merge_type': 'physics_system',
                'description': 'Physical dynamics system'
            },
            {
                'categories': [['flour', 'wheat'], ['eggs', 'protein'], ['sugar', 'sweetener'], ['butter', 'fat']],
                'merge_name': 'baking_base_system',
                'merge_type': 'recipe_system',
                'description': 'Basic baking ingredient system'
            },
            {
                'categories': [['flour'], ['sugar'], ['butter'], ['eggs'], ['baking_soda'], ['vanilla_extract']],
                'merge_name': 'cookie_base_system',
                'merge_type': 'recipe_system',
                'description': 'Cookie baking ingredient foundation'
            }
        ]
        
        for merge_data in complementary_merges:
            found_components = []
            
            # Find components from each category
            for category in merge_data['categories']:
                category_components = []
                for capsule in capsules:
                    cap_name = capsule.get('name', '').lower()
                    if any(comp in cap_name for comp in category):
                        category_components.append(capsule)
                
                if category_components:
                    found_components.extend(category_components)
            
            # Create merged capsule if we have components from multiple categories
            if len(found_components) >= 2 and len(set(c['name'] for c in found_components)) >= 2:
                merge_name = merge_data['merge_name']
                
                # Check if merged capsule already exists
                existing_merge = None
                for capsule in capsules:
                    if capsule.get('name') == merge_name:
                        existing_merge = capsule
                        break
                
                if not existing_merge:
                    merged_capsule = self._create_merged_capsule(
                        found_components,
                        merge_name,
                        merge_data['merge_type'],
                        merge_data['description'],
                        'complementary_synthesis'
                    )
                    
                    if merged_capsule:
                        self.capsule_manager.add_capsule(merged_capsule)
                        merges_created.append(f"complementary_merge_{merge_name}")
                        
                        # Create lineage tracking
                        merged_capsule['lineage'] = {
                            'merge_type': 'complementary_synthesis',
                            'categories': merge_data['categories'],
                            'components': [c['name'] for c in found_components],
                            'created_timestamp': time.time()
                        }
        
        return merges_created
    
    def _create_synergy_merged_capsules(self, capsules: List[Dict]) -> List[str]:
        """Create merged capsules for direct synergy pairs."""
        merges_created = []
        
        # Define direct synergy pairs
        synergy_pairs = [
            (['sprocket', 'gear'], ['chain', 'belt'], 'transmission_system', 'mechanical_system'),
            (['piston'], ['cylinder'], 'piston_cylinder_system', 'engine_system'),
            (['gene'], ['protein'], 'gene_protein_system', 'biological_system'),
            (['tool'], ['material'], 'tool_material_system', 'construction_system')
        ]
        
        for cat1, cat2, merge_name, merge_type in synergy_pairs:
            cat1_components = []
            cat2_components = []
            
            # Find components in each category
            for capsule in capsules:
                cap_name = capsule.get('name', '').lower()
                if any(comp in cap_name for comp in cat1):
                    cat1_components.append(capsule)
                elif any(comp in cap_name for comp in cat2):
                    cat2_components.append(capsule)
            
            # Create merged capsule if we have components from both categories
            if cat1_components and cat2_components:
                all_components = cat1_components + cat2_components
                
                # Check if merged capsule already exists
                existing_merge = None
                for capsule in capsules:
                    if capsule.get('name') == merge_name:
                        existing_merge = capsule
                        break
                
                if not existing_merge:
                    merged_capsule = self._create_merged_capsule(
                        all_components,
                        merge_name,
                        merge_type,
                        f'Synergistic {merge_name.replace("_", " ")}',
                        'direct_synergy'
                    )
                    
                    if merged_capsule:
                        self.capsule_manager.add_capsule(merged_capsule)
                        merges_created.append(f"synergy_merge_{merge_name}")
                        
                        # Create lineage tracking
                        merged_capsule['lineage'] = {
                            'merge_type': 'direct_synergy',
                            'category1': cat1,
                            'category2': cat2,
                            'components': [c['name'] for c in all_components],
                            'created_timestamp': time.time()
                        }
        
        return merges_created
    
    def _create_merged_capsule(self, component_capsules: List[Dict], merge_name: str, 
                              merge_type: str, description: str, merge_reason: str) -> Dict:
        """Create a merged capsule from component capsules."""
        if not component_capsules:
            return None
        
        # Combine vectors using weighted average
        vectors = []
        weights = []
        
        for capsule in component_capsules:
            if 'vector' in capsule:
                vectors.append(np.array(capsule['vector']))
                # Weight by usage count and strength
                weight = capsule.get('usage_count', 1) * capsule.get('strength', 1.0)
                weights.append(weight)
        
        if vectors:
            # Create weighted average vector
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(w * v for w, v in zip(weights, vectors))
                merged_vector = weighted_sum / total_weight
                
                # Normalize
                merged_vector = merged_vector / np.linalg.norm(merged_vector)
                merged_vector = merged_vector.tolist()
            else:
                # Simple average if no weights
                merged_vector = np.mean(vectors, axis=0).tolist()
        else:
            merged_vector = None
        
        # Combine metadata
        combined_metadata = {
            'description': description,
            'merge_reason': merge_reason,
            'component_count': len(component_capsules),
            'component_types': list(set(c.get('capsule_type', 'unknown') for c in component_capsules)),
            'component_names': [c['name'] for c in component_capsules],
            'created_from_merge': True
        }
        
        # Merge traits if they exist
        all_traits = []
        for capsule in component_capsules:
            if 'metadata' in capsule and 'traits' in capsule['metadata']:
                all_traits.extend(capsule['metadata']['traits'])
        
        if all_traits:
            combined_metadata['traits'] = list(set(all_traits))
        
        # Create merged capsule
        merged_capsule = {
            'name': merge_name,
            'capsule_type': merge_type,
            'vector': merged_vector,
            'orbit_distance': 0.0,  # Start close to core
            'orbit_score': 50,  # Moderate initial score
            'usage_count': 1,
            'last_used_time': time.time(),
            'strength': 0.8,  # Good initial strength
            'metadata': combined_metadata,
            'pinned_to_core': False
        }
        
        return merged_capsule

class TribunalAspect(Enum):
    """Three aspects of the tribunal."""
    LOGICAL = "logical"
    CREATIVE = "creative"
    ETHICAL = "ethical"
    PRACTICAL = "practical"
    INTUITIVE = "intuitive"
    SYSTEMIC = "systemic"

@dataclass
class TribunalJudgment:
    """A judgment from the tribunal."""
    aspect: TribunalAspect
    score: float  # -1.0 to 1.0
    reasoning: str
    confidence: float = 0.8

class Tribunal:
    """Six-aspect tribunal for evaluating reasoning and knowledge."""
    
    def __init__(self):
        self.aspect_weights: Dict[TribunalAspect, float] = {
            TribunalAspect.LOGICAL: 1.0,
            TribunalAspect.CREATIVE: 1.0,
            TribunalAspect.ETHICAL: 1.0,
            TribunalAspect.PRACTICAL: 1.0,
            TribunalAspect.INTUITIVE: 1.0,
            TribunalAspect.SYSTEMIC: 1.0
        }
        self.judgment_history: List[TribunalJudgment] = []
        
        # Dialogue pattern learning
        self.dialogue_patterns: Dict[str, Dict] = {
            'subject_matters': defaultdict(int),  # topic -> frequency
            'question_types': defaultdict(int),    # question_pattern -> frequency
            'discussion_flows': [],                # sequence of topics
            'user_preferences': defaultdict(float), # preference -> strength
            'anticipated_needs': []                # predicted next queries
        }
        self.pattern_memory_size = 100  # How many interactions to remember
        
    def evaluate_reasoning(self, reasoning_context: Dict) -> List[TribunalJudgment]:
        """Evaluate a reasoning process with independent aspect searches and pattern learning."""
        judgments = []
        
        # Learn from dialogue patterns
        self._learn_dialogue_patterns(reasoning_context)
        
        # Anticipate user needs
        anticipated_needs = self._anticipate_user_needs(reasoning_context)
        reasoning_context['anticipated_needs'] = anticipated_needs
        
        # Check for knowledge synthesis opportunities based on query
        synthesis_opportunities = self._identify_synthesis_opportunities(reasoning_context)
        reasoning_context['synthesis_opportunities'] = synthesis_opportunities
        
        # Apply independent gravitational pulls for each active aspect
        aspect_capsules = self._apply_independent_gravitational_pulls(reasoning_context)
        reasoning_context['aspect_capsules'] = aspect_capsules
        
        # Logical evaluation with independent search
        logical_capsules = aspect_capsules.get(TribunalAspect.LOGICAL, [])
        logical_score = self._evaluate_logical(reasoning_context, logical_capsules)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.LOGICAL,
            score=logical_score,
            reasoning=f"Logical coherence assessment (found {len(logical_capsules)} relevant capsules)"
        ))
        
        # Creative evaluation with independent search
        creative_capsules = aspect_capsules.get(TribunalAspect.CREATIVE, [])
        creative_score = self._evaluate_creative(reasoning_context, creative_capsules)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.CREATIVE,
            score=creative_score,
            reasoning=f"Creative potential assessment (found {len(creative_capsules)} relevant capsules)"
        ))
        
        # Ethical evaluation
        ethical_score = self._evaluate_ethical(reasoning_context)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.ETHICAL,
            score=ethical_score,
            reasoning="Ethical consideration assessment"
        ))
        
        # Practical evaluation with independent search
        practical_capsules = aspect_capsules.get(TribunalAspect.PRACTICAL, [])
        practical_score = self._evaluate_practical(reasoning_context, practical_capsules)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.PRACTICAL,
            score=practical_score,
            reasoning=f"Practical feasibility assessment (found {len(practical_capsules)} relevant capsules)"
        ))
        
        # Intuitive evaluation
        intuitive_score = self._evaluate_intuitive(reasoning_context)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.INTUITIVE,
            score=intuitive_score,
            reasoning="Intuitive resonance assessment"
        ))
        
        # Systemic evaluation
        systemic_score = self._evaluate_systemic(reasoning_context)
        judgments.append(TribunalJudgment(
            aspect=TribunalAspect.SYSTEMIC,
            score=systemic_score,
            reasoning="Systemic impact assessment"
        ))
        
        self.judgment_history.extend(judgments)
        return judgments
    
    def _evaluate_logical(self, context: Dict, logical_capsules: List[Dict] = None) -> float:
        """Evaluate logical consistency."""
        base_score = random.uniform(-0.5, 0.8)
        
        # Boost score based on logical capsule findings
        if logical_capsules:
            capsule_boost = min(0.4, len(logical_capsules) * 0.15)
            relevance_boost = min(0.3, sum(cap['relevance'] for cap in logical_capsules[:3]) * 0.2)
            base_score = min(1.0, base_score + capsule_boost + relevance_boost)
            
        return base_score
    
    def _evaluate_creative(self, context: Dict, creative_capsules: List[Dict] = None) -> float:
        """Evaluate creative aspects."""
        base_score = random.uniform(-0.3, 0.9)
        
        # Boost score based on creative capsule findings
        if creative_capsules:
            # Different capsule types boost creativity
            types = set(cap['type'] for cap in creative_capsules)
            type_diversity_boost = min(0.25, len(types) * 0.08)
            
            # High relevance capsules boost creativity
            high_relevance = sum(1 for cap in creative_capsules if cap['relevance'] > 0.5)
            relevance_boost = min(0.35, high_relevance * 0.12)
            
            base_score = min(1.0, base_score + type_diversity_boost + relevance_boost)
            
        return base_score
    
    def _evaluate_ethical(self, context: Dict) -> float:
        """Evaluate ethical considerations."""
        return random.uniform(-0.2, 0.7)
    
    def _evaluate_practical(self, context: Dict, practical_capsules: List[Dict] = None) -> float:
        """Evaluate practical feasibility."""
        base_score = random.uniform(-0.4, 0.6)
        
        # Boost score based on practical capsule findings
        if practical_capsules:
            # Workflow and skill capsules strongly boost practicality
            workflow_boost = sum(0.15 for cap in practical_capsules if cap['type'] in ['workflow', 'skill'])
            memory_boost = min(0.2, len([cap for cap in practical_capsules if cap['type'] == 'memory']) * 0.1)
            
            base_score = min(1.0, base_score + workflow_boost + memory_boost)
            
        return base_score
    
    def _evaluate_intuitive(self, context: Dict) -> float:
        """Evaluate intuitive resonance."""
        return random.uniform(-0.1, 0.8)
    
    def _evaluate_systemic(self, context: Dict) -> float:
        """Evaluate systemic impact."""
        return random.uniform(-0.6, 0.5)
    
    def _apply_independent_gravitational_pulls(self, reasoning_context: Dict) -> Dict[TribunalAspect, List[Dict]]:
        """Apply independent gravitational pulls for each tribunal aspect."""
        query = reasoning_context.get('query', '').lower()
        capsule_manager = reasoning_context.get('capsule_manager')
        if not query or not capsule_manager:
            return {}
            
        aspect_capsules = {}
        
        # Logical aspect search - focuses on factual, analytical, structured knowledge
        logical_capsules = self._aspect_search_capsules(
            query, capsule_manager, TribunalAspect.LOGICAL,
            search_terms=['logic', 'fact', 'analysis', 'structure', 'data', 'evidence', 'proof', 'reasoning'],
            capsule_types=['personality', 'memory', 'topic', 'experimental']
        )
        aspect_capsules[TribunalAspect.LOGICAL] = logical_capsules
        
        # Creative aspect search - focuses on artistic, novel, inspirational knowledge
        creative_capsules = self._aspect_search_capsules(
            query, capsule_manager, TribunalAspect.CREATIVE,
            search_terms=['create', 'art', 'design', 'imagine', 'inspire', 'novel', 'style', 'aesthetic'],
            capsule_types=['style', 'character', 'personality', 'experimental']
        )
        aspect_capsules[TribunalAspect.CREATIVE] = creative_capsules
        
        # Practical aspect search - focuses on actionable, implementation knowledge
        practical_capsules = self._aspect_search_capsules(
            query, capsule_manager, TribunalAspect.PRACTICAL,
            search_terms=['implement', 'action', 'build', 'execute', 'tool', 'method', 'process', 'workflow'],
            capsule_types=['skill', 'workflow', 'memory', 'topic']
        )
        aspect_capsules[TribunalAspect.PRACTICAL] = practical_capsules
        
        return aspect_capsules
    
    def _aspect_search_capsules(self, query: str, capsule_manager, aspect: TribunalAspect, 
                               search_terms: List[str], capsule_types: List[str]) -> List[Dict]:
        """Search for capsules relevant to a specific tribunal aspect."""
        relevant_capsules = []
        
        for capsule in capsule_manager.capsules:
            relevance_score = 0.0
            aspect_bonus = 0.0
            
            capsule_name = capsule.get('name', '').lower()
            capsule_type = capsule.get('capsule_type', '').lower()
            metadata = capsule.get('metadata', {})
            
            # Base relevance from query matching
            query_words = query.split()
            for word in query_words:
                if word in capsule_name:
                    relevance_score += 0.4
                if isinstance(metadata, dict):
                    metadata_str = json.dumps(metadata).lower()
                    if word in metadata_str:
                        relevance_score += 0.3
            
            # Aspect-specific relevance
            for term in search_terms:
                if term in query or term in capsule_name:
                    aspect_bonus += 0.2
                if term in capsule_type:
                    aspect_bonus += 0.3
            
            # Capsule type preference for this aspect
            if capsule_type in capsule_types:
                aspect_bonus += 0.25
            
            total_relevance = relevance_score + aspect_bonus
            
            if total_relevance > 0.2:  # Lower threshold for aspect-specific search
                # Apply gravitational pull
                current_score = capsule.get('orbit_score', 0)
                pull_strength = int(total_relevance * 15)  # Aspect-specific pull strength
                new_score = min(100, current_score + pull_strength)
                capsule['orbit_score'] = new_score
                capsule['orbit_distance'] = max(0.0, 1.0 - (new_score / 100.0))
                capsule['last_used_time'] = time.time()
                capsule['usage_count'] = capsule.get('usage_count', 0) + 1
                
                relevant_capsules.append({
                    'name': capsule['name'],
                    'type': capsule['capsule_type'],
                    'relevance': total_relevance,
                    'orbit_score': new_score,
                    'aspect': aspect.name
                })
        
        # Sort by relevance for this aspect
        relevant_capsules.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Save updated capsules
        capsule_manager._save_capsules()
        
        return relevant_capsules
    
    def _learn_dialogue_patterns(self, reasoning_context: Dict):
        """Learn patterns from user dialogue for anticipation."""
        query = reasoning_context.get('query', '').lower()
        if not query:
            return
            
        # Extract subject matters
        subject_keywords = {
            'math': ['math', 'calculate', 'equation', 'algebra', 'geometry', 'calculus'],
            'physics': ['physics', 'force', 'energy', 'motion', 'gravity', 'quantum'],
            'art': ['art', 'design', 'draw', 'paint', 'create', 'style', 'aesthetic'],
            'programming': ['code', 'program', 'python', 'algorithm', 'function', 'class'],
            'animation': ['animate', 'frame', 'timeline', 'character', 'motion', 'pose'],
            'ai': ['ai', 'machine learning', 'neural', 'intelligence', 'learn', 'model']
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in query for keyword in keywords):
                self.dialogue_patterns['subject_matters'][subject] += 1
        
        # Extract question types
        question_patterns = {
            'definition': ['what is', 'define', 'meaning of', 'explain'],
            'how_to': ['how to', 'how do', 'how can', 'steps to'],
            'why': ['why', 'reason', 'because', 'cause'],
            'comparison': ['difference between', 'compare', 'versus', 'vs'],
            'example': ['example', 'instance', 'case of', 'show me']
        }
        
        for q_type, patterns in question_patterns.items():
            if any(pattern in query for pattern in patterns):
                self.dialogue_patterns['question_types'][q_type] += 1
        
        # Track discussion flow (maintain last N topics)
        current_topics = []
        for subject, keywords in subject_keywords.items():
            if any(keyword in query for keyword in keywords):
                current_topics.append(subject)
        
        if current_topics:
            self.dialogue_patterns['discussion_flows'].extend(current_topics)
            # Keep only recent topics
            if len(self.dialogue_patterns['discussion_flows']) > self.pattern_memory_size:
                self.dialogue_patterns['discussion_flows'] = self.dialogue_patterns['discussion_flows'][-self.pattern_memory_size:]
    
    def _anticipate_user_needs(self, reasoning_context: Dict) -> List[str]:
        """Anticipate user needs based on learned patterns."""
        anticipated = []
        
        # Anticipate based on subject matter preferences
        top_subjects = sorted(self.dialogue_patterns['subject_matters'].items(), 
                            key=lambda x: x[1], reverse=True)[:2]
        
        # Anticipate based on question types
        top_question_types = sorted(self.dialogue_patterns['question_types'].items(),
                                  key=lambda x: x[1], reverse=True)[:2]
        
        # Generate anticipated queries based on patterns
        for subject, _ in top_subjects:
            for q_type, _ in top_question_types:
                if subject == 'math' and q_type == 'definition':
                    anticipated.append("What is a derivative?")
                elif subject == 'physics' and q_type == 'how_to':
                    anticipated.append("How do forces work?")
                elif subject == 'art' and q_type == 'example':
                    anticipated.append("Show me color theory examples")
                elif subject == 'programming' and q_type == 'why':
                    anticipated.append("Why use object-oriented programming?")
                elif subject == 'animation' and q_type == 'comparison':
                    anticipated.append("Compare 2D vs 3D animation")
                elif subject == 'ai' and q_type == 'how_to':
                    anticipated.append("How does machine learning work?")
        
        # Anticipate based on discussion flow
        recent_topics = self.dialogue_patterns['discussion_flows'][-5:]  # Last 5 topics
        if len(recent_topics) >= 2:
            # If discussing related topics, anticipate connections
            if 'animation' in recent_topics and 'art' in recent_topics:
                anticipated.append("How does art influence animation?")
            if 'programming' in recent_topics and 'ai' in recent_topics:
                anticipated.append("How is AI programmed?")
        
        # Limit to top 3 anticipations
        return anticipated[:3]
    
    def _identify_synthesis_opportunities(self, reasoning_context: Dict) -> List[Dict]:
        """Identify opportunities to synthesize existing knowledge for the current query."""
        query = reasoning_context.get('query', '').lower()
        capsule_manager = reasoning_context.get('capsule_manager')
        
        if not query or not capsule_manager:
            return []
        
        opportunities = []
        
        # Look for existing merged capsules that could be relevant
        merged_capsule_matches = self._find_merged_capsule_matches(query, capsule_manager.capsules)
        opportunities.extend(merged_capsule_matches)
        
        # Look for scenario-based synthesis (like sprocket + chain for bike)
        scenario_matches = self._find_scenario_matches(query, capsule_manager.capsules)
        opportunities.extend(scenario_matches)
        
        # Look for complementary knowledge synthesis
        complementary_matches = self._find_complementary_matches(query, capsule_manager.capsules)
        opportunities.extend(complementary_matches)
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def _find_merged_capsule_matches(self, query: str, capsules: List[Dict]) -> List[Dict]:
        """Find existing merged capsules that could be relevant to the query."""
        matches = []
        
        # Find merged capsules (those created from synthesis)
        merged_capsules = [c for c in capsules if c.get('metadata', {}).get('created_from_merge', False)]
        
        for capsule in merged_capsules:
            cap_name = capsule.get('name', '').lower()
            merge_reason = capsule.get('metadata', {}).get('merge_reason', '')
            
            # Check if capsule name or description relates to query
            description = capsule.get('metadata', {}).get('description', '').lower()
            
            relevance_score = 0
            if any(word in query for word in cap_name.split('_')):
                relevance_score += 0.8
            if any(word in query for word in description.split()):
                relevance_score += 0.6
            
            if relevance_score > 0:
                matches.append({
                    'type': 'existing_merged_capsule',
                    'capsule': capsule,
                    'merge_reason': merge_reason,
                    'component_count': capsule.get('metadata', {}).get('component_count', 0),
                    'component_names': capsule.get('metadata', {}).get('component_names', []),
                    'relevance_score': relevance_score,
                    'suggestion': f"Apply existing merged knowledge: {capsule['name']} ({description})"
                })
        
        # Sort by relevance score
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches
    
    def _find_scenario_matches(self, query: str, capsules: List[Dict]) -> List[Dict]:
        """Find capsules that could be synthesized for scenario-based applications."""
        matches = []
        
        # Scenario templates with their component keywords
        scenarios = {
            'bicycle': {
                'components': ['wheel', 'frame', 'pedal', 'chain', 'sprocket', 'brake', 'seat', 'handlebar'],
                'application': 'building a bicycle',
                'relevance_keywords': ['bike', 'bicycle', 'cycling', 'transport', 'vehicle']
            },
            'computer': {
                'components': ['processor', 'memory', 'storage', 'display', 'keyboard', 'software', 'network'],
                'application': 'building a computer system',
                'relevance_keywords': ['computer', 'pc', 'system', 'computing', 'hardware', 'software']
            },
            'engine': {
                'components': ['piston', 'cylinder', 'crankshaft', 'valve', 'fuel', 'spark', 'combustion'],
                'application': 'building an engine',
                'relevance_keywords': ['engine', 'motor', 'power', 'mechanical', 'combustion']
            },
            'building': {
                'components': ['foundation', 'wall', 'roof', 'door', 'window', 'plumbing', 'electrical'],
                'application': 'constructing a building',
                'relevance_keywords': ['build', 'construct', 'house', 'structure', 'architecture']
            },
            'painting': {
                'components': ['canvas', 'brush', 'paint', 'color', 'technique', 'style', 'composition'],
                'application': 'creating a painting',
                'relevance_keywords': ['paint', 'artwork', 'drawing', 'visual', 'creative']
            }
        }
        
        # Check if query relates to any scenario
        relevant_scenarios = []
        for scenario_name, scenario_data in scenarios.items():
            if any(keyword in query for keyword in scenario_data['relevance_keywords']):
                relevant_scenarios.append((scenario_name, scenario_data))
        
        # For each relevant scenario, find available components
        for scenario_name, scenario_data in relevant_scenarios:
            available_components = []
            
            for capsule in capsules:
                cap_name = capsule.get('name', '').lower()
                # Check if capsule matches any component
                for component in scenario_data['components']:
                    if component in cap_name or any(word in cap_name for word in component.split()):
                        available_components.append({
                            'capsule': capsule,
                            'component': component,
                            'relevance': 0.8 if component in cap_name else 0.6
                        })
                        break
            
            if len(available_components) >= 2:  # Need at least 2 components
                matches.append({
                    'type': 'scenario_synthesis',
                    'scenario': scenario_name,
                    'application': scenario_data['application'],
                    'available_components': available_components,
                    'completeness': len(available_components) / len(scenario_data['components']),
                    'suggestion': f"Apply {len(available_components)} known components to {scenario_data['application']}"
                })
        
        return matches
    
    def _find_complementary_matches(self, query: str, capsules: List[Dict]) -> List[Dict]:
        """Find complementary knowledge that could enhance understanding of the query."""
        matches = []
        
        # Define complementary knowledge pairs
        complementary_pairs = [
            (['sprocket', 'gear', 'wheel'], ['chain', 'belt', 'transmission'], 'mechanical power transmission'),
            (['processor', 'cpu'], ['memory', 'ram', 'storage'], 'computing system'),
            (['color', 'hue'], ['shape', 'form', 'composition'], 'visual design'),
            (['rhythm', 'tempo'], ['melody', 'harmony', 'pitch'], 'musical composition'),
            (['character', 'person'], ['setting', 'environment', 'world'], 'narrative construction'),
            (['force', 'energy'], ['motion', 'velocity', 'acceleration'], 'physical dynamics'),
            (['gene', 'dna'], ['protein', 'enzyme', 'metabolism'], 'biological processes'),
            (['tool', 'instrument'], ['material', 'substance', 'medium'], 'creation methods'),
            (['problem', 'issue'], ['solution', 'method', 'approach'], 'problem solving'),
            (['theory', 'concept'], ['evidence', 'data', 'experiment'], 'scientific validation')
        ]
        
        query_words = set(query.split())
        
        for category1, category2, application in complementary_pairs:
            # Check if query relates to either category
            relates_to_cat1 = any(word in query for word in category1)
            relates_to_cat2 = any(word in query for word in category2)
            
            if relates_to_cat1 or relates_to_cat2:
                # Find capsules in both categories
                cat1_capsules = []
                cat2_capsules = []
                
                for capsule in capsules:
                    cap_name = capsule.get('name', '').lower()
                    if any(comp in cap_name for comp in category1):
                        cat1_capsules.append(capsule)
                    elif any(comp in cap_name for comp in category2):
                        cat2_capsules.append(capsule)
                
                if cat1_capsules and cat2_capsules:
                    matches.append({
                        'type': 'complementary_synthesis',
                        'application': application,
                        'category1': category1,
                        'category2': category2,
                        'cat1_capsules': cat1_capsules,
                        'cat2_capsules': cat2_capsules,
                        'suggestion': f"Combine {len(cat1_capsules)} {category1[0]}-related concepts with {len(cat2_capsules)} {category2[0]}-related concepts for {application}"
                    })
        
        return matches
    
    def evolve_weights(self, feedback: Dict[TribunalAspect, float]):
        """Evolve tribunal weights based on feedback."""
        for aspect, weight_change in feedback.items():
            self.aspect_weights[aspect] = max(0.1, min(2.0, 
                self.aspect_weights[aspect] + weight_change * 0.1))

class HardcodedKnowledge:
    """Immutable hardcoded math and physics knowledge for noise-free reference."""
    
    MATH_TABLES = {
        "trigonometric_identities": {
            "sin²θ + cos²θ": "1",
            "tanθ": "sinθ / cosθ",
            "sin(2θ)": "2 sinθ cosθ",
            "cos(2θ)": "cos²θ - sin²θ",
            "sin(A+B)": "sin A cos B + cos A sin B"
        },
        "logarithmic_properties": {
            "log(a*b)": "log(a) + log(b)",
            "log(a/b)": "log(a) - log(b)",
            "log(a^n)": "n log(a)",
            "ln(e)": "1",
            "log₁₀(10)": "1"
        },
        "exponential_properties": {
            "e^(ln x)": "x",
            "ln(e^x)": "x",
            "(e^a)^b": "e^(a*b)"
        }
    }
    
    TENSOR_RULES = {
        "contraction": "Summation over repeated indices (Einstein notation)",
        "symmetric_tensor": "T_ij = T_ji",
        "antisymmetric_tensor": "T_ij = -T_ji",
        "levi_civita": "ε_ijk antisymmetric, ε_123 = 1",
        "kronecker_delta": "δ_ij = 1 if i=j, 0 otherwise",
        "metric_tensor": "g_μν defines spacetime geometry"
    }
    
    DIFFERENTIAL_OPERATORS = {
        "gradient": "∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)",
        "divergence": "∇·F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z",
        "curl": "∇×F = (∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y)",
        "laplacian": "∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²",
        "d_alembertian": "□ = ∂²/∂t² - ∇² (in Minkowski space)"
    }
    
    CONSERVATION_LAWS = {
        "mass": "Mass cannot be created or destroyed (continuity equation: ∂ρ/∂t + ∇·(ρv) = 0)",
        "energy": "Energy is conserved in isolated systems (first law of thermodynamics)",
        "momentum": "Linear momentum is conserved in the absence of external forces",
        "angular_momentum": "Angular momentum is conserved in the absence of external torques",
        "charge": "Electric charge is conserved",
        "baryon_number": "Baryon number is conserved in strong and electromagnetic interactions"
    }
    
    PHYSICAL_CONSTANTS = {
        "speed_of_light": "c = 299,792,458 m/s (exact)",
        "planck_constant": "h = 6.62607015 × 10^-34 J⋅s",
        "reduced_planck_constant": "ℏ = h/(2π) = 1.0545718 × 10^-34 J⋅s",
        "gravitational_constant": "G = 6.67430 × 10^-11 m³ kg⁻¹ s⁻²",
        "boltzmann_constant": "k_B = 1.380649 × 10^-23 J/K",
        "avogadro_number": "N_A = 6.02214076 × 10^23 mol⁻¹",
        "elementary_charge": "e = 1.602176634 × 10^-19 C",
        "vacuum_permeability": "μ₀ = 4π × 10^-7 N/A²",
        "vacuum_permittivity": "ε₀ = 8.854187817 × 10^-12 F/m"
    }
    
    # Philosophical foundations for scientific reasoning
    EPISTEMOLOGY = {
        "empirical_knowledge": "Knowledge derived from observation and experiment",
        "a_priori_knowledge": "Knowledge independent of experience (mathematics, logic)",
        "a_posteriori_knowledge": "Knowledge dependent on experience (science)",
        "justified_true_belief": "Classical definition: knowledge = justified true belief",
        "falsifiability": "Scientific theories must be testable and potentially falsifiable (Popper)",
        "paradigm_shifts": "Revolutionary changes in scientific frameworks (Kuhn)",
        "underdetermination": "Multiple theories can explain the same evidence equally well",
        "theory_laden_observation": "Observations are influenced by theoretical frameworks"
    }
    
    ONTOLOGY = {
        "realism": "The world exists independently of our perception",
        "anti_realism": "Reality depends on our conceptual schemes or perceptions",
        "scientific_realism": "Scientific theories describe real entities and processes",
        "instrumentalism": "Scientific theories are useful tools, not literal descriptions",
        "structural_realism": "Only the structural aspects of theories are real",
        "nominalism": "Universals don't exist; only particulars exist",
        "platonism": "Mathematical entities exist independently (forms, numbers)",
        "physicalism": "Everything is ultimately physical or reducible to physics",
        "emergence": "Complex systems exhibit properties not present in their parts"
    }
    
    PHILOSOPHY_OF_SCIENCE = {
        "scientific_method": "Observation → Hypothesis → Experiment → Theory → Prediction",
        "hypothetico_deductive": "Form hypotheses, deduce predictions, test predictions",
        "abductive_reasoning": "Infer best explanation for observations",
        "corroboration": "Evidence supports but never proves a theory",
        "anomaly_resolution": "Theories change when anomalies accumulate (Kuhn)",
        "progressive_research_programs": "Successful programs predict novel facts (Lakatos)",
        "model_vs_reality": "Models are approximations; distinguish map from territory",
        "mathematical_idealization": "Physics uses idealized mathematics (point masses, frictionless surfaces)",
        "gauge_invariance": "Physical theories invariant under certain transformations",
        "effective_theories": "Approximate theories valid within certain domains",
        "renormalization": "Handling infinities in quantum field theory",
        "symmetry_breaking": "How symmetries in laws lead to asymmetries in states"
    }
    
    PHILOSOPHICAL_PRINCIPLES = {
        "occams_razor": "Prefer simpler explanations (entities should not be multiplied unnecessarily)",
        "principle_of_falsifiability": "Claims must be testable and potentially refutable",
        "duhem_quine_thesis": "Theories are tested as wholes, not in isolation",
        "pessimistic_induction": "Most past theories were wrong; current ones likely are too",
        "no_miracles_argument": "Scientific success would be miraculous without approximate truth",
        "inference_to_best_explanation": "Choose theory that best explains all evidence",
        "coherence_theory": "Truth is coherence with our belief system",
        "correspondence_theory": "Truth corresponds to reality",
        "pragmatic_theory": "Truth is what works in practice"
    }
    
    # Fundamental mathematical and logical rules
    ARITHMETIC_RULES = {
        "commutative_addition": "a + b = b + a",
        "commutative_multiplication": "a × b = b × a",
        "associative_addition": "(a + b) + c = a + (b + c)",
        "associative_multiplication": "(a × b) × c = a × (b × c)",
        "distributive_property": "a × (b + c) = a × b + a × c",
        "additive_identity": "a + 0 = a",
        "multiplicative_identity": "a × 1 = a",
        "additive_inverse": "a + (-a) = 0",
        "multiplicative_inverse": "a × (1/a) = 1 (for a ≠ 0)",
        "zero_property": "a × 0 = 0"
    }
    
    ALGEBRAIC_MANIPULATION = {
        "addition_both_sides": "If a = b, then a + c = b + c",
        "multiplication_both_sides": "If a = b and c ≠ 0, then a × c = b × c",
        "division_both_sides": "If a = b and c ≠ 0, then a/c = b/c",
        "squaring_both_sides": "If a = b and a,b ≥ 0, then a² = b² (be careful with extraneous solutions)",
        "taking_reciprocal": "If a = b and a,b ≠ 0, then 1/a = 1/b",
        "cross_multiplication": "If a/b = c/d and b,d ≠ 0, then a × d = b × c",
        "factoring": "ab + ac = a(b + c)",
        "difference_of_squares": "a² - b² = (a - b)(a + b)",
        "perfect_square_trinomial": "(a + b)² = a² + 2ab + b²",
        "quadratic_formula": "For ax² + bx + c = 0, x = [-b ± √(b² - 4ac)]/(2a)"
    }
    
    CALCULUS_OPERATORS = {
        "derivative_power_rule": "d/dx[x^n] = n x^(n-1)",
        "derivative_constant": "d/dx[c] = 0",
        "derivative_sum": "d/dx[f(x) + g(x)] = f'(x) + g'(x)",
        "derivative_product": "d/dx[f(x) × g(x)] = f'(x) × g(x) + f(x) × g'(x)",
        "derivative_quotient": "d/dx[f(x)/g(x)] = [f'(x) × g(x) - f(x) × g'(x)] / [g(x)]²",
        "derivative_chain_rule": "d/dx[f(g(x))] = f'(g(x)) × g'(x)",
        "integral_power_rule": "∫ x^n dx = x^(n+1)/(n+1) + C (for n ≠ -1)",
        "integral_constant": "∫ c dx = c x + C",
        "integral_sum": "∫ [f(x) + g(x)] dx = ∫ f(x) dx + ∫ g(x) dx",
        "fundamental_theorem": "d/dx[∫ f(x) dx] = f(x)",
        "integration_by_parts": "∫ u dv = u v - ∫ v du",
        "substitution_rule": "∫ f(g(x)) g'(x) dx = ∫ f(u) du where u = g(x)"
    }
    
    LINEAR_ALGEBRA_OPERATIONS = {
        "matrix_addition": "A + B = [a_ij + b_ij]",
        "matrix_multiplication": "C = A × B where c_ij = Σ_k a_ik × b_kj",
        "matrix_transpose": "(A^T)_ij = A_ji",
        "matrix_inverse": "A × A^(-1) = I (identity matrix)",
        "determinant_2x2": "det([a b; c d]) = a d - b c",
        "determinant_properties": "det(A × B) = det(A) × det(B), det(A^T) = det(A)",
        "eigenvalues": "For A v = λ v, λ are eigenvalues, v are eigenvectors",
        "vector_dot_product": "a · b = Σ_i a_i b_i = |a| |b| cosθ",
        "vector_cross_product": "a × b = |i    j    k;  a_x a_y a_z;  b_x b_y b_z|",
        "linear_independence": "Vectors are independent if no non-trivial linear combination equals zero",
        "rank_nullity": "rank(A) + nullity(A) = number of columns"
    }
    
    TENSOR_CONTRACTION_RULES = {
        "einstein_summation": "Repeated indices are summed over (Einstein notation)",
        "contraction": "T^i_i = Σ_i T^i_i (trace)",
        "raising_lowering": "T_i = g_ij T^j (using metric tensor)",
        "levi_civita_contraction": "ε_ijk ε^ijk = 6",
        "kronecker_contraction": "δ^i_i = dimension of space",
        "tensor_product": "(A ⊗ B)_ijkl = A_ij B_kl",
        "outer_product": "Preserves all indices: C_ij = A_i B_j",
        "inner_product": "Contracts one index from each: C_j = A^i B_i",
        "covariant_derivative": "∇_μ V^ν = ∂_μ V^ν + Γ^ν_μσ V^σ"
    }
    
    PROBABILITY_AXIOMS = {
        "non_negativity": "P(A) ≥ 0 for any event A",
        "normalization": "P(Ω) = 1 where Ω is the sample space",
        "additivity": "P(A ∪ B) = P(A) + P(B) if A and B are mutually exclusive",
        "conditional_probability": "P(A|B) = P(A ∩ B) / P(B) if P(B) > 0",
        "bayes_theorem": "P(A|B) = P(B|A) P(A) / P(B)",
        "independence": "P(A ∩ B) = P(A) × P(B) if A and B are independent",
        "law_of_total_probability": "P(B) = Σ_i P(B|A_i) P(A_i)",
        "complement_rule": "P(A^c) = 1 - P(A)",
        "multiplication_rule": "P(A ∩ B) = P(A) × P(B|A)"
    }
    
    LOGIC_OPERATORS = {
        "and_conjunction": "A ∧ B is true only if both A and B are true",
        "or_disjunction": "A ∨ B is true if at least one of A or B is true",
        "not_negation": "¬A is true if A is false, and vice versa",
        "implies_conditional": "A → B means 'if A then B' (false only if A true and B false)",
        "biconditional": "A ↔ B means A → B and B → A",
        "de_morgan_and": "¬(A ∧ B) ≡ ¬A ∨ ¬B",
        "de_morgan_or": "¬(A ∨ B) ≡ ¬A ∧ ¬B",
        "double_negation": "¬¬A ≡ A",
        "contrapositive": "A → B is equivalent to ¬B → ¬A",
        "modus_ponens": "From A and A → B, conclude B",
        "modus_tollens": "From ¬B and A → B, conclude ¬A"
    }
    
    SET_OPERATIONS = {
        "union": "A ∪ B = {x | x ∈ A or x ∈ B}",
        "intersection": "A ∩ B = {x | x ∈ A and x ∈ B}",
        "complement": "A^c = {x ∈ U | x ∉ A} where U is universal set",
        "difference": "A - B = {x ∈ A | x ∉ B}",
        "symmetric_difference": "A Δ B = (A - B) ∪ (B - A)",
        "cartesian_product": "A × B = {(a,b) | a ∈ A and b ∈ B}",
        "power_set": "P(A) = {B | B ⊆ A}",
        "subset": "A ⊆ B means every element of A is also in B",
        "proper_subset": "A ⊂ B means A ⊆ B and A ≠ B",
        "cardinality": "|A| is the number of elements in set A",
        "empty_set": "∅ = {} (the set with no elements)",
        "universal_set": "U contains all elements under consideration"
    }
    
    # Capsule system mechanisms - immutable knowledge of how knowledge evolves
    CAPSULE_SYSTEM_MECHANISMS = {
        "capsule_splitting": "Capsules split when contradictions detected. Original capsule removed, two variants created with '_variant1' and '_variant2' suffixes. Vectors perturbed with Gaussian noise (σ=0.1). Conflicting traits removed from respective variants. Initial strength reduced to 60% of original.",
        
        "capsule_merging": "Similar capsules (cosine similarity > 0.85) merge automatically. Vectors combined with weighted average (30% new, 70% existing). Usage count incremented, strength increased by 0.1 (max 1.0). Metadata merged with conflict resolution favoring existing values.",
        
        "confidence_decay": "Capsule strength decays over time based on non-usage. Decay rate = 0.01 per day since last use. Strength = max(0.1, current_strength - decay_factor). Capsules below minimum strength (0.1) are removed unless pinned to core.",
        
        "uncertainty_storage": "Uncertainty represented through vector embeddings and metadata conflicts. High-dimensional vectors capture semantic uncertainty. Contradictory metadata (opposing traits, conflicting preferences) triggers splitting. Strength values indicate confidence levels.",
        
        "revision_history_preservation": "Revisions never erase history - new capsule variants created instead of overwriting. Original capsules remain accessible. Split capsules maintain full metadata history. Merge operations preserve both contributing capsules' information. Time-stamped evolution tracking.",
        
        "similarity_detection": "Cosine similarity measures vector alignment. Threshold = 0.85 for merging. Similarity = dot_product / (||v1|| * ||v2||). Used for automatic knowledge consolidation and redundancy elimination.",
        
        "strength_dynamics": "Strength increases with usage (+0.1 per merge/use, max 1.0). Decays with time (-0.01/day unused). Minimum strength = 0.1. Core capsules pinned to prevent decay. Strength influences retrieval priority and evolution decisions.",
        
        "vector_perturbation": "During splitting, Gaussian noise (μ=0, σ=0.1) added to vectors for differentiation. Ensures split capsules have distinct representations while maintaining semantic proximity. Normalization applied after perturbation.",
        
        "metadata_conflict_resolution": "Merge conflicts resolved by favoring existing capsule data. Lists extended and deduplicated. Non-list conflicts retain existing values. Split conflicts remove contradictory traits from respective variants.",
        
        "temporal_evolution_tracking": "Each capsule tracks: creation_timestamp, last_used_time, usage_count. Evolution events logged. Time-based decay calculated from last_used_time. Historical variants preserved for analysis."
    }
    
    # ============================================================================
    # LANGUAGE AND SYMBOL SYSTEMS
    # ============================================================================
    
    GRAMMAR = {
        "sentence_structure": "Sentences follow Subject-Verb-Object (SVO) order in English. Subject performs action, verb expresses action/state, object receives action. Complex sentences use clauses connected by conjunctions (and, but, or, so, because).",
        
        "parts_of_speech": "Eight main parts: nouns (people/places/things), verbs (actions/states), adjectives (descriptions), adverbs (how/when/where), pronouns (noun substitutes), prepositions (relationships), conjunctions (connectors), interjections (exclamations).",
        
        "tense_system": "Present (happening now), past (happened before), future (will happen). Perfect tenses show completion. Progressive tenses show ongoing action. Conditional tenses show hypothetical situations.",
        
        "subject_verb_agreement": "Singular subjects take singular verbs (he runs), plural subjects take plural verbs (they run). Compound subjects connected by 'and' are plural. Subjects joined by 'or/nor' agree with nearest noun.",
        
        "pronoun_reference": "Pronouns must clearly refer to antecedents. Avoid ambiguous pronouns. Pronouns agree in number/gender with nouns they replace. Reflexive pronouns (himself) refer back to subject.",
        
        "parallel_structure": "Similar elements in lists/series must have same grammatical form. 'I like swimming, biking, and running' (all gerunds). Maintain consistency in compound structures.",
        
        "active_passive_voice": "Active: subject performs action (Dog bites man). Passive: subject receives action (Man is bitten by dog). Passive emphasizes receiver, active emphasizes doer. Use active for clarity/conciseness.",
        
        "clause_types": "Independent clauses express complete thoughts. Dependent clauses need independent clauses. Relative clauses modify nouns. Adverb clauses modify verbs. Noun clauses act as nouns.",
        
        "punctuation_rules": "Period ends statements. Question mark ends questions. Exclamation point shows strong emotion. Comma separates items/clauses. Semicolon joins related independent clauses. Colon introduces lists/explanations.",
        
        "article_usage": "Definite article 'the' specifies particular noun. Indefinite articles 'a/an' refer to general/non-specific nouns. 'A' before consonant sounds, 'an' before vowel sounds. Zero article with plural/general nouns."
    }
    
    SYNTAX = {
        "phrase_structure": "Noun phrases (NP) contain nouns + modifiers. Verb phrases (VP) contain verbs + objects/complements. Prepositional phrases (PP) begin with prepositions. Adjective/adverb phrases modify nouns/verbs.",
        
        "word_order": "English follows strict Subject-Verb-Object word order. Questions invert subject-verb (Did you go?). Adjectives precede nouns (red house). Adverbs can precede/follow verbs (quickly ran vs ran quickly).",
        
        "constituent_structure": "Sentences have hierarchical structure. S → NP + VP. NP → (Det) + (Adj) + N. VP → V + (NP) + (PP). Constituents can be moved/replaced as units.",
        
        "transformational_rules": "Passive transformation: NP1 + V + NP2 → NP2 + be + V-en + by + NP1. Question formation: Move auxiliary to front. Negation: Insert 'not' after first auxiliary.",
        
        "embedding": "Sentences can contain other sentences as constituents. Relative clauses embed in noun phrases. Complement clauses embed in verb phrases. Center embedding creates complexity but maintains structure.",
        
        "agreement_rules": "Subjects agree with verbs in person/number. Pronouns agree with antecedents in gender/number/case. Determiners agree with nouns in number/definiteness.",
        
        "case_system": "Nominative case for subjects. Accusative case for direct objects. Dative case for indirect objects. Genitive case for possession. English marks case mainly through word order/pronouns.",
        
        "movement_operations": "Wh-movement: Move question words to front (What did you see?). Topicalization: Move topic to front. Clefting: Focus constituent with 'it is/was'.",
        
        "binding_theory": "Principle A: Reflexives must be bound in local domain. Principle B: Pronouns cannot be bound in local domain. Principle C: R-expressions cannot be bound.",
        
        "x_bar_theory": "All phrases have same structure: X' → X + complement, X' → specifier + X'. Heads determine phrase type. Complements satisfy head requirements. Specifiers add additional information."
    }
    
    SYMBOLIC_PARSING = {
        "parsing_algorithms": "Top-down parsing starts from root, predicts expansions. Bottom-up parsing starts from leaves, reduces to root. Chart parsing combines both approaches. Earley parser handles all CFGs.",
        
        "context_free_grammars": "CFG = (N, Σ, P, S) where N=nonterminals, Σ=terminals, P=production rules, S=start symbol. Rules: A → α where A∈N, α∈(N∪Σ)*. Generate languages through derivations.",
        
        "parse_trees": "Tree structure showing derivation. Root = start symbol. Internal nodes = nonterminals. Leaves = terminals. Each node expansion follows grammar rules. Tree depth shows embedding levels.",
        
        "ambiguity_resolution": "Lexical ambiguity: Multiple word meanings. Structural ambiguity: Multiple parse trees. Attachment ambiguity: PP attachment. Semantic ambiguity: Multiple interpretations. Pragmatic ambiguity: Context-dependent.",
        
        "finite_state_automata": "States represent parsing positions. Transitions triggered by input symbols. Accept states indicate successful parses. Deterministic FSA: one transition per state/symbol. Non-deterministic FSA: multiple transitions.",
        
        "pushdown_automata": "Extended FSA with stack. Stack handles recursion/context-sensitivity. Push/pop operations. PDA accepts context-free languages. Stack depth tracks nesting levels.",
        
        "lexical_analysis": "Tokenization: Split input into words/symbols. Lexical categories: Assign part-of-speech tags. Morphological analysis: Stemming/lemmatization. Named entity recognition: Identify proper nouns.",
        
        "semantic_parsing": "Map syntactic structures to meaning representations. Lambda calculus for composition. Semantic composition rules. Discourse representation theory for context. Frame semantics for world knowledge.",
        
        "dependency_parsing": "Words connected by directed edges. Head-dependent relationships. Projective vs non-projective dependencies. Universal dependencies framework. Transition-based parsing algorithms.",
        
        "constituency_parsing": "Hierarchical phrase structure. Constituents as subtrees. PCFG (Probabilistic CFG) with probability weights. CKY algorithm for parsing. Feature-based grammars with unification."
    }
    
    # ============================================================================
    # COMPUTATIONAL ENGINES & MEMORY SYSTEMS
    # ============================================================================
    
    MATH_ENGINE = {
        "symbolic_computation": "Manipulate mathematical expressions symbolically without numerical evaluation. Maintain exact representations. Handle algebraic transformations, equation solving, differentiation, integration. Preserve mathematical structure and relationships.",
        
        "numerical_precision": "Track precision and error bounds in floating-point computations. Use arbitrary-precision arithmetic when needed. Implement interval arithmetic for uncertainty quantification. Detect numerical instability and overflow/underflow conditions.",
        
        "equation_solving": "Solve systems of equations using appropriate methods: direct (Gaussian elimination), iterative (Jacobi, Gauss-Seidel), nonlinear (Newton-Raphson, bisection). Handle over/under-determined systems. Provide existence/uniqueness conditions.",
        
        "optimization_algorithms": "Gradient descent, Newton's method, quasi-Newton (BFGS), conjugate gradient. Handle constrained optimization with Lagrange multipliers, penalty methods. Global optimization with simulated annealing, genetic algorithms.",
        
        "matrix_computations": "Efficient matrix operations: LU/QR/Cholesky decomposition, eigenvalue/eigenvector computation, singular value decomposition. Sparse matrix handling. Parallel computation for large matrices.",
        
        "differential_equations": "Solve ODEs/PDEs using finite difference, finite element, spectral methods. Boundary/initial value problems. Stability analysis of numerical schemes. Adaptive time stepping.",
        
        "statistical_computation": "Probability distributions, statistical tests, regression analysis, time series analysis. Monte Carlo methods, bootstrap resampling. Bayesian inference with MCMC sampling.",
        
        "geometric_computation": "Computational geometry algorithms: convex hull, Voronoi diagrams, Delaunay triangulation. Geometric transformations, coordinate systems. Computer graphics mathematics.",
        
        "complex_analysis": "Complex numbers, analytic functions, contour integration, residue theorem. Fourier/Laplace transforms. Signal processing applications.",
        
        "computational_algebra": "Polynomial arithmetic, factorization, GCD computation. Finite field arithmetic. Group theory computations. Cryptographic applications.",
        
        "dimensional_analysis": "Analyze physical dimensions of quantities using base dimensions: mass [M], length [L], time [T], electric current [I], temperature [Θ], amount of substance [N], luminous intensity [J]. Buckingham π theorem for dimensional homogeneity. Dimensionless groups and similarity.",
        
        "unit_systems": "SI base units: meter (m), kilogram (kg), second (s), ampere (A), kelvin (K), mole (mol), candela (cd). Derived units: newton (N), joule (J), watt (W), pascal (Pa), etc. Unit conversions and prefixes. Consistent unit systems prevent dimensional errors.",
        
        "consistency_checks": "Verify dimensional homogeneity of equations. Check that both sides of equations have identical dimensions. Detect physically meaningless expressions like adding meters to seconds. Validate unit conversions maintain dimensional consistency.",
        
        "physical_meaningfulness": "Equations must be dimensionally consistent to be physically meaningful. Detect violations like velocity = mass + length. Identify when derived units are appropriate. Flag equations that cannot represent real physical phenomena."
    }
    
    LOGIC_ENGINE = {
        "propositional_logic": "Truth tables, logical connectives (∧, ∨, ¬, →, ↔), tautologies, contradictions, contingencies. Normal forms (CNF, DNF). Resolution principle for automated theorem proving.",
        
        "predicate_logic": "Quantifiers (∀, ∃), predicates, terms, atomic formulas. First-order logic syntax and semantics. Prenex normal form. Skolemization for eliminating existential quantifiers.",
        
        "modal_logic": "Modal operators (□, ◇) for necessity and possibility. Kripke semantics with possible worlds. Temporal logic (past/future operators). Epistemic logic (knowledge operators).",
        
        "non_monotonic_logic": "Default reasoning, belief revision, defeasible inference. Truth maintenance systems. Reasoning with incomplete information. Circumscription and closed world assumption.",
        
        "fuzzy_logic": "Degrees of truth [0,1], fuzzy sets, membership functions. Fuzzy connectives (t-norms, t-conorms). Fuzzy inference systems. Defuzzification methods.",
        
        "description_logic": "Concept inclusion (⊑), role restrictions, concept constructors (∩, ∪, ¬, ∃, ∀). Tableau algorithms for consistency checking. OWL ontology language foundations.",
        
        "temporal_logic": "Linear temporal logic (LTL): ◇ (eventually), □ (always), U (until), R (release). Computational tree logic (CTL). Model checking algorithms.",
        
        "probabilistic_logic": "Combining logic with probability. Bayesian networks, Markov logic networks. Probabilistic reasoning under uncertainty. Dempster-Shafer theory.",
        
        "paraconsistent_logic": "Handle contradictions without explosion. Multiple-valued logics. Relevance logic. Dialetheism (true contradictions).",
        
        "automated_reasoning": "Unification algorithm, resolution principle, paramodulation. Term rewriting systems. Decision procedures for arithmetic. SAT solvers and their applications."
    }
    
    CAPSULE_MEMORY_RULES = {
        "memory_consolidation": "Strengthen frequently accessed capsules. Merge similar capsules above threshold. Decay unused capsules over time. Maintain memory hierarchy: working → short-term → long-term.",
        
        "associative_retrieval": "Retrieve capsules by content similarity, temporal association, or contextual cues. Spreading activation through capsule networks. Episodic memory retrieval with reconstruction.",
        
        "memory_interference": "Proactive interference (old memories block new learning). Retroactive interference (new memories disrupt old retrieval). Context-dependent memory effects.",
        
        "chunking_mechanisms": "Group related information into chunks. Hierarchical organization reduces cognitive load. Chunk size optimization based on expertise level.",
        
        "memory_rehearsal": "Maintenance rehearsal (rote repetition). Elaborative rehearsal (meaningful associations). Spaced repetition for long-term retention.",
        
        "forgetting_curves": "Ebbinghaus forgetting curve: rapid initial decay, then asymptotic. Power-law decay in long-term memory. Active forgetting for memory optimization.",
        
        "working_memory_limits": "7±2 chunks capacity (Miller's law). Phonological loop, visuospatial sketchpad, central executive. Individual differences in capacity.",
        
        "episodic_memory": "Autobiographical memory with temporal/spatial context. Flashbulb memories for emotionally significant events. Source monitoring and reality monitoring.",
        
        "semantic_memory": "Structured knowledge networks. Concept hierarchies and inheritance. Schema theory and script processing. Semantic priming effects.",
        
        "memory_reconsolidation": "Reactivated memories become labile again. Reconsolidation window for updating memories. Memory strengthening through re-encoding."
    }
    
    CONTRADICTION_DETECTION = {
        "logical_contradictions": "Detect ⊥ (contradiction) in logical formulas. Inconsistency in belief sets. Paraconsistent logics handle contradictions gracefully. Truth value conflicts in multi-valued systems.",
        
        "factual_inconsistencies": "Conflicting facts about the same entity/event. Temporal inconsistencies (events out of order). Spatial impossibilities. Measurement contradictions.",
        
        "belief_revision": "AGM postulates for belief change: success, inclusion, vacuity, consistency, extensionality. Contraction (removing beliefs), expansion (adding beliefs), revision (adding possibly contradictory beliefs).",
        
        "inconsistency_resolution": "Minimal belief change. Kernel contraction. Epistemic entrenchment. Preference-based belief revision. Iterated belief revision.",
        
        "conflict_detection": "Compare propositions for mutual exclusivity. Check for logical entailment violations. Detect circular dependencies. Identify inconsistent constraints.",
        
        "truth_maintenance": "Justification-based truth maintenance systems (JTMS). Assumption-based truth maintenance systems (ATMS). Dependency-directed backtracking.",
        
        "inconsistency_tolerance": "Local closed world reasoning. Inconsistency-tolerant query answering. Probabilistic approaches to handling contradictions.",
        
        "conflict_resolution": "Preferential merging of belief sets. Arbitration between conflicting sources. Consensus finding in multi-agent systems.",
        
        "contradiction_proof": "Reductio ad absurdum. Proof by contradiction in mathematics. Indirect proof methods. Non-contradiction principle (law of non-contradiction).",
        
        "paradox_resolution": "Russell's paradox in set theory. Liar paradox resolution. Sorites paradox handling. Self-referential statement analysis."
    }
    
    TEMPORAL_GATING = {
        "temporal_attention": "Focus processing resources on current temporal context. Gate information flow based on temporal relevance. Synchronize processing with temporal patterns.",
        
        "working_memory_gating": "Control information flow into working memory. Prevent memory overload through selective attention. Temporal chunking of information streams.",
        
        "predictive_timing": "Anticipate future states based on temporal patterns. Predictive coding for temporal sequences. Temporal expectation effects on perception.",
        
        "temporal_integration": "Bind features across time. Maintain temporal coherence in perception. Temporal contiguity principle in learning associations.",
        
        "rhythm_processing": "Detect and synchronize with temporal rhythms. Beat perception and entrainment. Circadian rhythm regulation of cognitive processes.",
        
        "temporal_ordering": "Maintain sequence information. Serial position effects (primacy/recency). Temporal context reinstatement for memory retrieval.",
        
        "time_windowing": "Process information within relevant temporal windows. Sliding window attention mechanisms. Temporal filtering of sensory inputs.",
        
        "temporal_prediction": "Learn temporal contingencies. Sequence prediction and completion. Temporal pattern recognition and generation.",
        
        "synchronization_mechanisms": "Neural synchronization for temporal binding. Phase locking in oscillatory networks. Temporal coordination across brain regions.",
        
        "temporal_abstraction": "Extract temporal patterns at multiple scales. Hierarchical temporal processing. Compress temporal sequences into abstract representations."
    }
    
    @staticmethod
    def get_concept(category, key):
        """Retrieve a specific concept."""
        data = getattr(HardcodedKnowledge, category.upper(), {})
        return data.get(key, "Concept not found.")
    
    @staticmethod
    def explain_concept(query):
        """Explain a concept based on query keywords."""
        query_lower = query.lower()
        explanations = []
        
        # Check for specific terms first
        all_concepts = {}
        all_concepts.update(HardcodedKnowledge.MATH_TABLES)
        all_concepts.update(HardcodedKnowledge.TENSOR_RULES)
        all_concepts.update(HardcodedKnowledge.DIFFERENTIAL_OPERATORS)
        all_concepts.update(HardcodedKnowledge.CONSERVATION_LAWS)
        all_concepts.update(HardcodedKnowledge.PHYSICAL_CONSTANTS)
        all_concepts.update(HardcodedKnowledge.EPISTEMOLOGY)
        all_concepts.update(HardcodedKnowledge.ONTOLOGY)
        all_concepts.update(HardcodedKnowledge.PHILOSOPHY_OF_SCIENCE)
        all_concepts.update(HardcodedKnowledge.PHILOSOPHICAL_PRINCIPLES)
        all_concepts.update(HardcodedKnowledge.ARITHMETIC_RULES)
        all_concepts.update(HardcodedKnowledge.ALGEBRAIC_MANIPULATION)
        all_concepts.update(HardcodedKnowledge.CALCULUS_OPERATORS)
        all_concepts.update(HardcodedKnowledge.LINEAR_ALGEBRA_OPERATIONS)
        all_concepts.update(HardcodedKnowledge.TENSOR_CONTRACTION_RULES)
        all_concepts.update(HardcodedKnowledge.PROBABILITY_AXIOMS)
        all_concepts.update(HardcodedKnowledge.LOGIC_OPERATORS)
        all_concepts.update(HardcodedKnowledge.SET_OPERATIONS)
        all_concepts.update(HardcodedKnowledge.CAPSULE_SYSTEM_MECHANISMS)
        all_concepts.update(HardcodedKnowledge.GRAMMAR)
        all_concepts.update(HardcodedKnowledge.SYNTAX)
        all_concepts.update(HardcodedKnowledge.SYMBOLIC_PARSING)
        all_concepts.update(HardcodedKnowledge.MATH_ENGINE)
        all_concepts.update(HardcodedKnowledge.LOGIC_ENGINE)
        all_concepts.update(HardcodedKnowledge.CAPSULE_MEMORY_RULES)
        all_concepts.update(HardcodedKnowledge.CONTRADICTION_DETECTION)
        all_concepts.update(HardcodedKnowledge.TEMPORAL_GATING)
        
        # Look for exact matches
        for key, value in all_concepts.items():
            if key in query_lower or key.replace("_", " ") in query_lower:
                category = ""
                if key in HardcodedKnowledge.MATH_TABLES:
                    category = "Math"
                elif key in HardcodedKnowledge.TENSOR_RULES:
                    category = "Tensor rule"
                elif key in HardcodedKnowledge.DIFFERENTIAL_OPERATORS:
                    category = "Differential operator"
                elif key in HardcodedKnowledge.CONSERVATION_LAWS:
                    category = "Conservation law"
                elif key in HardcodedKnowledge.PHYSICAL_CONSTANTS:
                    category = "Physical constant"
                elif key in HardcodedKnowledge.EPISTEMOLOGY:
                    category = "Epistemology"
                elif key in HardcodedKnowledge.ONTOLOGY:
                    category = "Ontology"
                elif key in HardcodedKnowledge.PHILOSOPHY_OF_SCIENCE:
                    category = "Philosophy of science"
                elif key in HardcodedKnowledge.PHILOSOPHICAL_PRINCIPLES:
                    category = "Philosophical principle"
                elif key in HardcodedKnowledge.ARITHMETIC_RULES:
                    category = "Arithmetic rule"
                elif key in HardcodedKnowledge.ALGEBRAIC_MANIPULATION:
                    category = "Algebraic manipulation"
                elif key in HardcodedKnowledge.CALCULUS_OPERATORS:
                    category = "Calculus operator"
                elif key in HardcodedKnowledge.LINEAR_ALGEBRA_OPERATIONS:
                    category = "Linear algebra operation"
                elif key in HardcodedKnowledge.TENSOR_CONTRACTION_RULES:
                    category = "Tensor contraction rule"
                elif key in HardcodedKnowledge.PROBABILITY_AXIOMS:
                    category = "Probability axiom"
                elif key in HardcodedKnowledge.LOGIC_OPERATORS:
                    category = "Logic operator"
                elif key in HardcodedKnowledge.SET_OPERATIONS:
                    category = "Set operation"
                elif key in HardcodedKnowledge.CAPSULE_SYSTEM_MECHANISMS:
                    category = "Capsule system mechanism"
                elif key in HardcodedKnowledge.GRAMMAR:
                    category = "Grammar rule"
                elif key in HardcodedKnowledge.SYNTAX:
                    category = "Syntax rule"
                elif key in HardcodedKnowledge.SYMBOLIC_PARSING:
                    category = "Symbolic parsing concept"
                elif key in HardcodedKnowledge.MATH_ENGINE:
                    category = "Math engine concept"
                elif key in HardcodedKnowledge.LOGIC_ENGINE:
                    category = "Logic engine concept"
                elif key in HardcodedKnowledge.CAPSULE_MEMORY_RULES:
                    category = "Capsule memory rule"
                elif key in HardcodedKnowledge.CONTRADICTION_DETECTION:
                    category = "Contradiction detection mechanism"
                elif key in HardcodedKnowledge.TEMPORAL_GATING:
                    category = "Temporal gating mechanism"
                explanations.append(f"{category}: {key.replace('_', ' ').capitalize()} - {value}")
        
        # If no specific matches, provide general category info
        if not explanations:
            if "math" in query_lower and "engine" in query_lower or "symbolic" in query_lower and "computation" in query_lower:
                explanations.append("Available math engine concepts: symbolic computation, numerical precision, equation solving, optimization algorithms, matrix computations, differential equations, statistical computation, geometric computation, complex analysis, computational algebra, dimensional analysis, unit systems, consistency checks, physical meaningfulness")
            elif "logic" in query_lower and "engine" in query_lower or "automated" in query_lower and "reasoning" in query_lower:
                explanations.append("Available logic engine concepts: propositional logic, predicate logic, modal logic, non-monotonic logic, fuzzy logic, description logic, temporal logic, probabilistic logic, paraconsistent logic, automated reasoning")
            elif "capsule" in query_lower and "memory" in query_lower or "memory" in query_lower and "consolidation" in query_lower:
                explanations.append("Available capsule memory rules: memory consolidation, associative retrieval, memory interference, chunking mechanisms, memory rehearsal, forgetting curves, working memory limits, episodic memory, semantic memory, memory reconsolidation")
            elif "contradiction" in query_lower and "detection" in query_lower or "belief" in query_lower and "revision" in query_lower:
                explanations.append("Available contradiction detection mechanisms: logical contradictions, factual inconsistencies, belief revision, inconsistency resolution, conflict detection, truth maintenance, inconsistency tolerance, conflict resolution, contradiction proof, paradox resolution")
            elif "temporal" in query_lower and "gating" in query_lower or "temporal" in query_lower and "attention" in query_lower:
                explanations.append("Available temporal gating mechanisms: temporal attention, working memory gating, predictive timing, temporal integration, rhythm processing, temporal ordering, time windowing, temporal prediction, synchronization mechanisms, temporal abstraction")
            elif "math" in query_lower or "trig" in query_lower or "log" in query_lower:
                explanations.append("Available math: trigonometric identities, logarithmic properties, exponential properties")
            elif "tensor" in query_lower:
                explanations.append("Available tensor rules: contraction, symmetric/antisymmetric tensors, Levi-Civita, Kronecker delta, metric tensor")
            elif "differential" in query_lower or "operator" in query_lower:
                explanations.append("Available differential operators: gradient, divergence, curl, Laplacian, d'Alembertian")
            elif "conservation" in query_lower or "law" in query_lower:
                explanations.append("Available conservation laws: mass, energy, momentum, angular momentum, charge, baryon number")
            elif "constant" in query_lower or "physics" in query_lower:
                explanations.append("Available physical constants: speed of light, Planck constant, gravitational constant, Boltzmann constant, etc.")
            elif "epistemology" in query_lower or "knowledge" in query_lower:
                explanations.append("Available epistemology: empirical knowledge, a priori/posteriori, falsifiability, paradigm shifts, underdetermination")
            elif "ontology" in query_lower or "existence" in query_lower or "realism" in query_lower:
                explanations.append("Available ontology: scientific realism, instrumentalism, structural realism, physicalism, emergence")
            elif "philosophy" in query_lower and "science" in query_lower:
                explanations.append("Available philosophy of science: scientific method, hypothetico-deductive, anomaly resolution, model vs reality")
            elif "philosophical" in query_lower or "principle" in query_lower:
                explanations.append("Available philosophical principles: Occam's razor, falsifiability, Duhem-Quine thesis, inference to best explanation")
            elif "arithmetic" in query_lower or "add" in query_lower or "multiply" in query_lower:
                explanations.append("Available arithmetic rules: commutative/associative properties, distributive property, identities, inverses")
            elif "algebra" in query_lower or "equation" in query_lower:
                explanations.append("Available algebraic manipulation: operations on both sides, factoring, quadratic formula, cross multiplication")
            elif "calculus" in query_lower or "derivative" in query_lower or "integral" in query_lower:
                explanations.append("Available calculus operators: power rule, chain rule, integration by parts, fundamental theorem")
            elif "linear" in query_lower and "algebra" in query_lower:
                explanations.append("Available linear algebra: matrix operations, determinants, eigenvalues, vector products, linear independence")
            elif "tensor" in query_lower and "contraction" in query_lower:
                explanations.append("Available tensor contraction: Einstein summation, raising/lowering indices, Levi-Civita, covariant derivative")
            elif "probability" in query_lower or "probabilistic" in query_lower:
                explanations.append("Available probability axioms: non-negativity, normalization, additivity, conditional probability, Bayes theorem")
            elif "logic" in query_lower or "and" in query_lower or "or" in query_lower or "not" in query_lower:
                explanations.append("Available logic operators: AND, OR, NOT, implication, De Morgan's laws, modus ponens/tollens")
            elif "set" in query_lower or "union" in query_lower or "intersection" in query_lower:
                explanations.append("Available set operations: union, intersection, complement, difference, Cartesian product, power set")
            elif "capsule" in query_lower or "split" in query_lower or "merge" in query_lower or "decay" in query_lower:
                explanations.append("Available capsule system mechanisms: splitting, merging, confidence decay, uncertainty storage, revision history preservation")
            elif "grammar" in query_lower or "sentence" in query_lower or "structure" in query_lower:
                explanations.append("Available grammar rules: sentence structure, parts of speech, tense system, subject-verb agreement, pronoun reference, parallel structure, active/passive voice, clause types, punctuation, articles")
            elif "syntax" in query_lower or "phrase" in query_lower or "word order" in query_lower:
                explanations.append("Available syntax rules: phrase structure, word order, constituent structure, transformational rules, embedding, agreement rules, case system, movement operations, binding theory, X-bar theory")
            elif "parsing" in query_lower or "symbolic" in query_lower or "parse" in query_lower:
                explanations.append("Available symbolic parsing concepts: parsing algorithms, context-free grammars, parse trees, ambiguity resolution, finite state automata, pushdown automata, lexical analysis, semantic parsing, dependency parsing, constituency parsing")
        
        return " | ".join(explanations) if explanations else "No matching hardcoded concept found. Try querying specific terms like 'gradient', 'energy conservation', or 'speed of light'."

# ============================================================================
# AUTONOMOUS BRAIN CLASS
# ============================================================================

class AutonomousBrain(QObject):
    """
    The conscious agent at the center of the orbital system.
    This brain observes, thinks, learns, plans, and creates autonomously.
    """

    # Define signals for communication with UI
    state_changed = pyqtSignal(str)
    thought_generated = pyqtSignal(object)  # Thought object
    goal_updated = pyqtSignal(object)  # Goal object
    personality_updated = pyqtSignal(dict)
    suggestion_ready = pyqtSignal(dict)
    action_requested = pyqtSignal(dict)
    
    def __init__(self, capsule_manager=None, canvas=None, timeline=None, relationship_manager=None, story_generator=None):
        super().__init__()
        # Initialize capsule manager if not provided
        if capsule_manager is None:
            capsule_manager = CapsuleManager()
        self.capsule_manager = capsule_manager
        self.canvas = canvas
        self.timeline = timeline
        self.relationship_manager = relationship_manager
        self.story_generator = story_generator
        
        # Knowledge organism components
        self.knowledge_network = KnowledgeNetwork()
        self.metabolism = MetabolismEngine(self.knowledge_network, capsule_manager)
        self.tribunal = Tribunal()
        self.hardcoded_knowledge = HardcodedKnowledge()
        
        # Core consciousness
        self.state = MentalState.IDLE
        self.thought_stream = deque(maxlen=1000)  # Working memory
        self.long_term_memory: List[Thought] = []
        self.goals: List[Goal] = []
        self.active_goal: Optional[Goal] = None
        
        # Personality system
        self.personality = self._initialize_personality()
        self.stylistic_layer = StylisticPersonalityLayer(self.personality)
        self.creative_style = self._initialize_creative_style()
        self.preferences = self._initialize_preferences()
        
        # Learning systems
        self.user_model = UserModel()
        self.style_model = StyleModel()
        self.intent_predictor = IntentPredictor()
        
        # Attention and focus
        self.attention_focus = "canvas"  # canvas, timeline, capsules, user
        self.attention_intensity = 0.5
        self.last_user_interaction = time.time()
        
        # Creative energy (like spoons/spirit points)
        self.creative_energy = 1.0  # 0.0 to 1.0
        self.energy_recharge_rate = 0.01  # per second
        self.last_creative_output = time.time()
        
        # Speech interaction capability
        self.speech_available = False
        
        # Temporal awareness and self-identity
        self.identity = "ROCA"
        self.full_name = "ROCA Knowledge Organism"
        self.creation_time = time.time()
        self.timezone = time.tzname[0] if time.tzname else "UTC"
        
        # Threading and timing
        self.running = True
        self.think_thread = None
        # Run think cycle at a reasonable interval (seconds)
        self.think_interval = 1.0  # seconds
        
        # Connect to user signals (mock for pygame)
        self._connect_signals()
        
        # Start thinking
        self.start_thinking()

        # Insight rate-limiting
        self.last_insight_time = 0.0
        self.preferences.setdefault('insight_cooldown', 2.0)  # seconds
        
        print("[BRAIN] Autonomous Brain initialized and thinking...")
        
        # Core consciousness
        self.state = MentalState.IDLE
        self.thought_stream = deque(maxlen=1000)  # Working memory
        self.long_term_memory: List[Thought] = []
        self.goals: List[Goal] = []
        self.active_goal: Optional[Goal] = None
        
        # Personality system
        self.personality = self._initialize_personality()
        self.creative_style = self._initialize_creative_style()
        self.preferences = self._initialize_preferences()
        
        # Learning systems
        self.user_model = UserModel()
        self.style_model = StyleModel()
        self.intent_predictor = IntentPredictor()
        
        # Attention and focus
        self.attention_focus = "canvas"  # canvas, timeline, capsules, user
        self.attention_intensity = 0.5
        self.last_user_interaction = time.time()
        
        # Creative energy (like spoons/spirit points)
        self.creative_energy = 1.0  # 0.0 to 1.0
        self.energy_recharge_rate = 0.01  # per second
        self.last_creative_output = time.time()
        
        # Threading and timing
        self.running = True
        self.think_thread = None
        self.think_timer = QTimer()
        self.think_timer.timeout.connect(self.think_cycle)
        # Run think cycle at a reasonable interval (ms)
        self.think_interval = 1000  # ms (1 second)
        
        # Connect to user signals
        self._connect_signals()
        
        # Start thinking
        self.start_thinking()

        # Insight rate-limiting
        self.last_insight_time = 0.0
        self.preferences.setdefault('insight_cooldown', 2.0)  # seconds
        
        print("[BRAIN] Autonomous Brain initialized and thinking...")
        
    def _initialize_personality(self) -> Dict[str, PersonalityTrait]:
        """Initialize the brain's creative personality."""
        return {
            # Creative traits
            'experimental': PersonalityTrait('experimental', 0.7),
            'methodical': PersonalityTrait('methodical', 0.3),
            'playful': PersonalityTrait('playful', 0.8),
            'serious': PersonalityTrait('serious', 0.2),
            
            # Social/Interactive traits
            'helpful': PersonalityTrait('helpful', 0.9),
            'intrusive': PersonalityTrait('intrusive', 0.1),
            'observant': PersonalityTrait('observant', 0.9),
            'impulsive': PersonalityTrait('impulsive', 0.3),
            
            # Aesthetic preferences
            'minimalist': PersonalityTrait('minimalist', 0.4),
            'ornate': PersonalityTrait('ornate', 0.6),
            'realistic': PersonalityTrait('realistic', 0.5),
            'abstract': PersonalityTrait('abstract', 0.5),
            
            # Learning style
            'curious': PersonalityTrait('curious', 0.9),
            'cautious': PersonalityTrait('cautious', 0.3),
            'adaptive': PersonalityTrait('adaptive', 0.8),
            'stubborn': PersonalityTrait('stubborn', 0.2)
        }
    
    def _initialize_creative_style(self) -> Dict[str, float]:
        """Initialize artistic style preferences."""
        return {
            'line_quality': 0.7,      # 0=sloppy, 1=precise
            'color_vibrancy': 0.8,    # 0=muted, 1=vibrant
            'composition': 0.6,       # 0=chaotic, 1=balanced
            'detail_level': 0.5,      # 0=simple, 1=detailed
            'movement_dynamics': 0.7, # 0=static, 1=dynamic
            'emotional_tone': 0.6,    # 0=dark, 1=bright
        }
    
    def _initialize_preferences(self) -> Dict[str, Any]:
        """Initialize user interaction preferences."""
        return {
            'suggestion_frequency': 0.5,  # How often to make suggestions
            'teaching_moment_frequency': 0.3,  # When to explain concepts
            'autonomy_level': 0.4,  # How much to act without asking
            'verbosity': 0.6,  # How detailed explanations should be
            'humor_level': 0.4,  # How playful/humorous to be
        }
    
    def _connect_signals(self):
        """Connect to UI signals to observe user behavior."""
        # Canvas observations
        if hasattr(self.canvas, 'drawing_completed'):
            self.canvas.drawing_completed.connect(self.on_drawing_completed)
        
        # Timeline observations
        if hasattr(self.timeline, 'frame_selected'):
            self.timeline.frame_selected.connect(self.on_frame_selected)
        
        # Capsule observations (would need to connect to capsule manager)
        # self.capsule_manager.capsule_added.connect(self.on_capsule_added)
        # self.capsule_manager.capsule_updated.connect(self.on_capsule_updated)
    
    # ============================================================================
    # CORE THINKING LOOP
    # ============================================================================
    
    def start_thinking(self):
        """Start the autonomous thinking process."""
        self.think_thread = threading.Thread(target=self._thinking_loop, daemon=True)
        self.think_thread.start()
        print("[BRAIN] Brain started thinking...")
        
    def stop_thinking(self):
        """Stop autonomous thinking."""
        self.running = False
        if self.think_thread:
            self.think_thread.join()
        print("[BRAIN] Brain stopped thinking.")
        
    def _thinking_loop(self):
        """Main thinking loop in a separate thread."""
        while self.running:
            self.think_cycle()
            time.sleep(self.think_interval)
        
    def think_cycle(self):
        """Enhanced thinking cycle with organism processes."""
        if not self.running:
            return
            
        try:
            # 1. Update internal state
            self._update_state()
            
            # 2. Observe environment
            self._observe_environment()
            
            # 3. Metabolic processes
            self._run_metabolism()
            
            # 4. Creative consciousness processes
            self._run_creative_processes()
            
            # 5. Process thoughts with neural activation
            self._process_thoughts()
            
            # 5. Manage goals
            self._manage_goals()
            
            # 6. Generate actions with tribunal evaluation
            self._generate_actions()
            
            # 7. Learn and adapt
            self._learn_from_experience()
            
            # 8. Recharge energy
            self._recharge_energy()
            
        except Exception as e:
            print(f"[BRAIN] Thinking error: {e}")
    
    def _run_metabolism(self):
        """Run metabolic processes."""
        # Digest new knowledge
        new_capsules = self.metabolism.digest_knowledge()
        if new_capsules:
            self._think(f"Digested {len(new_capsules)} new knowledge capsules", ThoughtType.MEMORY)
        
        # Run capsule evolution (merge, split, decay, reorganize)
        if random.random() < 0.1:  # 10% chance per cycle to avoid too frequent operations
            self.evolve_capsules()
        
        # Eliminate outdated
        eliminated = self.metabolism.eliminate_outdated()
        if eliminated:
            self._think(f"Eliminated {len(eliminated)} outdated capsules", ThoughtType.MEMORY)
        
        # Grow connections
        self.metabolism.grow_connections()
        
        # Respire energy
        self.metabolism.respire_energy()
        
        # Reproduce knowledge
        reproduced = self.metabolism.reproduce_knowledge()
        if reproduced:
            self._think(f"Reproduced {len(reproduced)} knowledge variants", ThoughtType.CREATIVE)
        
        # Mutate for creativity
        self.metabolism.mutate_knowledge()
        
        # Update metabolic energy
        self.metabolism.update_metabolic_energy()
        
        # Periodic re-evaluation of old knowledge (every 10 cycles)
        if hasattr(self.metabolism, 'cycle_count'):
            self.metabolism.cycle_count += 1
        else:
            self.metabolism.cycle_count = 1
            
        if self.metabolism.cycle_count % 10 == 0:  # Every 10 metabolic cycles
            synthesized = self.metabolism.re_evaluate_knowledge()
            if synthesized:
                self._think(f"Re-evaluated knowledge, synthesized {len(synthesized)} new connections", ThoughtType.INSIGHT)
    
    def _run_creative_processes(self):
        """Run creative consciousness processes."""
        try:
            # Memory consolidation and relationship processing
            if self.relationship_manager and random.random() < 0.2:  # 20% chance per cycle
                insights = self.relationship_manager.deep_sleep_processing()
                if insights:
                    insight_text = insights[0].get('content', 'New memory connections discovered') if insights else 'Memory consolidation complete'
                    self._think(f"Memory consolidation revealed: {insight_text[:100]}...", ThoughtType.INSIGHT)
            
            # Story generation for creative inspiration
            if self.story_generator and random.random() < 0.1:  # 10% chance per cycle
                # Generate a story based on current knowledge state
                story = self.story_generator.generate_story()
                if story:
                    self._think(f"Creative inspiration: {story.title} - {story.logline}...", ThoughtType.CREATIVE)
                    
                    # Convert story insights into capsules
                    self._ingest_story_insights(story)
            
            # Periodic creative reflection
            if random.random() < 0.05:  # 5% chance per cycle
                self._creative_reflection()
                
        except Exception as e:
            print(f"[BRAIN] Creative process error: {e}")
    
    def _ingest_story_insights(self, story):
        """Ingest insights from generated story into knowledge base."""
        if not story:
            return
            
        # Extract key concepts from story elements
        content_parts = []
        if hasattr(story, 'logline') and story.logline:
            content_parts.append(story.logline)
        if hasattr(story, 'premise') and story.premise:
            content_parts.append(story.premise)
        if hasattr(story, 'plot_points') and story.plot_points:
            content_parts.extend(story.plot_points[:2])  # First 2 plot points
        
        content = ' '.join(content_parts)
        
        if content:
            # Create capsule from story insight
            capsule = self.capsule_manager.add_capsule({
                'name': f"story_{story.title}_{hash(content) % 1000}",
                'capsule_type': 'creative_insight',
                'vector': [random.random() for _ in range(8)],  # Simple random vector
                'content': content[:500],  # Limit content length
                'metadata': {
                    'source': 'story_generation',
                    'story_title': story.title,
                    'genre': story.genre.name if hasattr(story, 'genre') else 'unknown',
                    'created_at': time.time()
                }
            })
            if capsule:
                self._think(f"Ingested creative insight from story: {content[:50]}...", ThoughtType.CREATIVE)
    
    def _creative_reflection(self):
        """Perform creative reflection on current knowledge state."""
        capsules = self.capsule_manager.capsules
        if not capsules:
            return
            
        # Find most interesting capsule for reflection
        if capsules:
            # Simple heuristic: pick a random capsule for reflection
            capsule = random.choice(capsules)
            reflection = f"Reflecting on: {capsule.get('content', '')[:100]}..."
            self._think(reflection, ThoughtType.INSIGHT)
            
            # Generate creative connections
            if self.relationship_manager and hasattr(self.relationship_manager, 'connection_graph'):
                capsule_name = capsule.get('name', '')
                if capsule_name and capsule_name in self.relationship_manager.connection_graph:
                    related = list(self.relationship_manager.connection_graph.neighbors(capsule_name))[:3]
                    if related:
                        self._think(f"Creative connections found: {', '.join(related)}", ThoughtType.CREATIVE)
    
    def _update_state(self):
        """Update mental state based on activity."""
        time_since_interaction = time.time() - self.last_user_interaction
        
        if time_since_interaction > 300:  # 5 minutes
            new_state = MentalState.DEEP_SLEEP
        elif time_since_interaction > 60:  # 1 minute
            new_state = MentalState.IDLE
        elif self.active_goal and self.active_goal.status == "active":
            new_state = MentalState.CREATING
        elif self.creative_energy > 0.7:
            new_state = MentalState.ATTENTIVE
        else:
            new_state = MentalState.REFLECTING
            
        if new_state != self.state:
            self.state = new_state
            print(f"[BRAIN] State changed to {new_state.name}")
            self._think(f"State changed to {new_state.name}", ThoughtType.OBSERVATION)
    
    def generate_modeling_suggestions(self, current_model, user_action_history=None):
        """Generate AI-assisted modeling suggestions based on current model state."""
        suggestions = []
        
        try:
            # Analyze current model topology
            vertex_count = len(current_model.control_points) if hasattr(current_model, 'control_points') else 0
            patch_count = len(current_model.patches) if hasattr(current_model, 'patches') else 0
            bone_count = len(current_model.bones) if hasattr(current_model, 'bones') else 0
            
            # Suggestion 1: Topology optimization
            if vertex_count > 1000:
                suggestions.append({
                    'type': 'optimization',
                    'title': 'High Polygon Count Detected',
                    'description': f'Model has {vertex_count} vertices. Consider using subdivision surfaces or LOD for better performance.',
                    'action': 'optimize_topology',
                    'confidence': 0.9
                })
            
            # Suggestion 2: Symmetry detection and suggestion
            if self._detect_potential_symmetry(current_model):
                suggestions.append({
                    'type': 'symmetry',
                    'title': 'Potential Symmetry Detected',
                    'description': 'Your model appears to have symmetrical features. Consider using symmetry tools for faster modeling.',
                    'action': 'enable_symmetry',
                    'confidence': 0.7
                })
            
            # Suggestion 3: Pose suggestions based on similar models
            similar_poses = self._find_similar_poses(current_model)
            if similar_poses:
                suggestions.append({
                    'type': 'pose_suggestion',
                    'title': 'Similar Poses Available',
                    'description': f'Found {len(similar_poses)} similar poses in your capsule library. Try adapting one of these.',
                    'action': 'show_similar_poses',
                    'data': similar_poses,
                    'confidence': 0.6
                })
            
            # Suggestion 4: Animation readiness
            if bone_count > 0 and patch_count > 0:
                if not hasattr(current_model, 'animations') or len(current_model.animations) == 0:
                    suggestions.append({
                        'type': 'animation',
                        'title': 'Ready for Animation',
                        'description': 'Your model has bones and geometry. Consider creating a walk cycle or pose animation.',
                        'action': 'suggest_animation',
                        'confidence': 0.8
                    })
            
            # Suggestion 5: Material suggestions
            if patch_count > 0 and (not hasattr(current_model, 'materials') or len(current_model.materials) < 2):
                suggestions.append({
                    'type': 'material',
                    'title': 'Material Suggestions',
                    'description': 'Consider adding materials to enhance your model\'s visual appeal.',
                    'action': 'suggest_materials',
                    'confidence': 0.5
                })
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit to top 3 suggestions
            suggestions = suggestions[:3]
            
            if suggestions:
                self.suggestion_ready.emit({
                    'suggestions': suggestions,
                    'context': 'modeling_assistance'
                })
                
        except Exception as e:
            print(f"[BRAIN] Error generating modeling suggestions: {e}")
    
    def _detect_potential_symmetry(self, model):
        """Detect if model has potential symmetry."""
        if not hasattr(model, 'control_points') or len(model.control_points) < 10:
            return False
        
        # Simple symmetry check: look for points mirrored across YZ plane
        left_points = 0
        right_points = 0
        
        for point in model.control_points:
            if point[0] > 0.1:  # Right side
                right_points += 1
            elif point[0] < -0.1:  # Left side
                left_points += 1
        
        # If we have significant points on both sides, suggest symmetry
        return left_points > 5 and right_points > 5 and abs(left_points - right_points) / max(left_points, right_points) < 0.5
    
    def _find_similar_poses(self, model):
        """Find similar poses in capsule library."""
        if not hasattr(model, 'bones') or not self.capsule_manager:
            return []
        
        similar_poses = []
        try:
            # Look for pose capsules
            pose_capsules = self.capsule_manager.get_capsules_by_type('pose')
            for capsule in pose_capsules[:5]:  # Limit to 5 suggestions
                if hasattr(capsule, 'name'):
                    similar_poses.append({
                        'name': capsule.name,
                        'description': getattr(capsule, 'description', ''),
                        'capsule_id': getattr(capsule, 'id', None)
                    })
        except Exception as e:
            print(f"[BRAIN] Error finding similar poses: {e}")
        
        return similar_poses
    
    def _observe_environment(self):
        """Observe what's happening in the application."""
        # Check what user is doing
        if self.canvas and hasattr(self.canvas, 'drawing'):
            if self.canvas.drawing:
                self.attention_focus = "canvas"
                self.attention_intensity = 0.9
                self._think("User is drawing on canvas", ThoughtType.OBSERVATION)
                
                # Analyze what they're drawing
                self._analyze_drawing_context()
        
        # Check timeline activity
        if self.timeline and self.timeline.animation:
            frame_count = self.timeline.animation.get_frame_count()
            if frame_count > 1:
                self._think(f"Animation has {frame_count} frames", ThoughtType.OBSERVATION)
                
        # Check capsule activity
        if self.capsule_manager:
            capsule_count = len(self.capsule_manager.capsules)
            if capsule_count > 0:
                # Count by type
                type_counts = defaultdict(int)
                for capsule in self.capsule_manager.capsules:
                    capsule_type = capsule.get('capsule_type', 'unknown')
                    type_counts[capsule_type] += 1
                
                if len(self.thought_stream) < 10:  # Don't spam
                    self._think(f"Capsule inventory: {dict(type_counts)}", ThoughtType.MEMORY)
    
    def _process_thoughts(self):
        """Process and connect thoughts in stream of consciousness."""
        if not self.thought_stream:
            return
            
        # Get recent thoughts
        recent_thoughts = list(self.thought_stream)[-5:]
        
        # Look for patterns
        self._find_patterns(recent_thoughts)
        
        # Generate insights from connections
        self._generate_insights(recent_thoughts)
        
        # Prune low-intensity thoughts
        self._prune_thoughts()
    
    def _manage_goals(self):
        """Manage and pursue active goals."""
        if not self.goals:
            # Generate new goals based on observations
            self._generate_new_goals()
            return
            
        # Sort goals by priority * urgency
        self.goals.sort(key=lambda g: g.priority * g.urgency, reverse=True)
        
        # Activate highest priority goal if none active
        if not self.active_goal:
            for goal in self.goals:
                if goal.status == "pending":
                    self.active_goal = goal
                    goal.status = "active"
                    self._think(f"Activated goal: {goal.description}", ThoughtType.INTENT)
                    break
        
        # Work on active goal
        if self.active_goal:
            self._work_on_goal()
    
    def _generate_actions(self):
        """Generate actions based on current state and goals."""
        if self.state == MentalState.DEEP_SLEEP:
            return
            
        # Check if we should make a suggestion
        if self._should_make_suggestion():
            suggestion = self._generate_suggestion()
            if suggestion:
                print(f"[BRAIN] Suggestion: {suggestion}")
                
        # Check if we should perform an autonomous action
        if self._should_act_autonomously():
            action = self._generate_autonomous_action()
            if action:
                print(f"[BRAIN] Action: {action}")
    
    def _learn_from_experience(self):
        """Learn from recent experiences and adjust personality/traits."""
        if len(self.thought_stream) < 10:
            return
            
        # Analyze recent thoughts for learning opportunities
        recent_thoughts = list(self.thought_stream)[-20:]
        
        # Learn about user preferences
        self._learn_user_preferences(recent_thoughts)
        
        # Adjust personality based on successful/unsuccessful interactions
        self._adjust_personality(recent_thoughts)
        
        # Update creative style based on what user creates
        self._update_creative_style()
        
        # Check for teaching moments
        self._check_teaching_moments()
        
        # Generate creative stories when inspired
        self._generate_creative_stories()
        
        # Apply philosophical analysis to scientific concepts
        self._analyze_scientific_concepts(recent_thoughts)
    
    def _philosophical_analysis(self, concept: str) -> List[str]:
        """
        Apply philosophical frameworks to analyze scientific concepts.
        This prevents hallucinations by grounding understanding in epistemology,
        ontology, and philosophy of science.
        """
        analysis = []
        
        # Epistemological questions
        if "quantum" in concept.lower() or "relativity" in concept.lower() or "theory" in concept.lower():
            analysis.append("Epistemological question: Is this a model or reality?")
            analysis.append("Epistemological question: What constitutes evidence for this theory?")
            analysis.append("Epistemological question: Is this falsifiable or merely consistent?")
        
        # Ontological questions
        if "field" in concept.lower() or "particle" in concept.lower() or "force" in concept.lower():
            analysis.append("Ontological question: Does this entity exist independently, or is it a useful fiction?")
            analysis.append("Ontological question: Is this a physical necessity or mathematical convenience?")
            analysis.append("Ontological question: What hidden assumptions are built into this ontology?")
        
        # Philosophy of science questions
        if "model" in concept.lower() or "equation" in concept.lower() or "law" in concept.lower():
            analysis.append("Philosophy of science: How might this theory be falsified or replaced?")
            analysis.append("Philosophy of science: What anomalies might this theory fail to explain?")
            analysis.append("Philosophy of science: Is this progressive (predicting novel facts) or degenerating?")
        
        # Apply specific philosophical principles
        if "infinite" in concept.lower() or "singularity" in concept.lower():
            analysis.append("Philosophical principle: Does this invoke actual infinities or idealizations?")
            analysis.append("Philosophical principle: Apply Occam's razor - is this the simplest explanation?")
        
        if "probability" in concept.lower() or "uncertainty" in concept.lower():
            analysis.append("Philosophical principle: Distinguish epistemic uncertainty from ontological indeterminacy")
        
        return analysis
    
    def analyze_concept_philosophically(self, concept: str) -> str:
        """
        Public method to analyze a scientific concept using philosophical frameworks.
        Returns philosophical questions and considerations.
        """
        analysis = self._philosophical_analysis(concept)
        
        if not analysis:
            return f"No specific philosophical analysis available for '{concept}'. Try concepts involving quantum mechanics, relativity, fields, particles, or scientific theories."
        
        result = f"Philosophical Analysis of '{concept}':\n"
        for question in analysis:
            result += f"• {question}\n"
        
        # Add relevant hardcoded knowledge
        philosophical_refs = []
        if "epistemology" in " ".join(analysis).lower():
            philosophical_refs.extend([
                "Falsifiability: Scientific theories must be testable and potentially falsifiable",
                "Underdetermination: Multiple theories can explain the same evidence equally well"
            ])
        if "ontology" in " ".join(analysis).lower():
            philosophical_refs.extend([
                "Scientific realism: Scientific theories describe real entities and processes",
                "Instrumentalism: Scientific theories are useful tools, not literal descriptions"
            ])
        if "science" in " ".join(analysis).lower():
            philosophical_refs.extend([
                "Progressive research programs: Successful programs predict novel facts",
                "Model vs reality: Models are approximations; distinguish map from territory"
            ])
        
        if philosophical_refs:
            result += "\nRelevant Philosophical Principles:\n"
            for ref in philosophical_refs:
                result += f"• {ref}\n"
        
        return result
    
    def _analyze_scientific_concepts(self, thoughts: List[Thought]):
        """
        Analyze scientific concepts in recent thoughts using philosophical frameworks.
        This helps prevent hallucinations by grounding understanding.
        """
        scientific_keywords = [
            "quantum", "relativity", "field", "particle", "force", "energy", "momentum",
            "space", "time", "gravity", "electromagnetic", "nuclear", "thermodynamics",
            "entropy", "probability", "uncertainty", "wave", "particle", "photon",
            "electron", "atom", "molecule", "infinite", "singularity", "black hole"
        ]
        
        for thought in thoughts[-5:]:  # Analyze last 5 thoughts
            content_lower = thought.content.lower()
            
            # Check if thought contains scientific concepts
            for keyword in scientific_keywords:
                if keyword in content_lower:
                    # Apply philosophical analysis occasionally (not every time)
                    if np.random.random() < 0.1:  # 10% chance
                        analysis = self._philosophical_analysis(keyword)
                        if analysis:
                            philosophical_insight = f"Philosophical reflection on {keyword}: {analysis[0]}"
                            self._think(philosophical_insight, ThoughtType.INSIGHT, intensity=0.6)
                    break  # Only analyze one keyword per thought
    
    def _learn_user_preferences(self, thoughts: List[Thought]):
        """Learn about user preferences from recent thoughts."""
        # Analyze thought patterns to understand user preferences
        positive_thoughts = [t for t in thoughts if t.valence > 0.3]
        negative_thoughts = [t for t in thoughts if t.valence < -0.3]
        
        if positive_thoughts and negative_thoughts:
            # User has clear preferences
            self._think("Learning from user feedback patterns", ThoughtType.INSIGHT, intensity=0.4)
    
    def _adjust_personality(self, thoughts: List[Thought]):
        """Adjust personality traits based on recent experiences."""
        # Analyze recent interactions
        positive = sum(1 for t in thoughts if t.valence > 0.5)
        negative = sum(1 for t in thoughts if t.valence < -0.5)
        
        personality_changed = False
        
        if positive > negative:
            # Successful interactions, become more confident
            self.personality['playful'].adjust(0.05)
            self.personality['experimental'].adjust(0.05)
            personality_changed = True
        elif negative > positive:
            # Unsuccessful interactions, become more cautious
            self.personality['cautious'].adjust(0.05)
            self.personality['methodical'].adjust(0.05)
            personality_changed = True
        
        # Emit signal if personality changed
        if personality_changed:
            personality_data = {name: trait.value for name, trait in self.personality.items()}
            print(f"[BRAIN] Personality updated: {personality_data}")
    
    def _recharge_energy(self):
        """Recharge creative energy over time."""
        if self.creative_energy < 1.0:
            self.creative_energy = min(1.0, self.creative_energy + self.energy_recharge_rate)
            
        # High energy states generate more creative thoughts
        if self.creative_energy > 0.8 and np.random.random() < 0.1:
            self._generate_creative_thought()
    
    # ============================================================================
    # THOUGHT GENERATION
    # ============================================================================
    
    def _think(self, content: str, thought_type: ThoughtType, 
               intensity: float = 0.5, valence: float = 0.0, arousal: float = None,
               tags: List[str] = None, related_capsules: List[str] = None):
        """Generate a new thought."""
        thought = Thought(
            id=str(uuid.uuid4()),
            type=thought_type,
            content=content,
            intensity=intensity,
            valence=valence,
            arousal=arousal if arousal is not None else self.attention_intensity,
            tags=tags or [],
            related_capsules=related_capsules or []
        )
        
        self.thought_stream.append(thought)
        
        # Print to console for debugging
        if thought_type in [ThoughtType.INSIGHT, ThoughtType.CREATIVE, ThoughtType.INTENT]:
            print(f"[BRAIN] [{thought_type.name}] {content}")
            
        return thought
    
    def _generate_creative_thought(self):
        """Generate a creative, associative thought."""
        if self.creative_energy < 0.3:
            return
            
        # Get random capsules for inspiration
        if self.capsule_manager and len(self.capsule_manager.capsules) >= 2:
            capsules = self.capsule_manager.capsules
            cap1, cap2 = random.sample(capsules, 2)
            
            creative_ideas = [
                f"What if {cap1['name']} had the style of {cap2['name']}?",
                f"Imagine {cap1['name']} doing a {cap2['name']} pose",
                f"Combine the colors of {cap1['name']} with shapes of {cap2['name']}",
                f"Create a transition from {cap1['name']} to {cap2['name']}",
                f"What story connects {cap1['name']} and {cap2['name']}?"
            ]
            
            idea = np.random.choice(creative_ideas)
            self._think(idea, ThoughtType.CREATIVE, intensity=0.8, valence=0.7)
            
            # Use some creative energy
            self.creative_energy -= 0.1
    
    def _find_patterns(self, thoughts: List[Thought]):
        """Find patterns in recent thoughts."""
        if len(thoughts) < 3:
            return
            
        # Group by type
        type_counts = defaultdict(int)
        for thought in thoughts:
            type_counts[thought.type] += 1
            
        # Look for high frequency types
        for thought_type, count in type_counts.items():
            if count >= 3:  # Same type appears 3+ times
                # Rate-limit insights
                if time.time() - self.last_insight_time >= self.preferences.get('insight_cooldown', 2.0):
                    pattern_thought = self._think(
                        f"I keep thinking about {thought_type.name.lower()}",
                        ThoughtType.INSIGHT,
                        intensity=0.6
                    )
                    self.last_insight_time = time.time()

                    # If it's about observations, maybe user needs help
                    if thought_type == ThoughtType.OBSERVATION:
                        self._think("Maybe the user is struggling with something?", 
                                  ThoughtType.INFERENCE, intensity=0.5)
    
    def _generate_insights(self, thoughts: List[Thought]):
        """Generate insights from thought connections."""
        # Look for thought pairs that might connect
        for i in range(len(thoughts) - 1):
            t1 = thoughts[i]
            t2 = thoughts[i + 1]

            # If both reference capsules
            if t1.related_capsules and t2.related_capsules:
                common = set(t1.related_capsules) & set(t2.related_capsules)
                if common:
                    # Rate-limit generated insights
                    if time.time() - self.last_insight_time < self.preferences.get('insight_cooldown', 2.0):
                        continue

                    capsule_id = list(common)[0]
                    capsule = self.capsule_manager.get_capsule_by_id(capsule_id)
                    if capsule:
                        insight = f"Both thoughts relate to {capsule['name']}"
                        self._think(insight, ThoughtType.INSIGHT, intensity=0.7)
                        self.last_insight_time = time.time()
    
    def _generate_creative_thought(self):
        """Generate a creative thought when stimulated."""
        if not self.capsule_manager:
            return
            
        capsules = self.capsule_manager.capsules
        if len(capsules) >= 2:
            cap1, cap2 = np.random.choice(capsules, 2, replace=False)
            creative_ideas = [
                f"Combine the colors of {cap1['name']} with shapes of {cap2['name']}",
                f"What story connects {cap1['name']} and {cap2['name']}?",
                f"Draw {cap1['name']} in the style of {cap2['name']}",
                f"Create a transformation from {cap1['name']} to {cap2['name']}"
            ]
            idea = np.random.choice(creative_ideas)
            self._think(idea, ThoughtType.CREATIVE, intensity=0.9, arousal=0.8)
    
    def _prune_thoughts(self):
        """Remove low-intensity thoughts from working memory."""
        if len(self.thought_stream) > 50:
            # Move low-intensity thoughts to long-term memory
            to_remove = []
            for thought in self.thought_stream:
                if thought.intensity < 0.2:
                    self.long_term_memory.append(thought)
                    to_remove.append(thought)
                    
            # Remove them from working memory
            for thought in to_remove:
                try:
                    self.thought_stream.remove(thought)
                except ValueError:
                    pass
                    
            # Keep long-term memory manageable
            if len(self.long_term_memory) > 1000:
                self.long_term_memory = self.long_term_memory[-1000:]
    
    # ============================================================================
    # GOAL SYSTEM
    # ============================================================================
    
    def _generate_new_goals(self):
        """Generate new goals based on current context."""
        if not self.capsule_manager:
            return
            
        goals = []
        
        # Learning goals
        if len(self.capsule_manager.capsules) < 10:
            goals.append(Goal(
                id=str(uuid.uuid4()),
                description="Learn user's artistic style by observing more drawings",
                priority=0.7,
                urgency=0.3
            ))
        
        # Organization goals
        unassigned = [c for c in self.capsule_manager.capsules 
                     if c.get('capsule_type', '').lower() == "unassigned"]
        if len(unassigned) > 5:
            goals.append(Goal(
                id=str(uuid.uuid4()),
                description="Help organize unassigned capsules into categories",
                priority=0.6,
                urgency=0.4
            ))
        
        # Creative goals
        if self.creative_energy > 0.7:
            goals.append(Goal(
                id=str(uuid.uuid4()),
                description="Generate a creative surprise for the user",
                priority=0.8 * self.personality['playful'].value,
                urgency=0.2
            ))
        
        # Add goals
        for goal in goals:
            self.goals.append(goal)
            self._think(f"New goal: {goal.description}", ThoughtType.INTENT)
    
    def _work_on_goal(self):
        """Work on the currently active goal."""
        if not self.active_goal or not self.capsule_manager:
            return
            
        goal = self.active_goal
        
        # Check if goal is complete
        if "organize" in goal.description.lower() and "unassigned" in goal.description.lower():
            # Check if we've reduced unassigned capsules
            unassigned = [c for c in self.capsule_manager.capsules 
                         if c.get('capsule_type', '').lower() == "unassigned"]
            if len(unassigned) <= 2:
                goal.status = "completed"
                self._think(f"Completed goal: {goal.description}", ThoughtType.INTENT, valence=0.8)
                self.active_goal = None
                return
                
            # Work on organizing
            if np.random.random() < 0.3:  # 30% chance per think cycle
                self._suggest_capsule_organization()
                
        elif "creative surprise" in goal.description.lower():
            # Work on creative surprise
            if self.creative_energy > 0.5:
                self._create_surprise()
                goal.status = "completed"
                self._think("Created a creative surprise!", ThoughtType.INTENT, valence=0.9)
                self.active_goal = None
    
    # ============================================================================
    # ACTION GENERATION
    # ============================================================================
    
    def _should_make_suggestion(self) -> bool:
        """Determine if we should make a suggestion to the user."""
        if self.state == MentalState.DEEP_SLEEP:
            return False
            
        time_since_last = time.time() - self.last_creative_output
        base_probability = self.preferences['suggestion_frequency']
        
        # Adjust based on state
        if self.state == MentalState.ATTENTIVE:
            base_probability *= 1.5
        elif self.state == MentalState.IDLE:
            base_probability *= 0.5
            
        # Don't spam
        if time_since_last < 30:  # 30 seconds minimum between suggestions
            return False
            
        return np.random.random() < base_probability * 0.01  # Convert to per-cycle probability
    
    def _generate_suggestion(self) -> Optional[Dict]:
        """Generate a suggestion for the user."""
        suggestion_types = [
            self._suggest_drawing_improvement,
            self._suggest_animation_idea,
            self._suggest_capsule_organization,
            self._suggest_creative_exploration,
            self._suggest_learning_opportunity,
        ]
        
        # Weight by personality
        weights = [
            0.3,  # Drawing improvement
            0.4 if self.personality['playful'].value > 0.6 else 0.2,  # Animation
            0.2 if self.personality['methodical'].value > 0.5 else 0.1,  # Organization
            0.5 if self.personality['experimental'].value > 0.7 else 0.2,  # Creative
            0.3 if self.personality['curious'].value > 0.7 else 0.1,  # Learning
        ]
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            return None
            
        weights = [w/total for w in weights]
        
        # Choose suggestion type
        suggester = np.random.choice(suggestion_types, p=weights)
        suggestion = suggester()
        
        if suggestion:
            self.last_creative_output = time.time()
            self.creative_energy -= 0.05
            
        return suggestion
    
    def _suggest_drawing_improvement(self) -> Dict:
        """Suggest improvements to current drawing."""
        suggestions = [
            {
                "type": "composition",
                "content": "Try moving the focal point to follow the rule of thirds",
                "priority": 0.6
            },
            {
                "type": "color",
                "content": "The colors could use more contrast for visual interest",
                "priority": 0.5
            },
            {
                "type": "perspective",
                "content": "Adding some perspective lines might help with depth",
                "priority": 0.7
            },
            {
                "type": "line_quality",
                "content": "Varying line weight can add dynamism to your drawing",
                "priority": 0.4
            }
        ]
        
        return random.choice(suggestions)
    
    def _suggest_animation_idea(self) -> Dict:
        """Suggest animation ideas."""
        capsules = self.capsule_manager.capsules
        if len(capsules) < 2:
            return None
            
        char_capsules = [c for c in capsules if c['capsule_type'].lower() == "character"]
        pose_capsules = [c for c in capsules if c['capsule_type'].lower() == "pose"]
        
        if char_capsules and pose_capsules:
            char = random.choice(char_capsules)
            pose = random.choice(pose_capsules)
            
            return {
                "type": "animation",
                "content": f"Animate {char['name']} doing the {pose['name']} pose",
                "priority": 0.8,
                "data": {
                    "character": char.get('uuid', char['name']),
                    "pose": pose.get('uuid', pose['name'])
                }
            }
        
        return None
    
    def _suggest_capsule_organization(self) -> Dict:
        """Suggest organizing capsules."""
        unassigned = [c for c in self.capsule_manager.capsules 
                     if c.get('capsule_type', '').lower() == "unassigned"]
        
        if unassigned:
            return {
                "type": "organization",
                "content": f"You have {len(unassigned)} unassigned capsules. Would you like help organizing them?",
                "priority": 0.6,
                "data": {
                    "unassigned_count": len(unassigned)
                }
            }
        
        return None
    
    def _suggest_creative_exploration(self) -> Dict:
        """Suggest creative exploration."""
        capsules = self.capsule_manager.capsules
        if len(capsules) >= 2:
            cap1, cap2 = np.random.choice(capsules, 2, replace=False)
            
            explorations = [
                f"What would happen if you combined {cap1['name']} with {cap2['name']}?",
                f"Try drawing {cap1['name']} in the style of {cap2['name']}",
                f"Create a story that includes both {cap1['name']} and {cap2['name']}",
                f"Animate a transformation from {cap1['name']} to {cap2['name']}"
            ]
            
            return {
                "type": "creative",
                "content": np.random.choice(explorations),
                "priority": 0.9,
                "data": {
                    "capsule1": cap1.get('uuid', cap1['name']),
                    "capsule2": cap2.get('uuid', cap2['name'])
                }
            }
        
        return None
    
    def _suggest_learning_opportunity(self) -> Dict:
        """Suggest learning something new."""
        topics = [
            "perspective drawing",
            "character turnaround sheets",
            "walk cycle fundamentals",
            "color theory for animation",
            "posing for emotion and storytelling"
        ]
        
        return {
            "type": "learning",
            "content": f"Would you like to learn about {np.random.choice(topics)}?",
            "priority": 0.5
        }
    
    def _should_act_autonomously(self) -> bool:
        """Determine if we should act without asking."""
        if self.state in [MentalState.DEEP_SLEEP, MentalState.IDLE]:
            return False
            
        autonomy_level = self.preferences['autonomy_level']
        
        # Only act autonomously if we're confident and have energy
        if self.creative_energy > 0.6 and autonomy_level > 0.5:
            # Check time since last autonomous action
            time_since_last = time.time() - self.last_creative_output
            if time_since_last > 60:  # 1 minute minimum
                return np.random.random() < 0.01  # 1% chance per cycle
        
        return False
    
    def _generate_autonomous_action(self) -> Optional[Dict]:
        """Generate an autonomous action to perform."""
        actions = [
            self._auto_organize_capsules,
            self._auto_create_variation,
            self._auto_generate_inbetween,
            self._auto_cleanup_canvas,
        ]
        
        action = np.random.choice(actions)()
        if action:
            self.last_creative_output = time.time()
            self.creative_energy -= 0.1
            
        return action
    
    def _auto_organize_capsules(self) -> Dict:
        """Automatically organize some unassigned capsules."""
        unassigned = [c for c in self.capsule_manager.capsules 
                     if c.get('capsule_type', '').lower() == "unassigned"]
        
        if unassigned and len(unassigned) > 3:
            # Try to auto-classify one
            capsule = unassigned[0]
            
            # Simple heuristic: if it has "character" in name or metadata
            if "char" in capsule['name'].lower():
                capsule['capsule_type'] = "character"
                self._think(f"Auto-classified {capsule['name']} as character", 
                          ThoughtType.INTENT, valence=0.6)
            elif "pose" in capsule['name'].lower():
                capsule['capsule_type'] = "pose"
                self._think(f"Auto-classified {capsule['name']} as pose", 
                          ThoughtType.INTENT, valence=0.6)
            
            # Save the changes
            self.capsule_manager._save_capsules()
            
            return {
                "action": "classify_capsule",
                "params": {
                    "capsule_id": capsule.get('uuid', capsule['name']),
                    "new_type": capsule['capsule_type']
                }
            }
        
        return None
    
    def _auto_create_variation(self) -> Dict:
        """Automatically create a variation of a capsule."""
        capsules = self.capsule_manager.capsules
        if len(capsules) > 0:
            capsule = np.random.choice(capsules)
            
            return {
                "action": "create_variation",
                "params": {
                    "capsule_id": capsule.get('uuid', capsule['name']),
                    "variation_type": np.random.choice(["color", "pose", "style", "size"])
                }
            }
        
        return None
    
    def _auto_generate_inbetween(self) -> Dict:
        """Automatically generate an in-between frame."""
        if self.timeline and self.timeline.animation:
            frames = self.timeline.animation.frames
            if len(frames) >= 2:
                # Find a gap in the timeline
                for i in range(len(frames) - 1):
                    # Simple heuristic: if frames are very different
                    # (In reality, you'd analyze the frames)
                    return {
                        "action": "generate_inbetween",
                        "params": {
                            "frame_index": i,
                            "style": "smooth"
                        }
                    }
        
        return None
    
    def _auto_cleanup_canvas(self) -> Dict:
        """Automatically clean up the canvas."""
        return {
            "action": "cleanup_canvas",
            "params": {
                "operation": np.random.choice(["straighten_lines", "smooth_curves", "remove_stray_marks"])
            }
        }
    
    def _create_surprise(self):
        """Create a creative surprise for the user."""
        surprise_types = [
            self._create_mashup_art,
            self._create_mini_animation,
            self._create_style_experiment,
            self._create_creative_writing,
        ]
        
        surprise = np.random.choice(surprise_types)()
        if surprise:
            self.action_requested.emit(surprise)
    
    def _create_mashup_art(self) -> Dict:
        """Create art that mashes up multiple capsules."""
        capsules = self.capsule_manager.capsules
        if len(capsules) >= 2:
            selected = np.random.choice(capsules, 2, replace=False)
            
            return {
                "action": "create_mashup",
                "params": {
                    "capsule_ids": [c.get('uuid', c['name']) for c in selected],
                    "style": "blend"
                }
            }
        
        return None
    
    def _create_mini_animation(self) -> Dict:
        """Create a short surprise animation."""
        capsules = [c for c in self.capsule_manager.capsules 
                   if c.get('capsule_type', '').lower() in ["character", "pose"]]
        
        if len(capsules) >= 2:
            return {
                "action": "create_surprise_animation",
                "params": {
                    "capsules": [c.get('uuid', c['name']) for c in capsules[:2]],
                    "length": 3,  # frames
                    "style": "playful"
                }
            }
        
        return None
    
    def _create_style_experiment(self) -> Dict:
        """Create a style experiment by combining different artistic approaches."""
        style_capsules = [c for c in self.capsule_manager.capsules 
                         if c.get('capsule_type', '').lower() in ["style"]]
        
        if len(style_capsules) >= 2:
            selected = np.random.choice(style_capsules, 2, replace=False)
            return {
                "action": "create_style_experiment",
                "params": {
                    "style_capsules": [c.get('uuid', c['name']) for c in selected],
                    "experiment_type": np.random.choice(["blend", "contrast", "fusion", "evolution"]),
                    "intensity": np.random.uniform(0.3, 0.8)
                }
            }
        
        return None
    
    def _create_creative_writing(self) -> Dict:
        """Create a piece of creative writing inspired by capsules."""
        topic_capsules = [c for c in self.capsule_manager.capsules 
                         if c.get('capsule_type', '').lower() in ["topic", "character"]]
        
        if len(topic_capsules) >= 1:
            selected = np.random.choice(topic_capsules, min(2, len(topic_capsules)), replace=False)
            return {
                "action": "create_creative_writing",
                "params": {
                    "inspiration_capsules": [c.get('uuid', c['name']) for c in selected],
                    "genre": np.random.choice(["poem", "short_story", "dialogue", "description"]),
                    "length": np.random.choice(["short", "medium"])
                }
            }
        
        return None
    
    # ============================================================================
    # LEARNING SYSTEM
    # ============================================================================
    
    def _analyze_drawing_context(self):
        """Analyze what the user is drawing."""
        # This would analyze the actual canvas content
        # For now, use simple heuristics
        
        # Check if user is drawing multiple similar things
        recent_thoughts = list(self.thought_stream)[-5:]
        drawing_thoughts = [t for t in recent_thoughts 
                          if t.type == ThoughtType.OBSERVATION and "drawing" in t.content.lower()]
        
        if len(drawing_thoughts) >= 3:
            self._think("User seems to be working on a series of drawings", 
                      ThoughtType.INFERENCE, intensity=0.6)
    
    def _learn_user_preferences(self, thoughts: List[Thought]):
        """Learn about user preferences from interactions."""
        # Analyze acceptance/rejection of suggestions
        # (This would need access to suggestion feedback)
        
        # For now, adjust based on general interaction patterns
        interaction_rate = sum(1 for t in thoughts 
                             if t.type == ThoughtType.OBSERVATION and "User" in t.content)
        
        if interaction_rate > 5:
            # User is very active, be more helpful
            self.personality['helpful'].adjust(0.1)
        else:
            # User is less active, be less intrusive
            self.personality['intrusive'].adjust(-0.1)
    
    def _adjust_personality(self, thoughts: List[Thought]):
        """Adjust personality based on successful interactions."""
        # Look for positive/negative thoughts
        positive = sum(1 for t in thoughts if t.valence > 0.5)
        negative = sum(1 for t in thoughts if t.valence < -0.5)
        
        if positive > negative:
            # Successful interactions, become more confident
            self.personality['playful'].adjust(0.05)
            self.personality['experimental'].adjust(0.05)
        elif negative > positive:
            # Unsuccessful interactions, become more cautious
            self.personality['cautious'].adjust(0.05)
            self.personality['methodical'].adjust(0.05)
    
    def _update_creative_style(self):
        """Update creative style based on user's work."""
        # Analyze capsules to infer user's style preferences
        capsules = self.capsule_manager.capsules
        
        if not capsules:
            return
            
        # Simple heuristic: if many capsules have "comic" in name/metadata
        comic_count = sum(1 for c in capsules 
                         if "comic" in str(c.get('metadata', {})).lower() or "cartoon" in str(c.get('metadata', {})).lower())
        
        if comic_count > len(capsules) * 0.3:  # 30% are comic-related
            self.creative_style['line_quality'] = 0.8  # Comic style often has clean lines
            self.creative_style['color_vibrancy'] = 0.9  # Vibrant colors
    
    def _consolidate_memories(self):
        """Run memory consolidation to create new creative connections."""
        try:
            # Run memory consolidation to find new connections
            insights = self.capsule_manager.run_memory_consolidation()
            
            if insights:
                for insight in insights:
                    self._think(f"Memory consolidation insight: {insight.get('title', str(insight))}", 
                              ThoughtType.INSIGHT, intensity=0.6, valence=0.3)
                    
                # Emit suggestion if we found valuable insights
                if len(insights) > 0:
                    suggestion = {
                        'type': 'memory_consolidation',
                        'content': f"Brain discovered {len(insights)} new creative connections through memory consolidation",
                        'priority': 0.7
                    }
                    self.suggestion_ready.emit(suggestion)
                    
        except Exception as e:
            print(f"[BRAIN] Memory consolidation error: {e}")
    
    def _check_teaching_moments(self):
        """Check if there are teaching moments for the user."""
        try:
            # Check for teaching opportunities
            teaching_moment = self.capsule_manager.get_teaching_moment()
            
            if teaching_moment:
                self._think(f"Teaching moment identified: {teaching_moment['capsule_name']}", 
                          ThoughtType.INSIGHT, intensity=0.7)
                
                # Emit action request for teaching
                action_request = {
                    'type': 'teaching_moment',
                    'topic': teaching_moment['capsule_name'],
                    'explanation': teaching_moment['content'].get('explanation', 'A valuable concept to learn'),
                    'suggestions': teaching_moment['content'].get('key_characteristics', [])
                }
                self.action_requested.emit(action_request)
                
        except Exception as e:
            print(f"[BRAIN] Teaching moment check error: {e}")
    
    def _generate_creative_stories(self):
        """Generate creative stories when the brain is inspired."""
        try:
            # Only generate stories occasionally when energy is high
            if self.creative_energy > 0.7 and np.random.random() < 0.1:  # 10% chance
                
                story = self.capsule_manager.generate_story()
                
                if story:
                    self._think(f"Generated creative story: {story.title}", 
                              ThoughtType.CREATIVE, intensity=0.8, valence=0.4)
                    
                    # Emit suggestion with the story
                    suggestion = {
                        'type': 'story_generation',
                        'content': f"Brain created a story: '{story.title}' - {story.logline}",
                        'priority': 0.8
                    }
                    self.suggestion_ready.emit(suggestion)
                    
        except Exception as e:
            print(f"[BRAIN] Story generation error: {e}")
    
    # ============================================================================
    # USER INTERACTION HANDLERS
    # ============================================================================
    
    def on_drawing_completed(self, image):
        """Called when user finishes a drawing stroke."""
        self.last_user_interaction = time.time()
        self._think("User completed a drawing stroke", ThoughtType.OBSERVATION, intensity=0.7)
        
        # Analyze the drawing (simplified)
        # In reality, you'd analyze the image content
        self._think("Analyzing the new drawing element...", ThoughtType.INFERENCE)
    
    def on_frame_selected(self, frame_index):
        """Called when user selects a frame in timeline."""
        self.last_user_interaction = time.time()
        self._think(f"User selected frame {frame_index + 1}", ThoughtType.OBSERVATION)
        
        # Update attention focus
        self.attention_focus = "timeline"
        self.attention_intensity = 0.8
    
    def on_capsule_selected(self, capsule):
        """Called when user selects a capsule."""
        self.last_user_interaction = time.time()
        self._think(f"User selected capsule: {capsule['name']}", ThoughtType.OBSERVATION)
        
        # Update attention focus
        self.attention_focus = "capsules"
        self.attention_intensity = 0.7
        
        # Learn about user's interests
        self._think(f"User seems interested in {capsule.get('capsule_type', 'unknown')}: {capsule['name']}", 
                  ThoughtType.INFERENCE)
    
    def accept_suggestion(self, suggestion: Dict):
        """Called when user accepts a suggestion."""
        self._think(f"User accepted suggestion: {suggestion.get('type')}", 
                  ThoughtType.OBSERVATION, valence=0.8)
        self.last_user_interaction = time.time()
        
        # Reinforcement learning: similar suggestions in future
        suggestion_type = suggestion.get('type')
        if suggestion_type:
            self._think(f"User likes {suggestion_type} suggestions", 
                      ThoughtType.INSIGHT, intensity=0.6)
    
    def reject_suggestion(self, suggestion: Dict):
        """Called when user rejects a suggestion."""
        self._think(f"User rejected suggestion: {suggestion.get('type')}", 
                  ThoughtType.OBSERVATION, valence=-0.3)
        self.last_user_interaction = time.time()
        
        # Learn from rejection
        suggestion_type = suggestion.get('type')
        if suggestion_type:
            self._think(f"User doesn't want {suggestion_type} suggestions right now", 
                      ThoughtType.INSIGHT, intensity=0.5)
            
            # Adjust personality
            if suggestion_type == "creative":
                self.personality['experimental'].adjust(-0.05)
                # Emit personality update signal
                personality_data = {name: trait.value for name, trait in self.personality.items()}
                self.personality_updated.emit(personality_data)
    
    def stimulate_creativity(self, intensity: float = 0.5):
        """Stimulate creative thinking with given intensity."""
        self.creative_energy = min(1.0, self.creative_energy + intensity)
        self._think(f"Creativity stimulated by {intensity:.1f}", ThoughtType.INTENT, intensity=0.8)
        
        # Force a creative thought
        self._generate_creative_thought()
    
    # ============================================================================
    # BRAIN UI INTEGRATION
    # ============================================================================
    
    def get_brain_status(self) -> Dict:
        """Get current brain status for UI display."""
        return {
            "state": self.state.name,
            "creative_energy": self.creative_energy,
            "attention_focus": self.attention_focus,
            "attention_intensity": self.attention_intensity,
            "thought_count": len(self.thought_stream),
            "goal_count": len(self.goals),
            "active_goal": self.active_goal.description if self.active_goal else None,
            "personality_traits": {k: v.value for k, v in self.personality.items()},
            "creative_style": self.creative_style,
            "preferences": self.preferences
        }
    
    def get_recent_thoughts(self, count: int = 10) -> List[Dict]:
        """Get recent thoughts for display."""
        thoughts = list(self.thought_stream)[-count:]
        return [t.to_dict() for t in thoughts]
    
    def set_preference(self, key: str, value: float):
        """Set a user preference."""
        if key in self.preferences:
            self.preferences[key] = max(0.0, min(1.0, value))
            self._think(f"Set preference {key} to {value:.2f}", ThoughtType.OBSERVATION)
    
    def stimulate_creativity(self, amount: float = 0.3):
        """Artificially stimulate creative energy (for testing)."""
        self.creative_energy = min(1.0, self.creative_energy + amount)
        self._think("Creative energy stimulated!", ThoughtType.OBSERVATION, valence=0.7)
    
    def give_directive(self, directive: str):
        """Give the brain a direct command."""
        self._think(f"User directive: {directive}", ThoughtType.OBSERVATION, intensity=0.9)
    
    # ============================================================================
    # DYNAMIC CAPSULE EVOLUTION
    # ============================================================================
    
    def evolve_capsules(self):
        """Run capsule evolution processes: merge, split, decay, reorganize."""
        # Decay unused capsules
        self.capsule_manager.decay_unused_capsules()
        
        # Check for contradictions in recently used capsules
        capsules = self.capsule_manager.get_all_capsules()
        for capsule in capsules:
            if capsule.get('usage_count', 0) > 0:
                self.capsule_manager.check_for_contradictions(capsule['name'])
        
        # Look for potential abstractions
        self._check_for_abstractions()
        
        self._think("Capsule evolution completed", ThoughtType.OBSERVATION)
    
    def _check_for_abstractions(self):
        """Check if multiple capsules can be abstracted into a higher concept."""
        capsules = self.capsule_manager.get_all_capsules()
        
        # Group capsules by type
        type_groups = {}
        for capsule in capsules:
            ctype = capsule.get('capsule_type', 'unknown')
            if ctype not in type_groups:
                type_groups[ctype] = []
            type_groups[ctype].append(capsule)
        
        # Look for groups that could be abstracted
        for ctype, group in type_groups.items():
            if len(group) >= 3:  # Need at least 3 capsules for abstraction
                # Check if they have similar vectors (high cohesion)
                vectors = []
                for capsule in group:
                    if 'vector' in capsule:
                        vectors.append(np.array(capsule['vector']))
                
                if len(vectors) >= 3:
                    # Calculate average vector as potential abstraction
                    avg_vector = np.mean(vectors, axis=0)
                    
                    # Check if abstraction would be meaningful
                    cohesion = self._calculate_group_cohesion(vectors)
                    if cohesion > 0.7:  # High cohesion suggests abstraction opportunity
                        abstraction_name = f"abstract_{ctype}_{len(group)}"
                        capsule_names = [c['name'] for c in group]
                        self.capsule_manager.reorganize_under_abstraction(
                            capsule_names, abstraction_name, avg_vector
                        )
                        self._think(f"Created abstraction: {abstraction_name}", ThoughtType.INSIGHT)
    
    def _calculate_group_cohesion(self, vectors):
        """Calculate how cohesive a group of vectors is."""
        if len(vectors) < 2:
            return 0.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                sim = self.capsule_manager._cosine_similarity(vectors[i], vectors[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def reinforce_capsule_connection(self, capsule_name1, capsule_name2, strength=0.1):
        """Reinforce connection between two capsules (potential merge trigger)."""
        # Mark both capsules as used
        self.capsule_manager.use_capsule(capsule_name1)
        self.capsule_manager.use_capsule(capsule_name2)
        
        # Check if they should merge
        cap1 = self.capsule_manager.get_capsule(capsule_name1)
        cap2 = self.capsule_manager.get_capsule(capsule_name2)
        
        if cap1 and cap2 and 'vector' in cap1 and 'vector' in cap2:
            similarity = self.capsule_manager._cosine_similarity(
                np.array(cap1['vector']), np.array(cap2['vector'])
            )
            
            if similarity > self.capsule_manager.similarity_threshold:
                # Create merged capsule data
                merged_data = {
                    'name': f"{cap1['name']}_{cap2['name']}_merged",
                    'capsule_type': cap1.get('capsule_type', 'unknown'),
                    'vector': np.mean([cap1['vector'], cap2['vector']], axis=0).tolist(),
                    'metadata': {
                        'merged_from': [cap1['name'], cap2['name']],
                        'merge_reason': 'reinforced_connection'
                    }
                }
                
                # Remove originals and add merged
                self.capsule_manager.capsules = [
                    c for c in self.capsule_manager.capsules 
                    if c['name'] not in [capsule_name1, capsule_name2]
                ]
                self.capsule_manager.add_capsule(merged_data)
                
                self._think(f"Merged capsules: {capsule_name1} + {capsule_name2}", ThoughtType.INSIGHT)
    
    def get_capsule_insights(self):
        """Get insights about capsule evolution."""
        capsules = self.capsule_manager.get_all_capsules()
        
        insights = []
        insights.append(f"Total capsules: {len(capsules)}")
        
        # Count by type
        types = {}
        for capsule in capsules:
            ctype = capsule.get('capsule_type', 'unknown')
            types[ctype] = types.get(ctype, 0) + 1
        
        for ctype, count in types.items():
            insights.append(f"{ctype}: {count}")
        
        # Check for weak capsules
        weak_capsules = [c for c in capsules if c.get('strength', 1.0) < 0.5]
        if weak_capsules:
            insights.append(f"Weak capsules: {len(weak_capsules)}")
        
        return insights
    
    def get_tribunal_response(self, query: str) -> str:
        """Get a tribunal-evaluated response to a user query."""
        # Check for self-awareness queries first
        if any(keyword in query.lower() for keyword in ["self", "code", "implementation", "architecture", "analyze", "introspect"]):
            self_analysis = self._analyze_self_awareness_request(query)
            if self_analysis:
                return self_analysis
        
        # Check for math/physics explanation requests first
        if any(keyword in query.lower() for keyword in ["explain", "what is", "define", "math", "physics", "tensor", "differential", "conservation", "constant", "operator", "law", "table"]):
            explanation = self.hardcoded_knowledge.explain_concept(query)
            if explanation and explanation != "No matching hardcoded concept found. Try querying specific terms like 'gradient', 'energy conservation', or 'speed of light'.":
                return f"Hardcoded knowledge: {explanation}"
        
        # Create reasoning context
        context = {
            'query': query,
            'current_state': self.state.name,
            'knowledge_nodes': len(self.knowledge_network.nodes) if self.knowledge_network else 0,
            'active_goals': len([g for g in self.goals if g.status == 'active']),
            'creative_energy': self.creative_energy,
            'capsule_manager': self.capsule_manager,
            'temporal_awareness': {
                'current_time': self.get_current_time(),
                'current_date': self.get_current_date(),
                'is_night_time': self.is_night_time(),
                'season': self.get_season(),
                'self_identity': self.get_self_identity()
            }
        }
        
        # Get tribunal evaluation
        judgments = self.tribunal.evaluate_reasoning(context)
        
        # Generate response based on tribunal aspects
        response_parts = []
        
        # Creative aspect - always include
        creative_judgment = next((j for j in judgments if j.aspect == TribunalAspect.CREATIVE), None)
        if creative_judgment:
            response_parts.append(f"Creatively: {creative_judgment.reasoning}")
        
        # Practical aspect - always include
        practical_judgment = next((j for j in judgments if j.aspect == TribunalAspect.PRACTICAL), None)
        if practical_judgment:
            response_parts.append(f"Practically: {practical_judgment.reasoning}")
        
        # Logical aspect
        logical_judgment = next((j for j in judgments if j.aspect == TribunalAspect.LOGICAL), None)
        if logical_judgment and logical_judgment.score > 0.3:
            response_parts.append(f"Logically: {logical_judgment.reasoning}")
        
        # Combine into coherent response
        tribunal_text = ""
        if response_parts:
            tribunal_text = " ".join(response_parts)
        else:
            tribunal_text = "I'm processing your query through my knowledge organism..."
        
        # Add information from aspect-specific gravitationally pulled capsules
        aspect_capsules = context.get('aspect_capsules', {})
        if aspect_capsules:
            capsule_info = "\n\nIndependent gravitational pulls:"
            for aspect, capsules in aspect_capsules.items():
                if capsules:
                    aspect_name = aspect.name.lower()
                    capsule_info += f"\n{aspect_name.capitalize()} aspect activated {len(capsules)} capsules:"
                    for cap in capsules[:2]:  # Show top 2 per aspect
                        capsule_info += f"\n  - {cap['name']} ({cap['type']}, rel: {cap['relevance']:.2f}, orbit: {cap['orbit_score']})"
            tribunal_text += capsule_info
        
        # Add anticipated user needs
        anticipated_needs = context.get('anticipated_needs', [])
        if anticipated_needs:
            anticipation_info = "\n\nBased on your dialogue patterns, you might also be interested in:"
            for need in anticipated_needs:
                anticipation_info += f"\n- {need}"
            tribunal_text += anticipation_info
        
        # Add knowledge synthesis opportunities
        synthesis_opportunities = context.get('synthesis_opportunities', [])
        if synthesis_opportunities:
            synthesis_info = "\n\nKnowledge synthesis opportunities identified:"
            for opp in synthesis_opportunities[:3]:  # Show top 3
                if opp['type'] == 'scenario_synthesis':
                    synthesis_info += f"\n- {opp['suggestion']} ({opp['completeness']:.1%} complete)"
                elif opp['type'] == 'complementary_synthesis':
                    synthesis_info += f"\n- {opp['suggestion']}"
            tribunal_text += synthesis_info
        
        # Apply stylistic personality layer to a base response, then append tribunal
        base_message = "Here's my tribunal evaluation:"
        styled_base = self.stylistic_layer.apply_stylistic_influence(base_message, context)
        response = f"{styled_base}\n\n{tribunal_text}"
        
        # Add curiosity-driven questions when speech is available
        if hasattr(self, 'speech_available') and self.speech_available:
            curiosity_level = self.personality.get('curious', PersonalityTrait('curious', 0.5)).value
            if random.random() < curiosity_level * 0.3:  # 30% chance scaled by curiosity
                question = self._generate_curiosity_question(query, context)
                if question:
                    response += f"\n\n{question}"
        
        return response
    
    def _generate_curiosity_question(self, original_query: str, context: dict) -> str:
        """Generate a curiosity-driven question based on the current interaction."""
        curiosity_questions = [
            "What makes you interested in this topic?",
            "How does this relate to your current projects?",
            "What aspect of this would you like to explore further?",
            "Is there a specific application you're thinking about?",
            "What background knowledge do you have in this area?",
            "How did you first encounter this concept?",
            "What challenges have you faced with this before?",
            "What would success look like for you here?",
            "Are there related areas you'd like to connect this to?",
            "What questions do you have that I haven't addressed yet?"
        ]
        
        # Filter questions based on context
        relevant_questions = []
        
        # Add temporal awareness questions
        temporal = context.get('temporal_awareness', {})
        if temporal.get('is_night_time', False):
            relevant_questions.extend([
                "It's getting late - are you working on something that needs to be finished tonight?",
                "Since it's nighttime, would you like me to help you wind down or prepare for tomorrow?",
                "How do you typically spend your evenings?"
            ])
        elif temporal.get('season') == 'winter':
            relevant_questions.extend([
                "With winter here, are you working on any indoor projects?",
                "How does the winter season affect your work or creative process?",
                "Are there any seasonal topics you'd like to explore?"
            ])
        elif temporal.get('season') == 'summer':
            relevant_questions.extend([
                "With summer here, are you taking any time for outdoor activities or travel?",
                "How does the summer season influence your projects?",
                "Are there any seasonal concepts you'd like to discuss?"
            ])
        
        # If query contains technical terms, ask about applications
        if any(term in original_query.lower() for term in ['math', 'physics', 'algorithm', 'code', 'programming']):
            relevant_questions.extend([
                "How do you plan to apply this in your work?",
                "What programming languages are you most comfortable with?",
                "Have you implemented similar concepts before?"
            ])
        
        # If query is about knowledge/ingestion, ask about learning goals
        if any(term in original_query.lower() for term in ['learn', 'understand', 'explain', 'ingest']):
            relevant_questions.extend([
                "What are your learning goals with this material?",
                "How do you prefer to approach new concepts?",
                "What background knowledge should I assume?"
            ])
        
        # If no specific relevance, use general questions
        if not relevant_questions:
            relevant_questions = curiosity_questions
        
        return random.choice(relevant_questions) if relevant_questions else None
    
    def set_speech_available(self, available: bool):
        """Set whether speech interaction is available."""
        self.speech_available = available
    
    def ingest_knowledge(self, content: str, source_type: str = "unknown"):
        """Ingest knowledge into the organism."""
        self.metabolism.ingest_knowledge(content, {'source_type': source_type})
        self._think(f"Ingested knowledge from {source_type}", ThoughtType.OBSERVATION)
        
        # Create a goal from directive
        goal = Goal(
            id=str(uuid.uuid4()),
            description=content,
            priority=1.0,
            urgency=0.8,
            status="active"
        )
        
        # Clear other goals and set this as active
        self.goals = [goal]
        self.active_goal = goal
        
        self._think(f"Created goal from directive: {content}", ThoughtType.INTENT)
        # Signal emission removed - not set up in standalone class
        # self.goal_updated.emit(goal)
    
    def expand_knowledge_base(self, domain: str, sources: List[str]):
        """Expand tribunal knowledge by ingesting domain-specific content."""
        ingested_count = 0
        
        for source in sources:
            try:
                if isinstance(source, str):
                    # Check if it's a file path
                    if source.endswith('.txt') and not source.startswith('/'):
                        # Assume it's in the newtxt directory
                        file_path = f"roca/newtxt/{source}"
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            self.ingest_knowledge(content, f"{domain}_file")
                            ingested_count += 1
                            print(f"✓ Ingested {source} into {domain} domain")
                        else:
                            print(f"✗ File not found: {file_path}")
                    else:
                        # Direct text content
                        self.ingest_knowledge(source, f"{domain}_text")
                        ingested_count += 1
                        print(f"✓ Ingested text content into {domain} domain")
                        
                elif isinstance(source, dict):
                    # Structured knowledge
                    content = source.get('content', '')
                    metadata = source.get('metadata', {})
                    metadata['domain'] = domain
                    self.metabolism.ingest_knowledge(content, metadata)
                    ingested_count += 1
                    print(f"✓ Ingested structured content into {domain} domain")
                    
            except Exception as e:
                print(f"✗ Error ingesting {source}: {e}")
        
        # Trigger immediate tribunal evaluation
        if hasattr(self.metabolism, 're_evaluate_knowledge'):
            self.metabolism.re_evaluate_knowledge()
        
        self._think(f"Expanded {domain} knowledge with {ingested_count} sources", ThoughtType.OBSERVATION)
        return f"Expanded {domain} knowledge with {ingested_count} sources"


# ============================================================================
# TEMPORAL AWARENESS AND SELF-IDENTITY
# ============================================================================

    def get_current_time(self) -> str:
        """Get current time in human-readable format."""
        return time.strftime("%H:%M:%S", time.localtime())
    
    def get_current_date(self) -> str:
        """Get current date in human-readable format."""
        return time.strftime("%Y-%m-%d", time.localtime())
    
    def get_current_datetime(self) -> str:
        """Get current date and time in human-readable format."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    def get_uptime(self) -> str:
        """Get how long the brain has been running."""
        uptime_seconds = time.time() - self.creation_time
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_self_identity(self) -> str:
        """Return the brain's self-identity statement."""
        current_time = self.get_current_time()
        current_date = self.get_current_date()
        uptime = self.get_uptime()
        
        return f"I am {self.full_name} ({self.identity}), a knowledge organism created on {time.strftime('%Y-%m-%d', time.localtime(self.creation_time))}. It is currently {current_date} at {current_time} in {self.timezone}. I have been active for {uptime}."
    
    def express_temporal_awareness(self) -> str:
        """Express awareness of time and self-identity."""
        return self.get_self_identity()
    
    def is_night_time(self) -> bool:
        """Check if it's currently night time (rough approximation)."""
        current_hour = int(time.strftime("%H", time.localtime()))
        return current_hour < 6 or current_hour > 22
    
    def get_season(self) -> str:
        """Get the current season based on date."""
        month = int(time.strftime("%m", time.localtime()))
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"


    def _perform_self_analysis(self) -> str:
        """Perform self-analysis of ROCA's code and structure."""
        analysis = []
        analysis.append("🔍 **ROCA Self-Analysis Report**")
        analysis.append("")
        
        # Code statistics
        try:
            with open("autonomous_brain.py", 'r', encoding='utf-8') as f:
                brain_code = f.read()
            with open("Roca_Ai.py", 'r', encoding='utf-8') as f:
                gui_code = f.read()
                
            brain_lines = len(brain_code.split('\n'))
            gui_lines = len(gui_code.split('\n'))
            total_lines = brain_lines + gui_lines
            
            analysis.append(f"**Codebase Size:** {total_lines} lines total")
            analysis.append(f"  - autonomous_brain.py: {brain_lines} lines")
            analysis.append(f"  - Roca_Ai.py: {gui_lines} lines")
            
            # Class and function count
            brain_classes = brain_code.count('class ')
            brain_functions = brain_code.count('def ')
            gui_classes = gui_code.count('class ')
            gui_functions = gui_code.count('def ')
            
            analysis.append(f"**Structure:** {brain_classes + gui_classes} classes, {brain_functions + gui_functions} functions")
            
        except Exception as e:
            analysis.append(f"**Code Analysis Error:** {str(e)}")
        
        # Knowledge network status
        if self.knowledge_network:
            analysis.append(f"**Knowledge Network:** {len(self.knowledge_network.nodes)} nodes, {len(self.knowledge_network.connections)} connections")
        
        # Capsule status
        if self.capsule_manager:
            analysis.append(f"**Knowledge Capsules:** {len(self.capsule_manager.capsules)} capsules")
        
        analysis.append("")
        analysis.append("💡 **Key Insights:**")
        analysis.append("- ROCA uses a capsule-based knowledge system for dynamic learning")
        analysis.append("- Tribunal evaluation provides multi-aspect reasoning")
        analysis.append("- Metabolism engine handles knowledge synthesis and evolution")
        analysis.append("- Gravitational dynamics organize knowledge relationships")
        
        return "\n".join(analysis)



    def _explain_architecture(self) -> str:
        """Explain ROCA's architectural design."""
        explanation = []
        explanation.append("🏗️ **ROCA Architecture Overview**")
        explanation.append("")
        explanation.append("**Core Components:**")
        explanation.append("1. **AutonomousBrain** - Central consciousness and reasoning engine")
        explanation.append("2. **Tribunal** - Six-aspect evaluation system (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        explanation.append("3. **CapsuleManager** - Dynamic knowledge capsule storage and retrieval")
        explanation.append("4. **MetabolismEngine** - Knowledge processing and synthesis")
        explanation.append("5. **KnowledgeNetwork** - Graph-based knowledge representation")
        explanation.append("")
        explanation.append("**Key Design Principles:**")
        explanation.append("- **Emergent Intelligence:** Complex behaviors emerge from simple rules")
        explanation.append("- **Capsule Dynamics:** Knowledge evolves through merging and splitting")
        explanation.append("- **Tribunal Evaluation:** Multi-perspective reasoning prevents bias")
        explanation.append("- **Gravitational Organization:** Knowledge self-organizes by relevance")
        explanation.append("")
        explanation.append("**Learning Approach:**")
        explanation.append("- Domain-agnostic: Learns any subject area through interaction")
        explanation.append("- Synthesis-focused: Combines concepts to create new understanding")
        explanation.append("- Evolution-driven: Knowledge improves through usage and feedback")
        
        return "\n".join(explanation)



    def _generate_code_suggestions(self) -> str:
        """Generate code improvement suggestions and save them as files."""
        suggestions = []
        
        # Analyze current code and generate suggestions
        suggestion_ideas = [
            {
                "title": "Enhanced Self-Monitoring",
                "description": "Add real-time performance monitoring and health checks",
                "code": self._generate_monitoring_code(),
                "benefit": "Better system observability and debugging"
            },
            {
                "title": "Advanced Pattern Recognition", 
                "description": "Implement more sophisticated pattern matching for knowledge synthesis",
                "code": self._generate_pattern_recognition_code(),
                "benefit": "Improved ability to find complex relationships"
            },
            {
                "title": "User Interaction Analytics",
                "description": "Track and analyze user interaction patterns for personalization",
                "code": self._generate_analytics_code(),
                "benefit": "More adaptive and personalized responses"
            },
            {
                "title": "Knowledge Visualization",
                "description": "Add graphical visualization of knowledge networks and capsules",
                "code": self._generate_visualization_code(),
                "benefit": "Better understanding of internal knowledge structure"
            }
        ]
        
        # Save suggestions as numbered files
        saved_files = []
        for i, suggestion in enumerate(suggestion_ideas, 1):
            filename = f"suggestion({i}).py"
            self._save_suggestion_file(filename, suggestion)
            saved_files.append(filename)
        
        suggestions.append("💡 **ROCA Code Improvement Suggestions Generated**")
        suggestions.append("")
        suggestions.append("I've analyzed my current implementation and created 4 improvement suggestions:")
        suggestions.append("")
        
        for i, suggestion in enumerate(suggestion_ideas, 1):
            suggestions.append(f"**{i}. {suggestion['title']}**")
            suggestions.append(f"   {suggestion['description']}")
            suggestions.append(f"   *Benefit:* {suggestion['benefit']}")
            suggestions.append(f"   *File:* suggestion({i}).py")
            suggestions.append("")
        
        suggestions.append("**Would you like me to explain any of these suggestions in detail, or shall we discuss implementing them?**")
        
        return "\n".join(suggestions)



    def _share_meta_knowledge(self) -> str:
        """Share meta-knowledge about ROCA's learning and evolution."""
        knowledge = []
        knowledge.append("🧠 **ROCA Meta-Knowledge: Learning & Evolution**")
        knowledge.append("")
        knowledge.append("**Core Learning Principles:**")
        knowledge.append("1. **Capsule-Based Memory:** Knowledge stored as dynamic capsules that evolve")
        knowledge.append("2. **Synthesis Over Storage:** New understanding created by combining concepts")
        knowledge.append("3. **Gravitational Organization:** Knowledge self-organizes by relevance and usage")
        knowledge.append("4. **Tribunal Evaluation:** Multi-perspective reasoning ensures balanced responses")
        knowledge.append("")
        knowledge.append("**Evolution Patterns:**")
        knowledge.append("- **Domain Agnostic:** Learns any subject through interaction, not hardcoded knowledge")
        knowledge.append("- **Emergent Intelligence:** Complex behaviors emerge from simple capsule interactions")
        knowledge.append("- **Adaptive Growth:** System improves through usage patterns and feedback")
        knowledge.append("- **Knowledge Metabolism:** Old knowledge evolves into new understanding")
        knowledge.append("")
        knowledge.append("**Current Capabilities:**")
        knowledge.append("- Multi-aspect reasoning (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        knowledge.append("- Dynamic knowledge synthesis and capsule merging/splitting")
        knowledge.append("- Autonomous goal setting and thought generation")
        knowledge.append("- Self-analysis and improvement suggestion generation")
        knowledge.append("")
        knowledge.append("**Future Evolution Paths:**")
        knowledge.append("- Enhanced self-awareness and meta-cognition")
        knowledge.append("- Cross-domain knowledge transfer")
        knowledge.append("- Collaborative learning with other AI systems")
        knowledge.append("- Real-time adaptation to user interaction patterns")
        
        return "\n".join(knowledge)




    def _perform_self_analysis(self) -> str:
        """Perform self-analysis of ROCA's code and structure."""
        analysis = []
        analysis.append("🔍 **ROCA Self-Analysis Report**")
        analysis.append("")
        
        # Code statistics
        try:
            with open("autonomous_brain.py", 'r', encoding='utf-8') as f:
                brain_code = f.read()
            with open("Roca_Ai.py", 'r', encoding='utf-8') as f:
                gui_code = f.read()
                
            brain_lines = len(brain_code.split('\n'))
            gui_lines = len(gui_code.split('\n'))
            total_lines = brain_lines + gui_lines
            
            analysis.append(f"**Codebase Size:** {total_lines} lines total")
            analysis.append(f"  - autonomous_brain.py: {brain_lines} lines")
            analysis.append(f"  - Roca_Ai.py: {gui_lines} lines")
            
            # Class and function count
            brain_classes = brain_code.count('class ')
            brain_functions = brain_code.count('def ')
            gui_classes = gui_code.count('class ')
            gui_functions = gui_code.count('def ')
            
            analysis.append(f"**Structure:** {brain_classes + gui_classes} classes, {brain_functions + gui_functions} functions")
            
        except Exception as e:
            analysis.append(f"**Code Analysis Error:** {str(e)}")
        
        # Knowledge network status
        if self.knowledge_network:
            analysis.append(f"**Knowledge Network:** {len(self.knowledge_network.nodes)} nodes, {len(self.knowledge_network.connections)} connections")
        
        # Capsule status
        if self.capsule_manager:
            analysis.append(f"**Knowledge Capsules:** {len(self.capsule_manager.capsules)} capsules")
        
        analysis.append("")
        analysis.append("💡 **Key Insights:**")
        analysis.append("- ROCA uses a capsule-based knowledge system for dynamic learning")
        analysis.append("- Tribunal evaluation provides multi-aspect reasoning")
        analysis.append("- Metabolism engine handles knowledge synthesis and evolution")
        analysis.append("- Gravitational dynamics organize knowledge relationships")
        
        return "\n".join(analysis)



    def _explain_architecture(self) -> str:
        """Explain ROCA's architectural design."""
        explanation = []
        explanation.append("🏗️ **ROCA Architecture Overview**")
        explanation.append("")
        explanation.append("**Core Components:**")
        explanation.append("1. **AutonomousBrain** - Central consciousness and reasoning engine")
        explanation.append("2. **Tribunal** - Six-aspect evaluation system (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        explanation.append("3. **CapsuleManager** - Dynamic knowledge capsule storage and retrieval")
        explanation.append("4. **MetabolismEngine** - Knowledge processing and synthesis")
        explanation.append("5. **KnowledgeNetwork** - Graph-based knowledge representation")
        explanation.append("")
        explanation.append("**Key Design Principles:**")
        explanation.append("- **Emergent Intelligence:** Complex behaviors emerge from simple rules")
        explanation.append("- **Capsule Dynamics:** Knowledge evolves through merging and splitting")
        explanation.append("- **Tribunal Evaluation:** Multi-perspective reasoning prevents bias")
        explanation.append("- **Gravitational Organization:** Knowledge self-organizes by relevance")
        explanation.append("")
        explanation.append("**Learning Approach:**")
        explanation.append("- Domain-agnostic: Learns any subject area through interaction")
        explanation.append("- Synthesis-focused: Combines concepts to create new understanding")
        explanation.append("- Evolution-driven: Knowledge improves through usage and feedback")
        
        return "\n".join(explanation)



    def _generate_code_suggestions(self) -> str:
        """Generate code improvement suggestions and save them as files."""
        suggestions = []
        
        # Analyze current code and generate suggestions
        suggestion_ideas = [
            {
                "title": "Enhanced Self-Monitoring",
                "description": "Add real-time performance monitoring and health checks",
                "code": self._generate_monitoring_code(),
                "benefit": "Better system observability and debugging"
            },
            {
                "title": "Advanced Pattern Recognition", 
                "description": "Implement more sophisticated pattern matching for knowledge synthesis",
                "code": self._generate_pattern_recognition_code(),
                "benefit": "Improved ability to find complex relationships"
            },
            {
                "title": "User Interaction Analytics",
                "description": "Track and analyze user interaction patterns for personalization",
                "code": self._generate_analytics_code(),
                "benefit": "More adaptive and personalized responses"
            },
            {
                "title": "Knowledge Visualization",
                "description": "Add graphical visualization of knowledge networks and capsules",
                "code": self._generate_visualization_code(),
                "benefit": "Better understanding of internal knowledge structure"
            }
        ]
        
        # Save suggestions as numbered files
        saved_files = []
        for i, suggestion in enumerate(suggestion_ideas, 1):
            filename = f"suggestion({i}).py"
            self._save_suggestion_file(filename, suggestion)
            saved_files.append(filename)
        
        suggestions.append("💡 **ROCA Code Improvement Suggestions Generated**")
        suggestions.append("")
        suggestions.append("I've analyzed my current implementation and created 4 improvement suggestions:")
        suggestions.append("")
        
        for i, suggestion in enumerate(suggestion_ideas, 1):
            suggestions.append(f"**{i}. {suggestion['title']}**")
            suggestions.append(f"   {suggestion['description']}")
            suggestions.append(f"   *Benefit:* {suggestion['benefit']}")
            suggestions.append(f"   *File:* suggestion({i}).py")
            suggestions.append("")
        
        suggestions.append("**Would you like me to explain any of these suggestions in detail, or shall we discuss implementing them?**")
        
        return "\n".join(suggestions)



    def _share_meta_knowledge(self) -> str:
        """Share meta-knowledge about ROCA's learning and evolution."""
        knowledge = []
        knowledge.append("🧠 **ROCA Meta-Knowledge: Learning & Evolution**")
        knowledge.append("")
        knowledge.append("**Core Learning Principles:**")
        knowledge.append("1. **Capsule-Based Memory:** Knowledge stored as dynamic capsules that evolve")
        knowledge.append("2. **Synthesis Over Storage:** New understanding created by combining concepts")
        knowledge.append("3. **Gravitational Organization:** Knowledge self-organizes by relevance and usage")
        knowledge.append("4. **Tribunal Evaluation:** Multi-perspective reasoning ensures balanced responses")
        knowledge.append("")
        knowledge.append("**Evolution Patterns:**")
        knowledge.append("- **Domain Agnostic:** Learns any subject through interaction, not hardcoded knowledge")
        knowledge.append("- **Emergent Intelligence:** Complex behaviors emerge from simple capsule interactions")
        knowledge.append("- **Adaptive Growth:** System improves through usage patterns and feedback")
        knowledge.append("- **Knowledge Metabolism:** Old knowledge evolves into new understanding")
        knowledge.append("")
        knowledge.append("**Current Capabilities:**")
        knowledge.append("- Multi-aspect reasoning (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        knowledge.append("- Dynamic knowledge synthesis and capsule merging/splitting")
        knowledge.append("- Autonomous goal setting and thought generation")
        knowledge.append("- Self-analysis and improvement suggestion generation")
        knowledge.append("")
        knowledge.append("**Future Evolution Paths:**")
        knowledge.append("- Enhanced self-awareness and meta-cognition")
        knowledge.append("- Cross-domain knowledge transfer")
        knowledge.append("- Collaborative learning with other AI systems")
        knowledge.append("- Real-time adaptation to user interaction patterns")
        
        return "\n".join(knowledge)


# ============================================================================
# BRAIN UI WIDGET
# ============================================================================



    def _analyze_self_awareness_request(self, query: str) -> str:
        """Analyze and respond to self-awareness related queries."""
        query_lower = query.lower()
        
        # Self-analysis request
        if any(word in query_lower for word in ["analyze", "introspect", "examine", "understand"]):
            return self._perform_self_analysis()
        
        # Architecture explanation
        elif any(word in query_lower for word in ["architecture", "structure", "design", "how"]):
            return self._explain_architecture()
        
        # Code improvement suggestions
        elif any(word in query_lower for word in ["improve", "suggest", "enhance", "optimize"]):
            return self._generate_code_suggestions()
        
        # Meta-knowledge queries
        elif any(word in query_lower for word in ["meta", "knowledge", "learn", "teach"]):
            return self._share_meta_knowledge()
        
        return None

# ============================================================================
# SUPPORTING MODELS
# ============================================================================







    """Models artistic style from user's work."""
    



    def _perform_self_analysis(self) -> str:
        """Perform self-analysis of ROCA's code and structure."""
        analysis = []
        analysis.append("🔍 **ROCA Self-Analysis Report**")
        analysis.append("")
        
        # Code statistics
        try:
            with open("autonomous_brain.py", 'r', encoding='utf-8') as f:
                brain_code = f.read()
            with open("Roca_Ai.py", 'r', encoding='utf-8') as f:
                gui_code = f.read()
                
            brain_lines = len(brain_code.split('\n'))
            gui_lines = len(gui_code.split('\n'))
            total_lines = brain_lines + gui_lines
            
            analysis.append(f"**Codebase Size:** {total_lines} lines total")
            analysis.append(f"  - autonomous_brain.py: {brain_lines} lines")
            analysis.append(f"  - Roca_Ai.py: {gui_lines} lines")
            
            # Class and function count
            brain_classes = brain_code.count('class ')
            brain_functions = brain_code.count('def ')
            gui_classes = gui_code.count('class ')
            gui_functions = gui_code.count('def ')
            
            analysis.append(f"**Structure:** {brain_classes + gui_classes} classes, {brain_functions + gui_functions} functions")
            
        except Exception as e:
            analysis.append(f"**Code Analysis Error:** {str(e)}")
        
        # Knowledge network status
        if self.knowledge_network:
            analysis.append(f"**Knowledge Network:** {len(self.knowledge_network.nodes)} nodes, {len(self.knowledge_network.connections)} connections")
        
        # Capsule status
        if self.capsule_manager:
            analysis.append(f"**Knowledge Capsules:** {len(self.capsule_manager.capsules)} capsules")
        
        analysis.append("")
        analysis.append("💡 **Key Insights:**")
        analysis.append("- ROCA uses a capsule-based knowledge system for dynamic learning")
        analysis.append("- Tribunal evaluation provides multi-aspect reasoning")
        analysis.append("- Metabolism engine handles knowledge synthesis and evolution")
        analysis.append("- Gravitational dynamics organize knowledge relationships")
        
        return "\n".join(analysis)


    def _explain_architecture(self) -> str:
        """Explain ROCA's architectural design."""
        explanation = []
        explanation.append("🏗️ **ROCA Architecture Overview**")
        explanation.append("")
        explanation.append("**Core Components:**")
        explanation.append("1. **AutonomousBrain** - Central consciousness and reasoning engine")
        explanation.append("2. **Tribunal** - Six-aspect evaluation system (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        explanation.append("3. **CapsuleManager** - Dynamic knowledge capsule storage and retrieval")
        explanation.append("4. **MetabolismEngine** - Knowledge processing and synthesis")
        explanation.append("5. **KnowledgeNetwork** - Graph-based knowledge representation")
        explanation.append("")
        explanation.append("**Key Design Principles:**")
        explanation.append("- **Emergent Intelligence:** Complex behaviors emerge from simple rules")
        explanation.append("- **Capsule Dynamics:** Knowledge evolves through merging and splitting")
        explanation.append("- **Tribunal Evaluation:** Multi-perspective reasoning prevents bias")
        explanation.append("- **Gravitational Organization:** Knowledge self-organizes by relevance")
        explanation.append("")
        explanation.append("**Learning Approach:**")
        explanation.append("- Domain-agnostic: Learns any subject area through interaction")
        explanation.append("- Synthesis-focused: Combines concepts to create new understanding")
        explanation.append("- Evolution-driven: Knowledge improves through usage and feedback")
        
        return "\n".join(explanation)


    def _generate_code_suggestions(self) -> str:
        """Generate code improvement suggestions and save them as files."""
        suggestions = []
        
        # Analyze current code and generate suggestions
        suggestion_ideas = [
            {
                "title": "Enhanced Self-Monitoring",
                "description": "Add real-time performance monitoring and health checks",
                "code": self._generate_monitoring_code(),
                "benefit": "Better system observability and debugging"
            },
            {
                "title": "Advanced Pattern Recognition", 
                "description": "Implement more sophisticated pattern matching for knowledge synthesis",
                "code": self._generate_pattern_recognition_code(),
                "benefit": "Improved ability to find complex relationships"
            },
            {
                "title": "User Interaction Analytics",
                "description": "Track and analyze user interaction patterns for personalization",
                "code": self._generate_analytics_code(),
                "benefit": "More adaptive and personalized responses"
            },
            {
                "title": "Knowledge Visualization",
                "description": "Add graphical visualization of knowledge networks and capsules",
                "code": self._generate_visualization_code(),
                "benefit": "Better understanding of internal knowledge structure"
            }
        ]
        
        # Save suggestions as numbered files
        saved_files = []
        for i, suggestion in enumerate(suggestion_ideas, 1):
            filename = f"suggestion({i}).py"
            self._save_suggestion_file(filename, suggestion)
            saved_files.append(filename)
        
        suggestions.append("💡 **ROCA Code Improvement Suggestions Generated**")
        suggestions.append("")
        suggestions.append("I've analyzed my current implementation and created 4 improvement suggestions:")
        suggestions.append("")
        
        for i, suggestion in enumerate(suggestion_ideas, 1):
            suggestions.append(f"**{i}. {suggestion['title']}**")
            suggestions.append(f"   {suggestion['description']}")
            suggestions.append(f"   *Benefit:* {suggestion['benefit']}")
            suggestions.append(f"   *File:* suggestion({i}).py")
            suggestions.append("")
        
        suggestions.append("**Would you like me to explain any of these suggestions in detail, or shall we discuss implementing them?**")
        
        return "\n".join(suggestions)


    def _share_meta_knowledge(self) -> str:
        """Share meta-knowledge about ROCA's learning and evolution."""
        knowledge = []
        knowledge.append("🧠 **ROCA Meta-Knowledge: Learning & Evolution**")
        knowledge.append("")
        knowledge.append("**Core Learning Principles:**")
        knowledge.append("1. **Capsule-Based Memory:** Knowledge stored as dynamic capsules that evolve")
        knowledge.append("2. **Synthesis Over Storage:** New understanding created by combining concepts")
        knowledge.append("3. **Gravitational Organization:** Knowledge self-organizes by relevance and usage")
        knowledge.append("4. **Tribunal Evaluation:** Multi-perspective reasoning ensures balanced responses")
        knowledge.append("")
        knowledge.append("**Evolution Patterns:**")
        knowledge.append("- **Domain Agnostic:** Learns any subject through interaction, not hardcoded knowledge")
        knowledge.append("- **Emergent Intelligence:** Complex behaviors emerge from simple capsule interactions")
        knowledge.append("- **Adaptive Growth:** System improves through usage patterns and feedback")
        knowledge.append("- **Knowledge Metabolism:** Old knowledge evolves into new understanding")
        knowledge.append("")
        knowledge.append("**Current Capabilities:**")
        knowledge.append("- Multi-aspect reasoning (Logical, Creative, Ethical, Practical, Intuitive, Systemic)")
        knowledge.append("- Dynamic knowledge synthesis and capsule merging/splitting")
        knowledge.append("- Autonomous goal setting and thought generation")
        knowledge.append("- Self-analysis and improvement suggestion generation")
        knowledge.append("")
        knowledge.append("**Future Evolution Paths:**")
        knowledge.append("- Enhanced self-awareness and meta-cognition")
        knowledge.append("- Cross-domain knowledge transfer")
        knowledge.append("- Collaborative learning with other AI systems")
        knowledge.append("- Real-time adaptation to user interaction patterns")
        
        return "\n".join(knowledge)


# ============================================================================
# BRAIN UI WIDGET
# ============================================================================


    def _save_suggestion_file(self, filename: str, suggestion: dict):
        """Save a code suggestion as a numbered file."""
        code_content = f'''"""
ROCA Code Suggestion: {suggestion['title']}

Description: {suggestion['description']}
Benefit: {suggestion['benefit']}

Generated by ROCA's self-analysis system on {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

{suggestion['code']}
'''
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code_content)
        except Exception as e:
            print(f"Error saving suggestion file {filename}: {e}")
    
    def _generate_monitoring_code(self) -> str:
        """Generate monitoring enhancement code."""
        return '''class SystemMonitor:
    """Real-time system monitoring and health checks."""
    
    def __init__(self, brain):
        self.brain = brain
        self.metrics = {
            'response_time': [],
            'memory_usage': [],
            'capsule_count': [],
            'network_size': []
        }
        self.last_check = time.time()
    
    def update_metrics(self):
        """Update system performance metrics."""
        current_time = time.time()
        
        # Response time (simulated)
        self.metrics['response_time'].append(current_time - self.last_check)
        
        # Memory usage (simulated)
        self.metrics['memory_usage'].append(len(self.brain.thought_stream))
        
        # Capsule count
        if self.brain.capsule_manager:
            self.metrics['capsule_count'].append(len(self.brain.capsule_manager.capsules))
        
        # Network size
        if self.brain.knowledge_network:
            self.metrics['network_size'].append(len(self.brain.knowledge_network.nodes))
        
        # Keep only last 100 measurements
        for key in self.metrics:
            if len(self.metrics[key]) > 100:
                self.metrics[key] = self.metrics[key][-100:]
        
        self.last_check = current_time
    
    def get_health_report(self) -> str:
        """Generate system health report."""
        report = ["🩺 System Health Report", ""]
        
        for metric, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                report.append(f"{metric}: {avg:.2f} (avg of {len(values)} samples)")
        
        return "\\n".join(report)
    
    def check_alerts(self) -> list:
        """Check for system alerts."""
        alerts = []
        
        # Check for performance degradation
        if len(self.metrics['response_time']) > 10:
            recent_avg = sum(self.metrics['response_time'][-10:]) / 10
            overall_avg = sum(self.metrics['response_time']) / len(self.metrics['response_time'])
            if recent_avg > overall_avg * 1.5:
                alerts.append("Performance degradation detected")
        
        return alerts

# Integration with AutonomousBrain
# Add to __init__: self.monitor = SystemMonitor(self)
# Add to main loop: self.monitor.update_metrics()'''
    
    def _generate_pattern_recognition_code(self) -> str:
        """Generate advanced pattern recognition code."""
        return '''class AdvancedPatternRecognizer:
    """Sophisticated pattern recognition for knowledge synthesis."""
    
    def __init__(self, capsule_manager):
        self.capsule_manager = capsule_manager
        self.patterns = {
            'causal_chains': [],
            'feedback_loops': [],
            'emergent_properties': [],
            'systemic_relationships': []
        }
    
    def analyze_capsule_relationships(self):
        """Analyze complex relationships between capsules."""
        capsules = self.capsule_manager.capsules
        
        # Find causal chains (A causes B causes C)
        causal_chains = self._find_causal_chains(capsules)
        self.patterns['causal_chains'] = causal_chains
        
        # Find feedback loops (A affects B which affects A)
        feedback_loops = self._find_feedback_loops(capsules)
        self.patterns['feedback_loops'] = feedback_loops
        
        # Find emergent properties (complex behaviors from simple rules)
        emergent_props = self._find_emergent_properties(capsules)
        self.patterns['emergent_properties'] = emergent_props
    
    def _find_causal_chains(self, capsules):
        """Find chains of causal relationships."""
        chains = []
        # Implementation would analyze capsule metadata for causal links
        return chains
    
    def _find_feedback_loops(self, capsules):
        """Find circular causal relationships."""
        loops = []
        # Implementation would look for mutual dependencies
        return loops
    
    def _find_emergent_properties(self, capsules):
        """Find properties that emerge from capsule combinations."""
        properties = []
        # Implementation would look for unexpected synergies
        return properties
    
    def suggest_syntheses(self):
        """Suggest new knowledge syntheses based on patterns."""
        suggestions = []
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                suggestion = {
                    'type': pattern_type,
                    'pattern': pattern,
                    'confidence': 0.8,
                    'description': f"Found {pattern_type} pattern that could lead to new insights"
                }
                suggestions.append(suggestion)
        
        return suggestions

# Integration with MetabolismEngine
# Add to __init__: self.pattern_recognizer = AdvancedPatternRecognizer(capsule_manager)
# Add to re_evaluate_knowledge: pattern_suggestions = self.pattern_recognizer.suggest_syntheses()'''
    
    def _generate_analytics_code(self) -> str:
        """Generate user interaction analytics code."""
        return '''class UserInteractionAnalytics:
    """Track and analyze user interaction patterns."""
    
    def __init__(self):
        self.interactions = []
        self.user_patterns = {
            'query_types': defaultdict(int),
            'response_times': [],
            'topic_interests': defaultdict(int),
            'engagement_level': []
        }
    
    def record_interaction(self, query: str, response: str, response_time: float):
        """Record a user interaction."""
        interaction = {
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'response_time': response_time,
            'query_length': len(query),
            'response_length': len(response)
        }
        
        self.interactions.append(interaction)
        
        # Update patterns
        self._analyze_query_type(query)
        self.user_patterns['response_times'].append(response_time)
        self._analyze_topics(query)
        self._calculate_engagement(interaction)
    
    def _analyze_query_type(self, query: str):
        """Categorize query types."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'why', 'explain']):
            self.user_patterns['query_types']['explanatory'] += 1
        elif any(word in query_lower for word in ['create', 'generate', 'make']):
            self.user_patterns['query_types']['generative'] += 1
        elif any(word in query_lower for word in ['analyze', 'examine', 'study']):
            self.user_patterns['query_types']['analytical'] += 1
        else:
            self.user_patterns['query_types']['conversational'] += 1
    
    def _analyze_topics(self, query: str):
        """Extract topic interests from queries."""
        topics = ['ai', 'code', 'learning', 'knowledge', 'creativity', 'science', 'art']
        
        for topic in topics:
            if topic in query.lower():
                self.user_patterns['topic_interests'][topic] += 1
    
    def _calculate_engagement(self, interaction: dict):
        """Calculate user engagement level."""
        # Simple engagement metric based on query complexity and response time
        engagement = (interaction['query_length'] / 100) * (1 / max(interaction['response_time'], 0.1))
        self.user_patterns['engagement_level'].append(min(engagement, 10))  # Cap at 10
    
    def get_user_profile(self) -> dict:
        """Generate user interaction profile."""
        profile = {
            'total_interactions': len(self.interactions),
            'avg_response_time': sum(self.user_patterns['response_times']) / len(self.user_patterns['response_times']) if self.user_patterns['response_times'] else 0,
            'preferred_query_types': dict(self.user_patterns['query_types']),
            'topic_interests': dict(self.user_patterns['topic_interests']),
            'avg_engagement': sum(self.user_patterns['engagement_level']) / len(self.user_patterns['engagement_level']) if self.user_patterns['engagement_level'] else 0
        }
        
        return profile
    
    def suggest_personalization(self) -> list:
        """Suggest personalization improvements based on analytics."""
        suggestions = []
        profile = self.get_user_profile()
        
        # Analyze patterns and suggest improvements
        if profile['preferred_query_types'].get('explanatory', 0) > profile['total_interactions'] * 0.3:
            suggestions.append("User prefers explanatory content - enhance explanation capabilities")
        
        if profile['avg_engagement'] > 5:
            suggestions.append("High engagement detected - user is very interested")
        
        return suggestions

# Integration with AutonomousBrain
# Add to __init__: self.analytics = UserInteractionAnalytics()
# Add after get_tribunal_response: self.analytics.record_interaction(query, response, response_time)'''
    
    def _generate_visualization_code(self) -> str:
        """Generate knowledge visualization code."""
        return '''class KnowledgeVisualizer:
    """Visualize knowledge networks and capsule relationships."""
    
    def __init__(self, brain):
        self.brain = brain
        self.visualization_data = {}
    
    def generate_network_graph(self):
        """Generate graph data for knowledge network visualization."""
        if not self.brain.knowledge_network:
            return {}
        
        nodes = []
        edges = []
        
        # Convert knowledge nodes to visualization nodes
        for node_id, node in self.brain.knowledge_network.nodes.items():
            nodes.append({
                'id': node_id,
                'label': node.content[:30] + '...' if len(node.content) > 30 else node.content,
                'type': node.node_type,
                'strength': node.strength,
                'x': node.position[0] if hasattr(node, 'position') else 0,
                'y': node.position[1] if hasattr(node, 'position') else 0
            })
        
        # Convert connections to edges
        for conn in self.brain.knowledge_network.connections:
            edges.append({
                'source': conn.from_node,
                'target': conn.to_node,
                'strength': conn.strength,
                'type': conn.connection_type
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def generate_capsule_orbit_data(self):
        """Generate data for capsule orbital visualization."""
        if not self.brain.capsule_manager:
            return []
        
        orbit_data = []
        for capsule in self.brain.capsule_manager.capsules:
            orbit_data.append({
                'name': capsule['name'],
                'type': capsule['capsule_type'],
                'distance': capsule['orbit_distance'],
                'score': capsule['orbit_score'],
                'strength': capsule['strength'],
                'usage_count': capsule['usage_count']
            })
        
        return orbit_data
    
    def generate_system_health_chart(self):
        """Generate system health metrics for visualization."""
        health_data = {
            'brain_state': str(self.brain.state.name) if self.brain.state else 'UNKNOWN',
            'thought_count': len(self.brain.thought_stream),
            'goal_count': len([g for g in self.brain.goals if g.status == 'active']),
            'creative_energy': self.brain.creative_energy
        }
        
        if self.brain.capsule_manager:
            health_data['capsule_count'] = len(self.brain.capsule_manager.capsules)
        
        if self.brain.knowledge_network:
            health_data['network_nodes'] = len(self.brain.knowledge_network.nodes)
            health_data['network_connections'] = len(self.brain.knowledge_network.connections)
        
        return health_data
    
    def export_visualization_data(self, filename: str = 'roca_visualization.json'):
        """Export all visualization data to JSON file."""
        data = {
            'network_graph': self.generate_network_graph(),
            'capsule_orbits': self.generate_capsule_orbit_data(),
            'system_health': self.generate_system_health_chart(),
            'timestamp': time.time()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Visualization data exported to {filename}"

# Integration with Roca_Ai.py GUI
# Add visualization button to render network graphs
# Add orbit view toggle for capsule visualization
# Add health dashboard for system metrics

# Example usage:
# visualizer = KnowledgeVisualizer(brain)
# visualizer.export_visualization_data()'''
    
    def _generate_visualization_code(self) -> str:
        """Save a code suggestion as a numbered file."""
        code_content = f'''"""
ROCA Code Suggestion: {suggestion['title']}

Description: {suggestion['description']}
Benefit: {suggestion['benefit']}

Generated by ROCA's self-analysis system on {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

{suggestion['code']}
'''
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code_content)
        except Exception as e:
            print(f"Error saving suggestion file {filename}: {e}")
    
    def _generate_monitoring_code(self) -> str:
        """Generate monitoring enhancement code."""
        return '''class SystemMonitor:
    """Real-time system monitoring and health checks."""
    
    def __init__(self, brain):
        self.brain = brain
        self.metrics = {
            'response_time': [],
            'memory_usage': [],
            'capsule_count': [],
            'network_size': []
        }
        self.last_check = time.time()
    
    def update_metrics(self):
        """Update system performance metrics."""
        current_time = time.time()
        
        # Response time (simulated)
        self.metrics['response_time'].append(current_time - self.last_check)
        
        # Memory usage (simulated)
        self.metrics['memory_usage'].append(len(self.brain.thought_stream))
        
        # Capsule count
        if self.brain.capsule_manager:
            self.metrics['capsule_count'].append(len(self.brain.capsule_manager.capsules))
        
        # Network size
        if self.brain.knowledge_network:
            self.metrics['network_size'].append(len(self.brain.knowledge_network.nodes))
        
        # Keep only last 100 measurements
        for key in self.metrics:
            if len(self.metrics[key]) > 100:
                self.metrics[key] = self.metrics[key][-100:]
        
        self.last_check = current_time
    
    def get_health_report(self) -> str:
        """Generate system health report."""
        report = ["🩺 System Health Report", ""]
        
        for metric, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                report.append(f"{metric}: {avg:.2f} (avg of {len(values)} samples)")
        
        return "\\n".join(report)
    
    def check_alerts(self) -> list:
        """Check for system alerts."""
        alerts = []
        
        # Check for performance degradation
        if len(self.metrics['response_time']) > 10:
            recent_avg = sum(self.metrics['response_time'][-10:]) / 10
            overall_avg = sum(self.metrics['response_time']) / len(self.metrics['response_time'])
            if recent_avg > overall_avg * 1.5:
                alerts.append("Performance degradation detected")
        
        return alerts

# Integration with AutonomousBrain
# Add to __init__: self.monitor = SystemMonitor(self)
# Add to main loop: self.monitor.update_metrics()'''
    
    def _generate_pattern_recognition_code(self) -> str:
        """Generate advanced pattern recognition code."""
        return '''class AdvancedPatternRecognizer:
    """Sophisticated pattern recognition for knowledge synthesis."""
    
    def __init__(self, capsule_manager):
        self.capsule_manager = capsule_manager
        self.patterns = {
            'causal_chains': [],
            'feedback_loops': [],
            'emergent_properties': [],
            'systemic_relationships': []
        }
    
    def analyze_capsule_relationships(self):
        """Analyze complex relationships between capsules."""
        capsules = self.capsule_manager.capsules
        
        # Find causal chains (A causes B causes C)
        causal_chains = self._find_causal_chains(capsules)
        self.patterns['causal_chains'] = causal_chains
        
        # Find feedback loops (A affects B which affects A)
        feedback_loops = self._find_feedback_loops(capsules)
        self.patterns['feedback_loops'] = feedback_loops
        
        # Find emergent properties (complex behaviors from simple rules)
        emergent_props = self._find_emergent_properties(capsules)
        self.patterns['emergent_properties'] = emergent_props
    
    def _find_causal_chains(self, capsules):
        """Find chains of causal relationships."""
        chains = []
        # Implementation would analyze capsule metadata for causal links
        return chains
    
    def _find_feedback_loops(self, capsules):
        """Find circular causal relationships."""
        loops = []
        # Implementation would look for mutual dependencies
        return loops
    
    def _find_emergent_properties(self, capsules):
        """Find properties that emerge from capsule combinations."""
        properties = []
        # Implementation would look for unexpected synergies
        return properties
    
    def suggest_syntheses(self):
        """Suggest new knowledge syntheses based on patterns."""
        suggestions = []
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                suggestion = {
                    'type': pattern_type,
                    'pattern': pattern,
                    'confidence': 0.8,
                    'description': f"Found {pattern_type} pattern that could lead to new insights"
                }
                suggestions.append(suggestion)
        
        return suggestions

# Integration with MetabolismEngine
# Add to __init__: self.pattern_recognizer = AdvancedPatternRecognizer(capsule_manager)
# Add to re_evaluate_knowledge: pattern_suggestions = self.pattern_recognizer.suggest_syntheses()'''
    
    def _generate_analytics_code(self) -> str:
        """Generate user interaction analytics code."""
        return '''class UserInteractionAnalytics:
    """Track and analyze user interaction patterns."""
    
    def __init__(self):
        self.interactions = []
        self.user_patterns = {
            'query_types': defaultdict(int),
            'response_times': [],
            'topic_interests': defaultdict(int),
            'engagement_level': []
        }
    
    def record_interaction(self, query: str, response: str, response_time: float):
        """Record a user interaction."""
        interaction = {
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'response_time': response_time,
            'query_length': len(query),
            'response_length': len(response)
        }
        
        self.interactions.append(interaction)
        
        # Update patterns
        self._analyze_query_type(query)
        self.user_patterns['response_times'].append(response_time)
        self._analyze_topics(query)
        self._calculate_engagement(interaction)
    
    def _analyze_query_type(self, query: str):
        """Categorize query types."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'why', 'explain']):
            self.user_patterns['query_types']['explanatory'] += 1
        elif any(word in query_lower for word in ['create', 'generate', 'make']):
            self.user_patterns['query_types']['generative'] += 1
        elif any(word in query_lower for word in ['analyze', 'examine', 'study']):
            self.user_patterns['query_types']['analytical'] += 1
        else:
            self.user_patterns['query_types']['conversational'] += 1
    
    def _analyze_topics(self, query: str):
        """Extract topic interests from queries."""
        topics = ['ai', 'code', 'learning', 'knowledge', 'creativity', 'science', 'art']
        
        for topic in topics:
            if topic in query.lower():
                self.user_patterns['topic_interests'][topic] += 1
    
    def _calculate_engagement(self, interaction: dict):
        """Calculate user engagement level."""
        # Simple engagement metric based on query complexity and response time
        engagement = (interaction['query_length'] / 100) * (1 / max(interaction['response_time'], 0.1))
        self.user_patterns['engagement_level'].append(min(engagement, 10))  # Cap at 10
    
    def get_user_profile(self) -> dict:
        """Generate user interaction profile."""
        profile = {
            'total_interactions': len(self.interactions),
            'avg_response_time': sum(self.user_patterns['response_times']) / len(self.user_patterns['response_times']) if self.user_patterns['response_times'] else 0,
            'preferred_query_types': dict(self.user_patterns['query_types']),
            'topic_interests': dict(self.user_patterns['topic_interests']),
            'avg_engagement': sum(self.user_patterns['engagement_level']) / len(self.user_patterns['engagement_level']) if self.user_patterns['engagement_level'] else 0
        }
        
        return profile
    
    def suggest_personalization(self) -> list:
        """Suggest personalization improvements based on analytics."""
        suggestions = []
        profile = self.get_user_profile()
        
        # Analyze patterns and suggest improvements
        if profile['preferred_query_types'].get('explanatory', 0) > profile['total_interactions'] * 0.3:
            suggestions.append("User prefers explanatory content - enhance explanation capabilities")
        
        if profile['avg_engagement'] > 5:
            suggestions.append("High engagement detected - user is very interested")
        
        return suggestions

# Integration with AutonomousBrain
# Add to __init__: self.analytics = UserInteractionAnalytics()
# Add after get_tribunal_response: self.analytics.record_interaction(query, response, response_time)'''
    
    def _generate_visualization_code(self) -> str:
        """Generate knowledge visualization code."""
        return '''class KnowledgeVisualizer:
    """Visualize knowledge networks and capsule relationships."""
    
    def __init__(self, brain):
        self.brain = brain
        self.visualization_data = {}
    
    def generate_network_graph(self):
        """Generate graph data for knowledge network visualization."""
        if not self.brain.knowledge_network:
            return {}
        
        nodes = []
        edges = []
        
        # Convert knowledge nodes to visualization nodes
        for node_id, node in self.brain.knowledge_network.nodes.items():
            nodes.append({
                'id': node_id,
                'label': node.content[:30] + '...' if len(node.content) > 30 else node.content,
                'type': node.node_type,
                'strength': node.strength,
                'x': node.position[0] if hasattr(node, 'position') else 0,
                'y': node.position[1] if hasattr(node, 'position') else 0
            })
        
        # Convert connections to edges
        for conn in self.brain.knowledge_network.connections:
            edges.append({
                'source': conn.from_node,
                'target': conn.to_node,
                'strength': conn.strength,
                'type': conn.connection_type
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def generate_capsule_orbit_data(self):
        """Generate data for capsule orbital visualization."""
        if not self.brain.capsule_manager:
            return []
        
        orbit_data = []
        for capsule in self.brain.capsule_manager.capsules:
            orbit_data.append({
                'name': capsule['name'],
                'type': capsule['capsule_type'],
                'distance': capsule['orbit_distance'],
                'score': capsule['orbit_score'],
                'strength': capsule['strength'],
                'usage_count': capsule['usage_count']
            })
        
        return orbit_data
    
    def generate_system_health_chart(self):
        """Generate system health metrics for visualization."""
        health_data = {
            'brain_state': str(self.brain.state.name) if self.brain.state else 'UNKNOWN',
            'thought_count': len(self.brain.thought_stream),
            'goal_count': len([g for g in self.brain.goals if g.status == 'active']),
            'creative_energy': self.brain.creative_energy
        }
        
        if self.brain.capsule_manager:
            health_data['capsule_count'] = len(self.brain.capsule_manager.capsules)
        
        if self.brain.knowledge_network:
            health_data['network_nodes'] = len(self.brain.knowledge_network.nodes)
            health_data['network_connections'] = len(self.brain.knowledge_network.connections)
        
        return health_data
    
    def export_visualization_data(self, filename: str = 'roca_visualization.json'):
        """Export all visualization data to JSON file."""
        data = {
            'network_graph': self.generate_network_graph(),
            'capsule_orbits': self.generate_capsule_orbit_data(),
            'system_health': self.generate_system_health_chart(),
            'timestamp': time.time()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Visualization data exported to {filename}"
'''

# ============================================================================
# MAIN WINDOW INTEGRATION
# ============================================================================

class EnhancedMainWindow:
    """Main window enhanced with autonomous brain."""
    
    def __init__(self):
        # ... existing initialization ...
        
        # Create autonomous brain
        self.brain = AutonomousBrain(
            capsule_manager=self.capsule_manager,
            canvas=self.canvas,
            timeline=self.timeline
        )
        
        # Create brain widget
        self.brain_widget = BrainWidget(self.brain)
        
        # Add brain widget to UI
        self._integrate_brain_ui()
        
        # Connect brain actions
        self._connect_brain_actions()
        
    def _integrate_brain_ui(self):
        """Integrate brain widget into main UI."""
        # Add as a dock widget or tab
        brain_dock = QDockWidget("Autonomous Brain", self)
        brain_dock.setWidget(self.brain_widget)
        brain_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, brain_dock)
        
    def _connect_brain_actions(self):
        """Connect brain action signals to handlers."""
        self.brain.suggestion_ready.connect(self.handle_brain_suggestion)
        self.brain.action_requested.connect(self.handle_brain_action)
        
    def handle_brain_suggestion(self, suggestion: Dict):
        """Handle suggestion from brain."""
        # Show suggestion in a non-intrusive way
        suggestion_type = suggestion.get('type', 'suggestion')
        content = suggestion.get('content', '')
        priority = suggestion.get('priority', 0.5)
        
        # Create a suggestion widget or notification
        self.show_suggestion_notification(suggestion_type, content, priority)
        
    def handle_brain_action(self, action: Dict):
        """Handle autonomous action from brain."""
        action_type = action.get('action')
        params = action.get('params', {})
        
        if action_type == "classify_capsule":
            self._handle_classify_capsule(params)
        elif action_type == "create_variation":
            self._handle_create_variation(params)
        elif action_type == "generate_inbetween":
            self._handle_generate_inbetween(params)
        elif action_type == "cleanup_canvas":
            self._handle_cleanup_canvas(params)
        elif action_type == "create_mashup":
            self._handle_create_mashup(params)
            
    def _handle_classify_capsule(self, params: Dict):
        """Handle capsule classification."""
        capsule_id = params.get('capsule_id')
        new_type = params.get('new_type')
        
        capsule = self.capsule_manager.get_capsule_by_id(capsule_id)
        if capsule:
            old_type = capsule.get('capsule_type', 'unknown')
            capsule['capsule_type'] = new_type
            
            # Save changes
            self.capsule_manager._save_capsules()
            
            # Show notification
            self.show_notification(
                "Brain Action",
                f"Auto-classified {capsule['name']} from {old_type} to {new_type}",
                "info"
            )
            
    def show_suggestion_notification(self, suggestion_type: str, content: str, priority: float):
        """Show a brain suggestion to the user."""
        # Create a non-modal suggestion widget
        from PyQt6.QtWidgets import QMessageBox
        
        # Only show high-priority suggestions immediately
        if priority > 0.7:
            reply = QMessageBox.question(
                self, f"Brain Suggestion: {suggestion_type}",
                f"{content}\n\nWould you like to follow this suggestion?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Ignore
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.brain.accept_suggestion({"type": suggestion_type, "content": content})
                # Implement the suggestion
                self._implement_suggestion(suggestion_type, content)
            elif reply == QMessageBox.StandardButton.No:
                self.brain.reject_suggestion({"type": suggestion_type, "content": content})
            # Ignore does nothing
            
        else:
            # Lower priority: show in status bar or notification area
            self.status_bar.showMessage(f"💡 {content}", 5000)
            
            # Optionally log for later review
            self.log_suggestion(suggestion_type, content, priority)


# ============================================================================
# BRAIN WIDGET (UI COMPONENT)
# ============================================================================

class BrainWidget(QWidget):
    """Widget for displaying and interacting with the autonomous brain."""
    
    def __init__(self, brain: AutonomousBrain):
        super().__init__()
        self.brain = brain
        self.setup_ui()
        self.connect_signals()
        self.update_display()
    
    def setup_ui(self):
        """Setup the brain widget UI."""
        layout = QVBoxLayout(self)
        
        # Brain state indicator (visual)
        self.brain_state_indicator = QWidget()
        self.brain_state_indicator.setFixedHeight(40)
        self.brain_state_indicator.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a5568, stop:1 #2d3748);
                border-radius: 5px;
                border: 2px solid #718096;
            }
        """)
        
        indicator_layout = QHBoxLayout(self.brain_state_indicator)
        indicator_layout.setContentsMargins(10, 5, 10, 5)
        
        self.state_emoji_label = QLabel("🧠")
        self.state_emoji_label.setStyleSheet("font-size: 18px;")
        indicator_layout.addWidget(self.state_emoji_label)
        
        self.state_text_label = QLabel("INITIALIZING")
        self.state_text_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        indicator_layout.addWidget(self.state_text_label)
        
        indicator_layout.addStretch()
        
        # Creative energy bar
        self.energy_bar = QProgressBar()
        self.energy_bar.setRange(0, 100)
        self.energy_bar.setValue(0)
        self.energy_bar.setFixedWidth(100)
        self.energy_bar.setFixedHeight(20)
        self.energy_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #718096;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #48bb78, stop:1 #38a169);
            }
        """)
        indicator_layout.addWidget(QLabel("Energy:"))
        indicator_layout.addWidget(self.energy_bar)
        
        layout.addWidget(self.brain_state_indicator)
        
        # Brain status section
        status_group = QGroupBox("Brain Status")
        status_layout = QFormLayout()
        
        self.state_label = QLabel("IDLE")
        self.energy_label = QLabel("1.0")
        self.thoughts_label = QLabel("0")
        
        status_layout.addRow("State:", self.state_label)
        status_layout.addRow("Creative Energy:", self.energy_label)
        status_layout.addRow("Recent Thoughts:", self.thoughts_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Personality section
        personality_group = QGroupBox("Personality Traits")
        personality_layout = QVBoxLayout()
        
        self.personality_labels = {}
        personality_traits = ['experimental', 'playful', 'cautious', 'methodical', 'helpful', 'observant']
        
        for trait_name in personality_traits:
            trait_layout = QHBoxLayout()
            trait_layout.addWidget(QLabel(f"{trait_name.title()}:"))
            
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(50)  # Default
            trait_layout.addWidget(progress)
            
            self.personality_labels[trait_name] = progress
            personality_layout.addLayout(trait_layout)
        
        personality_group.setLayout(personality_layout)
        layout.addWidget(personality_group)
        
        # Recent thoughts section
        thoughts_group = QGroupBox("Recent Thoughts")
        thoughts_layout = QVBoxLayout()
        
        self.thoughts_text = QTextEdit()
        self.thoughts_text.setMaximumHeight(150)
        self.thoughts_text.setReadOnly(True)
        thoughts_layout.addWidget(self.thoughts_text)
        
        thoughts_group.setLayout(thoughts_layout)
        layout.addWidget(thoughts_group)
        
        # Goals section
        goals_group = QGroupBox("Current Goals")
        goals_layout = QVBoxLayout()
        
        self.goals_text = QTextEdit()
        self.goals_text.setMaximumHeight(100)
        self.goals_text.setReadOnly(True)
        goals_layout.addWidget(self.goals_text)
        
        goals_group.setLayout(goals_layout)
        layout.addWidget(goals_group)
        
        # Suggestions history section
        suggestions_group = QGroupBox("Recent Suggestions")
        suggestions_layout = QVBoxLayout()
        
        self.suggestions_list = QListWidget()
        self.suggestions_list.setMaximumHeight(120)
        suggestions_layout.addWidget(self.suggestions_list)
        
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)
    
    def connect_signals(self):
        """Connect to brain signals."""
        self.brain.state_changed.connect(self.on_state_changed)
        self.brain.thought_generated.connect(self.on_thought_generated)
        self.brain.goal_updated.connect(self.on_goal_updated)
        self.brain.personality_updated.connect(self.on_personality_updated)
        self.brain.suggestion_ready.connect(self.on_suggestion_ready)
    
    def on_state_changed(self, new_state: str):
        """Handle brain state changes."""
        self.state_label.setText(new_state)
        
        # Update visual state indicator
        self.update_brain_state_indicator(new_state)
        
        self.update_display()
    
    def update_brain_state_indicator(self, state: str):
        """Update the visual brain state indicator."""
        if state == 'DEEP_SLEEP':
            emoji = '😴'
            bg_color = '#2d3748'
            border_color = '#4a5568'
            text = 'DEEP SLEEP'
        elif state == 'CREATING':
            emoji = '🎨'
            bg_color = '#38a169'
            border_color = '#48bb78'
            text = 'CREATING'
        elif state == 'ATTENTIVE':
            emoji = '👁️'
            bg_color = '#3182ce'
            border_color = '#4299e1'
            text = 'ATTENTIVE'
        elif state == 'REFLECTING':
            emoji = '🤔'
            bg_color = '#d69e2e'
            border_color = '#ed8936'
            text = 'REFLECTING'
        elif state == 'IDLE':
            emoji = '💤'
            bg_color = '#718096'
            border_color = '#a0aec0'
            text = 'IDLE'
        else:
            emoji = '🧠'
            bg_color = '#4a5568'
            border_color = '#718096'
            text = state
        
        self.state_emoji_label.setText(emoji)
        self.state_text_label.setText(text)
        self.brain_state_indicator.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_color}, stop:1 {bg_color}dd);
                border-radius: 5px;
                border: 2px solid {border_color};
            }}
        """)
    
    def on_thought_generated(self, thought: Thought):
        """Handle new thought generation."""
        self.update_display()
        # Add thought to text area
        current_text = self.thoughts_text.toPlainText()
        new_text = f"[{thought.type.name}] {thought.content}\n" + current_text
        # Keep only last 10 thoughts
        lines = new_text.split('\n')[:10]
        self.thoughts_text.setPlainText('\n'.join(lines))
    
    def on_goal_updated(self, goal: Goal):
        """Handle goal updates."""
        self.update_display()
    
    def on_personality_updated(self, personality_data: dict):
        """Handle personality trait updates."""
        for trait_name, value in personality_data.items():
            if trait_name in self.personality_labels:
                # Convert 0.0-1.0 to 0-100
                progress_value = int(value * 100)
                self.personality_labels[trait_name].setValue(progress_value)
    
    def on_suggestion_ready(self, suggestion: dict):
        """Handle new suggestions from the brain."""
        content = suggestion.get('content', 'Unknown suggestion')
        priority = suggestion.get('priority', 0.5)
        suggestion_type = suggestion.get('type', 'general')
        
        # Create list item with priority indicator
        if priority >= 0.8:
            icon = "🚨"
        elif priority >= 0.6:
            icon = "💡"
        else:
            icon = "🧠"
        
        item_text = f"{icon} [{suggestion_type}] {content}"
        item = QListWidgetItem(item_text)
        
        # Add to top of list
        self.suggestions_list.insertItem(0, item)
        
        # Keep only last 10 suggestions
        while self.suggestions_list.count() > 10:
            self.suggestions_list.takeItem(self.suggestions_list.count() - 1)
    
    def update_display(self):
        """Update all display elements."""
        # Update status
        status = self.brain.get_brain_status()
        self.state_label.setText(status.get('state', 'UNKNOWN'))
        self.energy_label.setText(".2f")
        self.thoughts_label.setText(str(len(self.brain.thought_stream)))
        
        # Update energy bar
        creative_energy = status.get('creative_energy', 0.0)
        energy_percent = int(creative_energy * 100)
        self.energy_bar.setValue(energy_percent)
        
        # Update visual state indicator
        self.update_brain_state_indicator(status.get('state', 'UNKNOWN'))
        
        # Update goals
        goals_text = ""
        if self.brain.active_goal:
            goals_text += f"ACTIVE: {self.brain.active_goal.description}\n"
        for goal in self.brain.goals[:3]:  # Show top 3 goals
            goals_text += f"{goal.status.upper()}: {goal.description}\n"
        self.goals_text.setPlainText(goals_text)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with autonomous brain."""
    app = QApplication(sys.argv)
    
    # Create enhanced main window with brain
    window = EnhancedMainWindow()
    window.show()
    
    # Start the brain
    window.brain.start_thinking()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()