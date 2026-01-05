"""
User Emulation System for Roca3D
Provides style mirroring and personality adaptation capabilities.
"""

import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import re
from collections import defaultdict, Counter

# PyQt6 imports
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox, QGroupBox, QProgressBar, QSlider, QListWidget, QListWidgetItem, QCheckBox
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, Qt

class PersonalityTrait(Enum):
    """Personality traits that can be learned and emulated"""
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    METHODICAL = "methodical"
    INTUITIVE = "intuitive"
    COLLABORATIVE = "collaborative"
    INDEPENDENT = "independent"
    DETAIL_ORIENTED = "detail_oriented"
    BIG_PICTURE = "big_picture"
    PATIENT = "patient"
    URGENT = "urgent"

class CommunicationStyle(Enum):
    """Communication styles that can be adapted"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    DIRECT = "direct"
    DIPLOMATIC = "diplomatic"
    VERBOSE = "verbose"
    CONCISE = "concise"

class WorkPattern(Enum):
    """Work patterns that can be learned"""
    MORNING_PERSON = "morning_person"
    NIGHT_OWL = "night_owl"
    BURST_WORKER = "burst_worker"
    STEADY_WORKER = "steady_worker"
    MULTITASKER = "multitasker"
    FOCUSED_WORKER = "focused_worker"
    COLLABORATIVE = "collaborative_worker"
    SOLO_WORKER = "solo_worker"

@dataclass
class UserAction:
    """Represents a user action that can be learned from"""
    action_type: str
    timestamp: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    success: bool = True

@dataclass
class StylePattern:
    """A learned style pattern"""
    pattern_type: str
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    last_observed: float = field(default_factory=time.time)

@dataclass
class UserProfile:
    """Complete user profile for emulation"""
    user_id: str = "default_user"
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    communication_style: Dict[CommunicationStyle, float] = field(default_factory=dict)
    work_patterns: Dict[WorkPattern, float] = field(default_factory=dict)
    style_patterns: Dict[str, StylePattern] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    action_history: List[UserAction] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def get_top_traits(self, trait_type: type, top_n: int = 3) -> List[Tuple[Any, float]]:
        """Get top N traits of a specific type"""
        if trait_type == PersonalityTrait:
            traits = self.personality_traits
        elif trait_type == CommunicationStyle:
            traits = self.communication_style
        elif trait_type == WorkPattern:
            traits = self.work_patterns
        else:
            return []

        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        return sorted_traits[:top_n]

class BehaviorAnalyzer(QThread):
    """Background thread for analyzing user behavior patterns"""

    analysis_complete = pyqtSignal(dict)  # analysis results

    def __init__(self, action_history: List[UserAction]):
        super().__init__()
        self.action_history = action_history
        self.is_cancelled = False

    def run(self):
        """Analyze user behavior patterns"""
        if self.is_cancelled:
            return

        analysis = {}

        try:
            # Analyze work patterns
            analysis['work_patterns'] = self._analyze_work_patterns()

            # Analyze communication style
            analysis['communication_style'] = self._analyze_communication_style()

            # Analyze personality traits
            analysis['personality_traits'] = self._analyze_personality_traits()

            # Analyze style patterns
            analysis['style_patterns'] = self._analyze_style_patterns()

            if not self.is_cancelled:
                self.analysis_complete.emit(analysis)

        except Exception as e:
            print(f"Behavior analysis error: {e}")

    def cancel(self):
        """Cancel analysis"""
        self.is_cancelled = True

    def _analyze_work_patterns(self) -> Dict[WorkPattern, float]:
        """Analyze work patterns from action history"""
        patterns = defaultdict(float)

        if not self.action_history:
            return dict(patterns)

        # Analyze timing patterns
        timestamps = [action.timestamp for action in self.action_history]
        hours = [datetime.fromtimestamp(ts).hour for ts in timestamps]

        morning_hours = sum(1 for h in hours if 6 <= h <= 12)
        evening_hours = sum(1 for h in hours if 18 <= h <= 24)

        if morning_hours > evening_hours * 1.5:
            patterns[WorkPattern.MORNING_PERSON] = 0.8
        elif evening_hours > morning_hours * 1.5:
            patterns[WorkPattern.NIGHT_OWL] = 0.8

        # Analyze session patterns
        sessions = self._group_into_sessions()
        session_lengths = [len(session) for session in sessions]

        if session_lengths:
            avg_session = np.mean(session_lengths)
            if avg_session > 20:  # Long focused sessions
                patterns[WorkPattern.FOCUSED_WORKER] = 0.7
            elif avg_session < 5:  # Short burst sessions
                patterns[WorkPattern.BURST_WORKER] = 0.7
            else:
                patterns[WorkPattern.STEADY_WORKER] = 0.6

        # Normalize patterns
        total = sum(patterns.values())
        if total > 0:
            for pattern in patterns:
                patterns[pattern] /= total

        return dict(patterns)

    def _analyze_communication_style(self) -> Dict[CommunicationStyle, float]:
        """Analyze communication style from actions"""
        styles = defaultdict(float)

        # This would analyze text inputs, command usage, etc.
        # For now, use simple heuristics based on action types

        action_types = [action.action_type for action in self.action_history]

        if 'voice_command' in action_types:
            styles[CommunicationStyle.CONVERSATIONAL] += 0.6
            styles[CommunicationStyle.CASUAL] += 0.4

        if 'complex_command' in action_types:
            styles[CommunicationStyle.TECHNICAL] += 0.7
            styles[CommunicationStyle.CONCISE] += 0.5

        if 'help_request' in action_types:
            styles[CommunicationStyle.DIPLOMATIC] += 0.5

        # Normalize
        total = sum(styles.values())
        if total > 0:
            for style in styles:
                styles[style] /= total

        return dict(styles)

    def _analyze_personality_traits(self) -> Dict[PersonalityTrait, float]:
        """Analyze personality traits from behavior"""
        traits = defaultdict(float)

        # Analyze creativity indicators
        creative_actions = ['create_capsule', 'experiment', 'customize', 'design']
        creative_count = sum(1 for action in self.action_history
                           if action.action_type in creative_actions)

        if creative_count > len(self.action_history) * 0.3:
            traits[PersonalityTrait.CREATIVE] = 0.8

        # Analyze analytical indicators
        analytical_actions = ['analyze', 'debug', 'optimize', 'measure']
        analytical_count = sum(1 for action in self.action_history
                             if action.action_type in analytical_actions)

        if analytical_count > len(self.action_history) * 0.4:
            traits[PersonalityTrait.ANALYTICAL] = 0.7

        # Analyze methodical indicators
        if self._has_methodical_pattern():
            traits[PersonalityTrait.METHODICAL] = 0.6

        # Analyze detail orientation
        detail_actions = ['fine_tune', 'adjust', 'refine', 'polish']
        detail_count = sum(1 for action in self.action_history
                          if action.action_type in detail_actions)

        if detail_count > len(self.action_history) * 0.25:
            traits[PersonalityTrait.DETAIL_ORIENTED] = 0.7

        # Normalize
        total = sum(traits.values())
        if total > 0:
            for trait in traits:
                traits[trait] /= total

        return dict(traits)

    def _analyze_style_patterns(self) -> Dict[str, StylePattern]:
        """Analyze recurring style patterns"""
        patterns = {}

        # Group actions by type
        action_groups = defaultdict(list)
        for action in self.action_history:
            action_groups[action.action_type].append(action)

        # Find patterns in each action type
        for action_type, actions in action_groups.items():
            if len(actions) < 3:  # Need at least 3 occurrences
                continue

            # Analyze parameter patterns
            param_patterns = self._find_parameter_patterns(actions)

            for pattern_name, pattern_data in param_patterns.items():
                pattern_key = f"{action_type}_{pattern_name}"
                patterns[pattern_key] = StylePattern(
                    pattern_type=pattern_name,
                    frequency=len(pattern_data['occurrences']),
                    confidence=pattern_data['confidence'],
                    examples=pattern_data['examples'][:5],  # Keep top 5 examples
                    last_observed=max(a.timestamp for a in actions)
                )

        return patterns

    def _group_into_sessions(self) -> List[List[UserAction]]:
        """Group actions into work sessions"""
        if not self.action_history:
            return []

        sessions = []
        current_session = []
        last_timestamp = None

        for action in sorted(self.action_history, key=lambda x: x.timestamp):
            if last_timestamp is None or action.timestamp - last_timestamp > 1800:  # 30 min gap
                if current_session:
                    sessions.append(current_session)
                current_session = [action]
            else:
                current_session.append(action)

            last_timestamp = action.timestamp

        if current_session:
            sessions.append(current_session)

        return sessions

    def _has_methodical_pattern(self) -> bool:
        """Check if user shows methodical behavior patterns"""
        # Look for sequential, planned actions
        action_sequence = [action.action_type for action in self.action_history[-20:]]  # Last 20 actions

        # Check for planning patterns
        planning_indicators = ['plan', 'organize', 'structure', 'prepare']
        planning_count = sum(1 for action in action_sequence if action in planning_indicators)

        return planning_count >= 3

    def _find_parameter_patterns(self, actions: List[UserAction]) -> Dict[str, Dict]:
        """Find recurring patterns in action parameters"""
        patterns = {}

        # Collect all parameter values
        param_values = defaultdict(list)
        for action in actions:
            for param_name, param_value in action.parameters.items():
                param_values[param_name].append(param_value)

        # Find frequent values for each parameter
        for param_name, values in param_values.items():
            if len(values) < 3:
                continue

            # Count frequency of each value
            value_counts = Counter(values)
            total = len(values)

            # Find values that appear frequently
            for value, count in value_counts.items():
                frequency = count / total
                if frequency >= 0.6:  # Appears in 60%+ of actions
                    pattern_name = f"{param_name}_{str(value)[:20]}"  # Truncate long values
                    patterns[pattern_name] = {
                        'occurrences': count,
                        'confidence': frequency,
                        'examples': [{'value': value, 'timestamp': actions[i].timestamp}
                                   for i in range(len(actions))
                                   if actions[i].parameters.get(param_name) == value][:3]
                    }

        return patterns

class UserEmulator(QObject):
    """Main user emulation system"""

    profile_updated = pyqtSignal(UserProfile)
    adaptation_suggested = pyqtSignal(str, dict)  # suggestion_type, parameters

    def __init__(self, profile_path: str = "user_profile.json"):
        super().__init__()
        self.profile_path = profile_path
        self.current_profile = self._load_profile()
        self.action_history = self.current_profile.action_history
        self.analyzers = []

    def _load_profile(self) -> UserProfile:
        """Load user profile from file"""
        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)

            # Convert string keys back to enums
            profile = UserProfile(
                user_id=data.get('user_id', 'default_user'),
                personality_traits={PersonalityTrait(k): v for k, v in data.get('personality_traits', {}).items()},
                communication_style={CommunicationStyle(k): v for k, v in data.get('communication_style', {}).items()},
                work_patterns={WorkPattern(k): v for k, v in data.get('work_patterns', {}).items()},
                preferences=data.get('preferences', {}),
                created_at=data.get('created_at', time.time()),
                updated_at=data.get('updated_at', time.time())
            )

            # Load action history
            for action_data in data.get('action_history', []):
                action = UserAction(
                    action_type=action_data['action_type'],
                    timestamp=action_data['timestamp'],
                    parameters=action_data.get('parameters', {}),
                    context=action_data.get('context', {}),
                    duration=action_data.get('duration'),
                    success=action_data.get('success', True)
                )
                profile.action_history.append(action)

            # Load style patterns
            for pattern_name, pattern_data in data.get('style_patterns', {}).items():
                pattern = StylePattern(
                    pattern_type=pattern_data['pattern_type'],
                    frequency=pattern_data['frequency'],
                    confidence=pattern_data['confidence'],
                    examples=pattern_data.get('examples', []),
                    last_observed=pattern_data.get('last_observed', time.time())
                )
                profile.style_patterns[pattern_name] = pattern

            return profile

        except (FileNotFoundError, json.JSONDecodeError):
            return UserProfile()

    def _save_profile(self):
        """Save user profile to file"""
        data = {
            'user_id': self.current_profile.user_id,
            'personality_traits': {k.value: v for k, v in self.current_profile.personality_traits.items()},
            'communication_style': {k.value: v for k, v in self.current_profile.communication_style.items()},
            'work_patterns': {k.value: v for k, v in self.current_profile.work_patterns.items()},
            'style_patterns': {
                name: {
                    'pattern_type': p.pattern_type,
                    'frequency': p.frequency,
                    'confidence': p.confidence,
                    'examples': p.examples,
                    'last_observed': p.last_observed
                } for name, p in self.current_profile.style_patterns.items()
            },
            'preferences': self.current_profile.preferences,
            'action_history': [
                {
                    'action_type': a.action_type,
                    'timestamp': a.timestamp,
                    'parameters': a.parameters,
                    'context': a.context,
                    'duration': a.duration,
                    'success': a.success
                } for a in self.current_profile.action_history
            ],
            'created_at': self.current_profile.created_at,
            'updated_at': self.current_profile.updated_at
        }

        with open(self.profile_path, 'w') as f:
            json.dump(data, f, indent=2)

    def record_action(self, action_type: str, parameters: Dict[str, Any] = None,
                     context: Dict[str, Any] = None, duration: float = None, success: bool = True):
        """Record a user action for learning"""
        action = UserAction(
            action_type=action_type,
            timestamp=time.time(),
            parameters=parameters or {},
            context=context or {},
            duration=duration,
            success=success
        )

        self.action_history.append(action)

        # Keep only recent history (last 1000 actions)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]

        self.current_profile.updated_at = time.time()
        self._save_profile()

        # Trigger adaptation suggestions occasionally
        if random.random() < 0.1:  # 10% chance
            self._generate_adaptation_suggestion()

    def analyze_behavior(self):
        """Start behavior analysis"""
        analyzer = BehaviorAnalyzer(self.action_history.copy())
        analyzer.analysis_complete.connect(self._on_analysis_complete)

        self.analyzers.append(analyzer)
        analyzer.start()

    def _on_analysis_complete(self, analysis: dict):
        """Handle completed behavior analysis"""
        # Update profile with analysis results
        self.current_profile.personality_traits.update(analysis.get('personality_traits', {}))
        self.current_profile.communication_style.update(analysis.get('communication_style', {}))
        self.current_profile.work_patterns.update(analysis.get('work_patterns', {}))
        self.current_profile.style_patterns.update(analysis.get('style_patterns', {}))

        self.current_profile.updated_at = time.time()
        self._save_profile()

        self.profile_updated.emit(self.current_profile)

        # Clean up analyzer
        for analyzer in self.analyzers:
            if analyzer.isFinished():
                self.analyzers.remove(analyzer)

    def get_adaptation_suggestions(self, context: str = None) -> List[Dict[str, Any]]:
        """Get adaptation suggestions based on user profile"""
        suggestions = []

        # Time-based suggestions
        current_hour = datetime.now().hour
        top_work_patterns = self.current_profile.get_top_traits(WorkPattern, 2)

        for pattern, strength in top_work_patterns:
            if pattern == WorkPattern.MORNING_PERSON and current_hour >= 18:
                suggestions.append({
                    'type': 'work_pattern',
                    'message': 'Consider wrapping up for the day - you seem most productive in the mornings',
                    'strength': strength
                })
            elif pattern == WorkPattern.NIGHT_OWL and current_hour < 6:
                suggestions.append({
                    'type': 'work_pattern',
                    'message': 'Early morning work detected - you might prefer evening sessions',
                    'strength': strength
                })

        # Style-based suggestions
        top_communication_styles = self.current_profile.get_top_traits(CommunicationStyle, 1)

        for style, strength in top_communication_styles:
            if style == CommunicationStyle.TECHNICAL and context == 'creative_task':
                suggestions.append({
                    'type': 'communication_style',
                    'message': 'Switching to more creative communication mode',
                    'strength': strength
                })

        return suggestions

    def adapt_response(self, base_response: str, context: Dict[str, Any] = None) -> str:
        """Adapt a response based on user profile"""
        adapted_response = base_response

        # Adapt based on communication style
        top_styles = self.current_profile.get_top_traits(CommunicationStyle, 2)

        for style, strength in top_styles:
            if style == CommunicationStyle.CASUAL and strength > 0.6:
                # Make more casual
                adapted_response = adapted_response.replace("Please ", "").replace("You should ", "Try ")
            elif style == CommunicationStyle.TECHNICAL and strength > 0.6:
                # Add technical details
                if "capsule" in adapted_response.lower():
                    adapted_response += " (utilizing dynamic routing algorithms)"
            elif style == CommunicationStyle.CONCISE and strength > 0.6:
                # Make more concise
                sentences = adapted_response.split('. ')
                if len(sentences) > 2:
                    adapted_response = '. '.join(sentences[:2]) + '.'

        return adapted_response

    def _generate_adaptation_suggestion(self):
        """Generate and emit an adaptation suggestion"""
        suggestions = self.get_adaptation_suggestions()
        if suggestions:
            suggestion = random.choice(suggestions)
            self.adaptation_suggested.emit(suggestion['type'], suggestion)

class UserEmulationWidget(QWidget):
    """GUI widget for user emulation system"""

    def __init__(self, emulator: UserEmulator, parent=None):
        super().__init__(parent)
        self.emulator = emulator
        self.init_ui()

        # Connect signals
        self.emulator.profile_updated.connect(self._on_profile_updated)
        self.emulator.adaptation_suggested.connect(self._on_adaptation_suggested)

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("User Emulation - Style Mirroring & Personality Adaptation")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Control buttons
        button_layout = QHBoxLayout()

        self.analyze_button = QPushButton("Analyze Behavior")
        self.analyze_button.clicked.connect(self._analyze_behavior)
        button_layout.addWidget(self.analyze_button)

        self.suggestions_button = QPushButton("Get Suggestions")
        self.suggestions_button.clicked.connect(self._show_suggestions)
        button_layout.addWidget(self.suggestions_button)

        layout.addLayout(button_layout)

        # Profile display
        profile_group = QGroupBox("Current User Profile")
        profile_layout = QVBoxLayout()

        # Personality traits
        traits_layout = QHBoxLayout()
        traits_layout.addWidget(QLabel("Top Personality Traits:"))
        self.traits_label = QLabel("Not analyzed yet")
        traits_layout.addWidget(self.traits_label)
        profile_layout.addLayout(traits_layout)

        # Communication style
        comm_layout = QHBoxLayout()
        comm_layout.addWidget(QLabel("Communication Style:"))
        self.comm_label = QLabel("Not analyzed yet")
        comm_layout.addWidget(self.comm_label)
        profile_layout.addLayout(comm_layout)

        # Work patterns
        work_layout = QHBoxLayout()
        work_layout.addWidget(QLabel("Work Patterns:"))
        self.work_label = QLabel("Not analyzed yet")
        work_layout.addWidget(self.work_label)
        profile_layout.addLayout(work_layout)

        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)

        # Adaptation settings
        adaptation_group = QGroupBox("Adaptation Settings")
        adaptation_layout = QVBoxLayout()

        self.adaptation_enabled = QCheckBox("Enable automatic adaptation")
        self.adaptation_enabled.setChecked(True)
        adaptation_layout.addWidget(self.adaptation_enabled)

        adapt_freq_layout = QHBoxLayout()
        adapt_freq_layout.addWidget(QLabel("Adaptation Frequency:"))
        self.adapt_freq_slider = QSlider()
        self.adapt_freq_slider.setOrientation(Qt.Orientation.Horizontal)  # Horizontal
        self.adapt_freq_slider.setRange(1, 10)
        self.adapt_freq_slider.setValue(5)
        adapt_freq_layout.addWidget(self.adapt_freq_slider)
        adaptation_layout.addLayout(adapt_freq_layout)

        adaptation_group.setLayout(adaptation_layout)
        layout.addWidget(adaptation_group)

        # Suggestions display
        suggestions_group = QGroupBox("Adaptation Suggestions")
        suggestions_layout = QVBoxLayout()
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setMaximumHeight(100)
        self.suggestions_text.setPlaceholderText("Suggestions will appear here...")
        suggestions_layout.addWidget(self.suggestions_text)
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)

        # Action recording
        record_group = QGroupBox("Record User Action (for testing)")
        record_layout = QVBoxLayout()

        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Action Type:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems([
            'create_capsule', 'delete_selection', 'analyze', 'experiment',
            'fine_tune', 'help_request', 'voice_command', 'complex_command'
        ])
        action_layout.addWidget(self.action_combo)

        self.record_button = QPushButton("Record Action")
        self.record_button.clicked.connect(self._record_test_action)
        action_layout.addWidget(self.record_button)

        record_layout.addLayout(action_layout)
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _analyze_behavior(self):
        """Start behavior analysis"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Analyzing user behavior...")

        self.emulator.analyze_behavior()

    def _on_profile_updated(self, profile: UserProfile):
        """Handle profile updates"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Profile updated")

        # Update trait displays
        top_traits = profile.get_top_traits(PersonalityTrait, 3)
        if top_traits:
            trait_text = ", ".join(f"{trait.value} ({strength:.2f})" for trait, strength in top_traits)
            self.traits_label.setText(trait_text)
        else:
            self.traits_label.setText("No traits analyzed yet")

        top_comm = profile.get_top_traits(CommunicationStyle, 2)
        if top_comm:
            comm_text = ", ".join(f"{style.value} ({strength:.2f})" for style, strength in top_comm)
            self.comm_label.setText(comm_text)
        else:
            self.comm_label.setText("No communication style analyzed yet")

        top_work = profile.get_top_traits(WorkPattern, 2)
        if top_work:
            work_text = ", ".join(f"{pattern.value} ({strength:.2f})" for pattern, strength in top_work)
            self.work_label.setText(work_text)
        else:
            self.work_label.setText("No work patterns analyzed yet")

    def _show_suggestions(self):
        """Show adaptation suggestions"""
        suggestions = self.emulator.get_adaptation_suggestions()

        if not suggestions:
            self.suggestions_text.setPlainText("No suggestions available at this time.")
            return

        suggestion_text = ""
        for i, suggestion in enumerate(suggestions, 1):
            suggestion_text += f"{i}. [{suggestion['type']}] {suggestion['message']}\n"
            suggestion_text += f"   Confidence: {suggestion.get('strength', 0):.2f}\n\n"

        self.suggestions_text.setPlainText(suggestion_text)

    def _on_adaptation_suggested(self, suggestion_type: str, parameters: dict):
        """Handle adaptation suggestions"""
        if self.adaptation_enabled.isChecked():
            message = parameters.get('message', 'Adaptation suggested')
            current_text = self.suggestions_text.toPlainText()
            new_text = f"[AUTO] {message}\n{current_text}"
            self.suggestions_text.setPlainText(new_text[:500])  # Limit length

    def _record_test_action(self):
        """Record a test action for demonstration"""
        action_type = self.action_combo.currentText()

        # Generate some test parameters
        test_params = {}
        if action_type == 'create_capsule':
            test_params = {'capsule_type': 'character', 'complexity': 'medium'}
        elif action_type == 'analyze':
            test_params = {'analysis_type': 'performance', 'target': 'scene'}

        self.emulator.record_action(action_type, test_params)
        self.status_label.setText(f"Recorded action: {action_type}")

# Export main classes
__all__ = ['UserEmulator', 'UserEmulationWidget', 'UserProfile', 'UserAction', 'PersonalityTrait', 'CommunicationStyle', 'WorkPattern']