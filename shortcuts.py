#!/usr/bin/env python3
"""
Keyboard Shortcuts System for PyQt6 JPatch
"""

from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt

class ShortcutManager:
    """Manages keyboard shortcuts for the application"""

    def __init__(self, settings):
        self.settings = settings
        self.shortcuts = {}
        self.load_shortcuts()

    def load_shortcuts(self):
        """Load shortcuts from settings"""
        shortcuts_data = self.settings.get('shortcuts', {})
        default_shortcuts = self.settings.default_settings['shortcuts']

        # Merge with defaults
        for action, shortcut in default_shortcuts.items():
            self.shortcuts[action] = shortcuts_data.get(action, shortcut)

    def get_shortcut(self, action_name):
        """Get shortcut for an action"""
        return self.shortcuts.get(action_name, '')

    def set_shortcut(self, action_name, shortcut):
        """Set shortcut for an action"""
        self.shortcuts[action_name] = shortcut
        self.settings.set_shortcut(action_name, shortcut)

    def apply_to_action(self, action, action_name):
        """Apply shortcut to a QAction"""
        shortcut = self.get_shortcut(action_name)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))

    def get_all_shortcuts(self):
        """Get all shortcuts"""
        return self.shortcuts.copy()

    def reset_to_defaults(self):
        """Reset shortcuts to defaults"""
        self.shortcuts = self.settings.default_settings['shortcuts'].copy()
        shortcuts_data = {}
        for action, shortcut in self.shortcuts.items():
            shortcuts_data[action] = shortcut
        self.settings.set('shortcuts', shortcuts_data)

    # Common shortcut definitions
    DEFAULT_SHORTCUTS = {
        # File operations
        'new_file': 'Ctrl+N',
        'open_file': 'Ctrl+O',
        'save_file': 'Ctrl+S',
        'save_as_file': 'Ctrl+Shift+S',
        'quit': 'Ctrl+Q',

        # Edit operations
        'undo': 'Ctrl+Z',
        'redo': 'Ctrl+Y',
        'cut': 'Ctrl+X',
        'copy': 'Ctrl+C',
        'paste': 'Ctrl+V',
        'delete_selection': 'Delete',
        'select_all': 'Ctrl+A',
        'select_none': 'Ctrl+Shift+A',
        'invert_selection': 'Ctrl+I',

        # View operations
        'zoom_in': 'Ctrl+=',
        'zoom_out': 'Ctrl+-',
        'zoom_fit': 'Ctrl+0',
        'reset_view': 'Ctrl+R',
        'single_viewport': 'Ctrl+1',
        'quad_viewport': 'Ctrl+2',
        'horizontal_split': 'Ctrl+3',
        'vertical_split': 'Ctrl+4',

        # Display modes
        'toggle_points': 'Ctrl+P',
        'toggle_curves': 'Ctrl+U',
        'toggle_patches': 'Ctrl+H',
        'toggle_shaded': 'Ctrl+G',
        'toggle_bones': 'Ctrl+B',
        'toggle_grid': 'Ctrl+G',

        # Modeling
        'add_point': 'Insert',
        'create_patch': 'Ctrl+Enter',
        'extrude': 'Ctrl+E',
        'lathe': 'Ctrl+L',

        # Animation
        'play_animation': 'Space',
        'stop_animation': 'Escape',
        'add_keyframe': 'K',
        'next_frame': 'Right',
        'prev_frame': 'Left',
    }