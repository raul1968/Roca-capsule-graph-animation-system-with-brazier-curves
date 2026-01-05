#!/usr/bin/env python3
"""
Settings System for PyQt6 JPatch
"""

import json
import os
from PyQt6.QtCore import QSettings

class JPatchSettings:
    """Persistent settings manager for PyQt6 JPatch"""

    def __init__(self):
        self.settings = QSettings("PyQt6JPatch", "JPatch")
        self.default_settings = {
            # Viewport settings
            'viewport_layout': 'single',
            'show_points': True,
            'show_curves': True,
            'show_patches': True,
            'show_shaded': False,
            'show_bones': False,
            'show_grid': True,
            'lighting_mode': 'head',
            'camera_distance': 5.0,
            'camera_rotation': [0.0, 0.0],

            # UI settings
            'window_geometry': None,
            'window_state': None,
            'sidebar_visible': True,
            'statusbar_visible': True,

            # File settings
            'last_open_dir': '',
            'last_save_dir': '',
            'auto_save_enabled': True,
            'auto_save_interval': 300,  # seconds

            # Animation settings
            'animation_fps': 30,
            'animation_loop': False,
            'keyframe_snap': True,
            'keyframe_snap_increment': 0.1,

            # Performance settings
            'max_undo_steps': 100,
            'viewport_antialiasing': True,
            'high_quality_rendering': False,

            # Keyboard shortcuts
            'shortcuts': {
                'new_file': 'Ctrl+N',
                'open_file': 'Ctrl+O',
                'save_file': 'Ctrl+S',
                'undo': 'Ctrl+Z',
                'redo': 'Ctrl+Y',
                'add_point': 'Ctrl+P',
                'delete_selection': 'Delete',
                'select_all': 'Ctrl+A',
            }
        }

    def get(self, key, default=None):
        """Get a setting value"""
        if default is None:
            default = self.default_settings.get(key)
        return self.settings.value(key, default)

    def set(self, key, value):
        """Set a setting value"""
        self.settings.setValue(key, value)
        self.settings.sync()

    def get_all(self):
        """Get all current settings"""
        all_settings = {}
        for key in self.default_settings.keys():
            all_settings[key] = self.get(key)
        return all_settings

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        for key, value in self.default_settings.items():
            self.set(key, value)

    def save_window_state(self, window):
        """Save window geometry and state"""
        self.set('window_geometry', window.saveGeometry())
        self.set('window_state', window.saveState())

    def restore_window_state(self, window):
        """Restore window geometry and state"""
        geometry = self.get('window_geometry')
        if geometry:
            window.restoreGeometry(geometry)

        state = self.get('window_state')
        if state:
            window.restoreState(state)

    def get_shortcut(self, action_name):
        """Get keyboard shortcut for an action"""
        shortcuts = self.get('shortcuts', {})
        return shortcuts.get(action_name, self.default_settings['shortcuts'].get(action_name, ''))

    def set_shortcut(self, action_name, shortcut):
        """Set keyboard shortcut for an action"""
        shortcuts = self.get('shortcuts', {})
        shortcuts[action_name] = shortcut
        self.set('shortcuts', shortcuts)