"""
Voice System for Roca3D
Provides speech recognition, text-to-speech, and voice pattern analysis capabilities.
"""

import threading
import time
import numpy as np
import queue
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import os

# Audio processing libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# PyQt6 integration
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox, QProgressBar

class VoiceCommand(Enum):
    """Voice commands that can be recognized"""
    CREATE_CAPSULE = "create capsule"
    DELETE_SELECTION = "delete"
    UNDO = "undo"
    REDO = "redo"
    SAVE_FILE = "save"
    OPEN_FILE = "open"
    ZOOM_IN = "zoom in"
    ZOOM_OUT = "zoom out"
    ROTATE_VIEW = "rotate"
    ADD_POINT = "add point"
    SELECT_ALL = "select all"
    CLEAR_SELECTION = "clear selection"

@dataclass
class VoicePattern:
    """Represents a voice pattern for analysis"""
    features: np.ndarray
    timestamp: float
    confidence: float
    command_type: Optional[VoiceCommand] = None
    raw_audio: Optional[np.ndarray] = None

class SpeechRecognizer(QThread):
    """Background thread for speech recognition"""

    recognition_complete = pyqtSignal(str, float)  # text, confidence
    recognition_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None
        self.is_listening = False
        self.stop_listening = False

    def run(self):
        """Main recognition loop"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            self.recognition_error.emit("Speech recognition not available")
            return

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.is_listening = True
        while not self.stop_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                confidence = getattr(audio, 'confidence', 0.8)  # Some engines provide confidence

                self.recognition_complete.emit(text.lower(), confidence)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.recognition_error.emit(f"Could not request results; {e}")
                break
            except Exception as e:
                self.recognition_error.emit(f"Error: {e}")
                break

        self.is_listening = False

    def stop(self):
        """Stop the recognition thread"""
        self.stop_listening = True

class TextToSpeechEngine(QObject):
    """Text-to-speech engine with multiple backend support"""

    speech_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.engine = None
        self.backend = 'pyttsx3'  # Default to pyttsx3

        if PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.backend = 'pyttsx3'
            except Exception as e:
                print(f"Pyttsx3 initialization failed: {e}")

        if not self.engine and GTTS_AVAILABLE:
            self.backend = 'gtts'
            print("Using gTTS as fallback TTS engine")

    def speak(self, text: str, voice: str = 'default', rate: float = 1.0):
        """Speak the given text"""
        if not text.strip():
            return

        if self.backend == 'pyttsx3' and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                if voice != 'default' and voices:
                    # Try to find matching voice
                    for v in voices:
                        if voice.lower() in v.name.lower():
                            self.engine.setProperty('voice', v.id)
                            break

                current_rate = self.engine.getProperty('rate')
                self.engine.setProperty('rate', int(current_rate * rate))

                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_complete.emit()

            except Exception as e:
                print(f"TTS error: {e}")

        elif self.backend == 'gtts':
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save("temp_speech.mp3")

                if pygame.mixer.get_init() is None:
                    pygame.mixer.init()

                pygame.mixer.music.load("temp_speech.mp3")
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                # Clean up
                try:
                    os.remove("temp_speech.mp3")
                except:
                    pass

                self.speech_complete.emit()

            except Exception as e:
                print(f"gTTS error: {e}")

class VoiceAnalyzer(QObject):
    """Analyzes voice patterns for emotion, stress, and command intent"""

    pattern_analyzed = pyqtSignal(VoicePattern)

    def __init__(self):
        super().__init__()
        self.feature_extractor = None

        if LIBROSA_AVAILABLE:
            self.feature_extractor = self._extract_features_librosa

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 44100) -> VoicePattern:
        """Analyze audio data for patterns"""
        if not LIBROSA_AVAILABLE or self.feature_extractor is None:
            # Fallback: basic analysis
            features = np.array([
                np.mean(audio_data),  # Mean amplitude
                np.std(audio_data),   # Standard deviation
                np.max(audio_data),   # Peak amplitude
                len(audio_data)       # Duration proxy
            ])
        else:
            features = self.feature_extractor(audio_data, sample_rate)

        # Simple confidence calculation based on signal strength
        confidence = min(1.0, np.mean(np.abs(audio_data)) * 1000)

        pattern = VoicePattern(
            features=features,
            timestamp=time.time(),
            confidence=confidence,
            raw_audio=audio_data
        )

        self.pattern_analyzed.emit(pattern)
        return pattern

    def _extract_features_librosa(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio features using librosa"""
        try:
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            chroma_mean = np.mean(chroma, axis=1)

            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            centroid_mean = np.mean(centroid)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            zcr_mean = np.mean(zcr)

            # RMS energy
            rms = librosa.feature.rms(y=audio_data)
            rms_mean = np.mean(rms)

            # Combine features
            features = np.concatenate([
                mfccs_mean,
                chroma_mean,
                [centroid_mean, zcr_mean, rms_mean]
            ])

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(20)  # Return zero features on error

class VoiceSystem(QObject):
    """Main voice system integrating recognition, synthesis, and analysis"""

    command_recognized = pyqtSignal(VoiceCommand, float)  # command, confidence
    speech_recognized = pyqtSignal(str, float)  # text, confidence
    voice_pattern_detected = pyqtSignal(VoicePattern)

    def __init__(self):
        super().__init__()

        # Initialize components
        self.recognizer = SpeechRecognizer()
        self.tts_engine = TextToSpeechEngine()
        self.analyzer = VoiceAnalyzer()

        # Connect signals
        self.recognizer.recognition_complete.connect(self._on_speech_recognized)
        self.recognizer.recognition_error.connect(self._on_recognition_error)
        self.analyzer.pattern_analyzed.connect(self.voice_pattern_detected)

        # Voice command mapping
        self.voice_commands = {
            'create capsule': VoiceCommand.CREATE_CAPSULE,
            'delete': VoiceCommand.DELETE_SELECTION,
            'undo': VoiceCommand.UNDO,
            'redo': VoiceCommand.REDO,
            'save': VoiceCommand.SAVE_FILE,
            'open': VoiceCommand.OPEN_FILE,
            'zoom in': VoiceCommand.ZOOM_IN,
            'zoom out': VoiceCommand.ZOOM_OUT,
            'rotate': VoiceCommand.ROTATE_VIEW,
            'add point': VoiceCommand.ADD_POINT,
            'select all': VoiceCommand.SELECT_ALL,
            'clear selection': VoiceCommand.CLEAR_SELECTION,
        }

        self.is_active = False

    def start_listening(self):
        """Start continuous speech recognition"""
        if not self.recognizer.isRunning():
            self.recognizer.stop_listening = False
            self.recognizer.start()
            self.is_active = True

    def stop_listening(self):
        """Stop speech recognition"""
        if self.recognizer.isRunning():
            self.recognizer.stop()
            self.is_active = False

    def speak(self, text: str, voice: str = 'default', rate: float = 1.0):
        """Convert text to speech"""
        self.tts_engine.speak(text, voice, rate)

    def analyze_voice_pattern(self, audio_data: np.ndarray, sample_rate: int = 44100):
        """Analyze voice pattern from audio data"""
        return self.analyzer.analyze_audio(audio_data, sample_rate)

    def _on_speech_recognized(self, text: str, confidence: float):
        """Handle recognized speech"""
        self.speech_recognized.emit(text, confidence)

        # Check for voice commands
        for phrase, command in self.voice_commands.items():
            if phrase in text:
                self.command_recognized.emit(command, confidence)
                break

    def _on_recognition_error(self, error: str):
        """Handle recognition errors"""
        print(f"Voice recognition error: {error}")

class VoiceWidget(QWidget):
    """GUI widget for voice system controls"""

    def __init__(self, voice_system: VoiceSystem, parent=None):
        super().__init__(parent)
        self.voice_system = voice_system
        self.init_ui()

        # Connect signals
        self.voice_system.speech_recognized.connect(self._on_speech_recognized)
        self.voice_system.voice_pattern_detected.connect(self._on_pattern_detected)

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Voice System: Inactive")
        layout.addWidget(self.status_label)

        # Control buttons
        button_layout = QHBoxLayout()

        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self._toggle_listening)
        button_layout.addWidget(self.listen_button)

        self.test_tts_button = QPushButton("Test TTS")
        self.test_tts_button.clicked.connect(self._test_tts)
        button_layout.addWidget(self.test_tts_button)

        layout.addLayout(button_layout)

        # Recognition display
        self.recognition_text = QTextEdit()
        self.recognition_text.setMaximumHeight(100)
        self.recognition_text.setPlaceholderText("Recognized speech will appear here...")
        layout.addWidget(self.recognition_text)

        # Voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(['default', 'female', 'male'])
        voice_layout.addWidget(self.voice_combo)
        layout.addLayout(voice_layout)

        # Pattern analysis display
        self.pattern_label = QLabel("Voice Pattern: No analysis")
        layout.addWidget(self.pattern_label)

        self.setLayout(layout)

    def _toggle_listening(self):
        """Toggle speech recognition on/off"""
        if self.voice_system.is_active:
            self.voice_system.stop_listening()
            self.listen_button.setText("Start Listening")
            self.status_label.setText("Voice System: Inactive")
        else:
            self.voice_system.start_listening()
            self.listen_button.setText("Stop Listening")
            self.status_label.setText("Voice System: Active")

    def _test_tts(self):
        """Test text-to-speech with sample text"""
        voice = self.voice_combo.currentText()
        self.voice_system.speak("Hello! This is a test of the text-to-speech system.", voice)

    def _on_speech_recognized(self, text: str, confidence: float):
        """Handle speech recognition results"""
        current_text = self.recognition_text.toPlainText()
        new_text = f"[{confidence:.2f}] {text}\n{current_text}"
        self.recognition_text.setPlainText(new_text[:1000])  # Limit text length

    def _on_pattern_detected(self, pattern: VoicePattern):
        """Handle voice pattern analysis results"""
        self.pattern_label.setText(f"Voice Pattern: {len(pattern.features)} features, "
                                 f"confidence: {pattern.confidence:.2f}")

# Export main classes
__all__ = ['VoiceSystem', 'VoiceWidget', 'VoiceCommand', 'VoicePattern']