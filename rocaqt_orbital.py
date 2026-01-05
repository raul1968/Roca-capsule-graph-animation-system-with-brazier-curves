#!/usr/bin/env python3
"""
Minimal Qt-based chatbot + orbital capsule visual.

This single-file GUI is intentionally self-contained and designed to be
copied to a thumb drive and run on machines that have PyQt5 or PySide6.

How to integrate with other ROCA systems:
- Call `window.set_message_handler(callable)` where the callable accepts a
  single `str` argument (user message) and returns a response `str`.
  The callable may block; it will be run in a worker thread to avoid UI lock.

"""
from __future__ import annotations
import sys
import math
import html
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, List

try:
    from PyQt6 import QtWidgets, QtGui, QtCore
    from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
except Exception as exc:
    raise ImportError("Requires PyQt6 installed") from exc


class Worker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)


class WorkerSignals(QtCore.QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)


class OrbitalWidget(QtWidgets.QWidget):
    """Simple orbital capsule visual.

    - Draws a circular orbit and a capsule that moves around it.
    - Provide `set_speed()` and `set_radius()` to adjust animation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._speed = 30.0  # degrees per second (unused for capsule individual motion)
        self._radius = 80
        self._capsule_size = 14
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(16)
        self.setMinimumSize(320, 320)

        # visualization state
        self.capsules: List[Capsule] = []
        self.zoom_level = 1.0
        self.selected_capsule = None
        self.hovered_capsule = None
        self.show_info = True

        # Colors matching `roca_chat.py`
        self.bg_color = QtGui.QColor(20, 20, 30)
        self.panel_color = QtGui.QColor(30, 30, 40)
        self.border_color = QtGui.QColor(60, 60, 80)
        self.highlight_color = QtGui.QColor(100, 200, 255)
        self.text_color = QtGui.QColor(255, 255, 255)

    def set_speed(self, deg_per_sec: float):
        self._speed = deg_per_sec

    def set_radius(self, r: int):
        self._radius = r
        self.update()

    def _step(self):
        dt = self._timer.interval() / 1000.0
        self._angle = (self._angle + self._speed * dt) % 360.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cx = self.width() // 2
            cy = self.height() // 2
            mx = event.pos().x()
            my = event.pos().y()
            for cap in self.capsules:
                orbit_radius = cap.orbit_radius * 50 * self.zoom_level
                x = cx + orbit_radius * math.cos(cap.angle)
                y = cy + orbit_radius * math.sin(cap.angle)
                dist = math.hypot(mx - x, my - y)
                if dist < 20:
                    self.selected_capsule = cap
                    self.update()
                    break

    def mouseMoveEvent(self, event):
        cx = self.width() // 2
        cy = self.height() // 2
        mx = event.pos().x()
        my = event.pos().y()
        self.hovered_capsule = None
        for cap in self.capsules:
            orbit_radius = cap.orbit_radius * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)
            dist = math.hypot(mx - x, my - y)
            if dist < 20:
                self.hovered_capsule = cap
                break

    def wheelEvent(self, event: "QtGui.QWheelEvent"):
        # Zoom in/out with wheel; use 120 units per notch standard
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # scale factor per notch
        factor = 1.2 ** (delta / 120)
        new_zoom = self.zoom_level * factor
        # clamp zoom
        self.zoom_level = max(0.2, min(3.0, new_zoom))
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        cx = self.width() // 2
        cy = self.height() // 2

        # Draw starfield background
        self.draw_starfield(p)

        # Update capsule angles
        for cap in self.capsules:
            # small autonomous drift
            cap.angle = (cap.angle + 0.005 * (0.5 + cap.certainty)) % (2 * math.pi)

        # Draw orbital paths grouped by radius
        orbit_radii = sorted(set(int(cap.orbit_radius * 100) / 100.0 for cap in self.capsules))
        p.setPen(QtGui.QPen(QtGui.QColor(60, 80, 120), 1))
        p.setOpacity(0.35)
        for radius in orbit_radii:
            orbit_radius = radius * 50 * self.zoom_level
            p.drawEllipse(int(cx - orbit_radius), int(cy - orbit_radius), int(orbit_radius * 2), int(orbit_radius * 2))
        p.setOpacity(1.0)

        # Draw capsules
        for cap in self.capsules:
            orbit_radius = cap.orbit_radius * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)

            # color by kind
            if cap.kind == "theory":
                color = QtGui.QColor(200, 100, 255)
            elif cap.kind == "hypothesis":
                color = QtGui.QColor(255, 200, 100)
            elif cap.kind == "method":
                color = QtGui.QColor(100, 200, 255)
            else:
                color = QtGui.QColor(150, 255, 150)

            size = max(6, int(14 * cap.certainty))

            # Glow
            glow_radius = size + 6
            for i in range(3, 0, -1):
                glow_color = QtGui.QColor(color)
                glow_color.setAlpha(int(40 / (i + 1)))
                p.setBrush(QtGui.QBrush(glow_color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(int(x - glow_radius / 2 - i * 2), int(y - glow_radius / 2 - i * 2), int(glow_radius + i * 4), int(glow_radius + i * 4))

            # Body
            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.drawEllipse(int(x - size / 2), int(y - size / 2), size, size)

            # Label
            if self.zoom_level > 0.8 or cap == self.selected_capsule or cap == self.hovered_capsule:
                p.setPen(QtGui.QPen(self.text_color))
                f = p.font()
                f.setPointSize(8)
                p.setFont(f)
                label = cap.character if cap.character else (cap.kind[:3])
                p.drawText(int(x + size / 2 + 5), int(y + 3), label)

            # Highlights
            if cap == self.selected_capsule:
                p.setPen(QtGui.QPen(self.highlight_color, 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 5), int(y - size / 2 - 5), size + 10, size + 10)
                p.setPen(QtGui.QPen(self.highlight_color, 1))
                p.setOpacity(0.5)
                p.drawLine(int(cx), int(cy), int(x), int(y))
                p.setOpacity(1.0)
            elif cap == self.hovered_capsule:
                p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 150), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 3), int(y - size / 2 - 3), size + 6, size + 6)

        # Central nucleus
        self.draw_central_nucleus(p, cx, cy)

        # Info panel
        if self.show_info:
            self.draw_info_panel(p)

        p.end()

    def draw_starfield(self, p: QtGui.QPainter):
        p.fillRect(self.rect(), self.bg_color)
        random.seed(42)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        num_stars = 160
        w = self.width()
        h = self.height()
        for _ in range(num_stars):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(1, 3)
            brightness = random.randint(100, 255)
            p.setBrush(QtGui.QBrush(QtGui.QColor(brightness, brightness, brightness)))
            p.drawEllipse(x, y, size, size)
        random.seed()

    def draw_central_nucleus(self, p: QtGui.QPainter, cx: int, cy: int):
        for i in range(30, 0, -5):
            glow_color = QtGui.QColor(255, 200, 0)
            glow_color.setAlpha(int(100 * (30 - i) / 30))
            p.setBrush(QtGui.QBrush(glow_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(cx - i), int(cy - i), i * 2, i * 2)
        gradient_color = QtGui.QColor(255, 220, 100)
        p.setBrush(QtGui.QBrush(gradient_color))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 240, 150), 2))
        p.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 200)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        f = p.font()
        f.setPointSize(11)
        f.setBold(True)
        p.setFont(f)
        p.drawText(int(cx + 20), int(cy - 5), "Identity")
        p.drawText(int(cx + 20), int(cy + 10), "Nucleus")

    def draw_info_panel(self, p: QtGui.QPainter):
        panel_w = 80
        panel_h = 60
        panel_bg = QtGui.QColor(0, 0, 0)
        panel_bg.setAlpha(200)
        p.fillRect(10, 10, panel_w, panel_h, panel_bg)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        p.drawRect(10, 10, panel_w, panel_h)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
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


class RocaOrbitalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROCA Orbital Chat")
        self.resize(900, 520)

        # Central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        central.setStyleSheet("background-color: rgb(20,20,30);")

        # Chat area
        chat_panel = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_panel)
        self.chat_view = QtWidgets.QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setStyleSheet("background-color: rgb(30,30,40); color: white; border: 1px solid rgb(60,60,80);")
        self.chat_view.setFont(QtGui.QFont("Consolas", 11))
        self.input_line = QtWidgets.QLineEdit()
        self.input_line.setStyleSheet("background-color: rgb(40,40,50); color: white; border: 1px solid rgb(60,60,80); padding:4px;")
        self.input_line.setFont(QtGui.QFont("Consolas", 11))
        self.input_line.returnPressed.connect(self._on_send)
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.setStyleSheet("background-color: rgb(0,200,255); color: black; font-weight: bold; padding:6px 12px; border-radius:4px;")
        send_btn.clicked.connect(self._on_send)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.input_line)
        bottom.addWidget(send_btn)
        chat_layout.addWidget(self.chat_view)
        chat_layout.addLayout(bottom)

        # Right: orbital visual + small controls
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.orbital = OrbitalWidget()
        right_layout.addWidget(self.orbital)
        speed_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        speed_slider.setRange(0, 360)
        speed_slider.setValue(30)
        speed_slider.valueChanged.connect(lambda v: self.orbital.set_speed(float(v)))
        radius_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        radius_slider.setRange(40, 200)
        radius_slider.setValue(80)
        radius_slider.valueChanged.connect(lambda v: self.orbital.set_radius(int(v)))
        right_layout.addWidget(QtWidgets.QLabel("Orbit speed (deg/s)"))
        right_layout.addWidget(speed_slider)
        right_layout.addWidget(QtWidgets.QLabel("Orbit radius"))
        right_layout.addWidget(radius_slider)

        # Combine
        h.addWidget(chat_panel, 2)
        h.addWidget(right_panel, 1)

        # Message handler can be set by integrator (sync function that returns str)
        self._message_handler: Optional[Callable[[str], str]] = None
        self._threadpool = QtCore.QThreadPool.globalInstance()

        # Populate orbital with example capsules so lanes are visible
        for i in range(12):
            kind = random.choice(["theory", "method", "hypothesis", "concept"])
            cap = Capsule(
                content=f"Example {i}",
                kind=kind,
                certainty=random.uniform(0.4, 1.0),
                orbit_radius=random.uniform(0.6, 2.2),
                angle=random.uniform(0, 2 * math.pi),
            )
            self.orbital.capsules.append(cap)

        # Menu / status bar similar to `roca_chat.py`
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.statusBar().showMessage("ROCA Orbital — ready")

    def set_message_handler(self, fn: Callable[[str], str]):
        """Set the callable that will be called with user messages.

        The callable should accept a single `str` (user message) and return a
        `str` (response). It will be executed in a worker thread to avoid
        blocking the UI.
        """
        self._message_handler = fn

    def append_chat(self, who: str, msg: str):
        safe = html.escape(msg)
        self.chat_view.append(f"<b>{who}:</b> {safe}")

    def _on_send(self):
        text = self.input_line.text().strip()
        if not text:
            return
        self.input_line.clear()
        self.append_chat("You", text)

        if not self._message_handler:
            # default echo
            resp = f"(local echo) {text}"
            self.append_chat("ROCA", resp)
            # also add to orbital as a new capsule
            cap = Capsule(
                content=text,
                kind="concept",
                certainty=min(1.0, 0.4 + len(text) / 200.0),
                orbit_radius=0.8 + random.random() * 1.6,
                angle=random.uniform(0, 2 * math.pi),
            )
            self.orbital.capsules.append(cap)
            self.tribunal_view.append(f"Added capsule: {text[:60]}")
            return

        # Run handler in worker
        def call_handler(msg):
            return self._message_handler(msg)

        worker = Worker(call_handler, text)
        worker.signals.result.connect(lambda r: self.append_chat("ROCA", str(r)))
        worker.signals.error.connect(lambda e: self.append_chat("ROCA", f"<i>Error:</i> {e}"))
        self._threadpool.start(worker)


def _demo_handler(msg: str) -> str:
    # Very small placeholder example — integrators should supply a real one
    return f"Received: {msg[:200]}"


@dataclass
class Capsule:
    content: str
    kind: str = "concept"
    certainty: float = 0.6
    orbit_radius: float = 1.0
    angle: float = 0.0
    character: Optional[str] = None
    id: str = field(default_factory=lambda: str(random.randint(100000, 999999)))


def main(argv=None):
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication(argv)
    win = RocaOrbitalWindow()
    win.set_message_handler(_demo_handler)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
