import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar,
    QTextEdit, QGroupBox, QCheckBox, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import subprocess
import json
import tempfile
import shutil


class PRJConverterThread(QThread):
    """Background thread for PRJ conversion"""
    progress_update = pyqtSignal(int, str)
    conversion_complete = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, prj_path, output_format, output_dir, options):
        super().__init__()
        self.prj_path = prj_path
        self.output_format = output_format
        self.output_dir = output_dir
        self.options = options
        self.running = True

    def run(self):
        try:
            # Create temporary directory for intermediate files
            temp_dir = tempfile.mkdtemp()
            self.log_message.emit(f"Created temporary directory: {temp_dir}")

            # Step 1: Extract information from PRJ file
            self.progress_update.emit(10, "Reading PRJ file...")
            prj_info = self.read_prj_file(self.prj_path)
            self.log_message.emit(f"PRJ info: {json.dumps(prj_info, indent=2)}")

            # Step 2: Convert PRJ to intermediate format
            self.progress_update.emit(30, "Converting to intermediate format...")
            intermediate_file = self.convert_to_intermediate(prj_info, temp_dir)

            if not intermediate_file or not os.path.exists(intermediate_file):
                raise Exception("Failed to create intermediate file")

            # Step 3: Convert intermediate to target format
            self.progress_update.emit(60, f"Converting to {self.output_format}...")
            output_file = self.convert_to_target(intermediate_file, self.output_format, self.output_dir)

            # Step 4: Clean up
            self.progress_update.emit(90, "Cleaning up...")
            shutil.rmtree(temp_dir)

            self.progress_update.emit(100, "Conversion complete!")
            self.conversion_complete.emit(True, output_file)

        except Exception as e:
            self.log_message.emit(f"Error: {str(e)}")
            self.conversion_complete.emit(False, str(e))

    def read_prj_file(self, prj_path):
        """Read and parse PRJ file (simplified implementation)"""
        prj_info = {
            'file_path': prj_path,
            'file_size': os.path.getsize(prj_path),
            'vertices': 0,
            'faces': 0,
            'materials': [],
            'animation_frames': 0
        }

        try:
            with open(prj_path, 'rb') as f:
                # Read file header (this is a simplified example)
                header = f.read(100)
                
                # For demonstration, we'll create dummy data
                # In a real implementation, you would parse the actual PRJ format
                prj_info['vertices'] = 1000
                prj_info['faces'] = 2000
                prj_info['materials'] = ['Material1', 'Material2']
                prj_info['animation_frames'] = 60

            self.log_message.emit(f"Successfully read PRJ file: {os.path.basename(prj_path)}")
        except Exception as e:
            self.log_message.emit(f"Warning: Could not fully parse PRJ file: {e}")
            
        return prj_info

    def convert_to_intermediate(self, prj_info, temp_dir):
        """Convert PRJ to an intermediate format (PLY in this example)"""
        intermediate_file = os.path.join(temp_dir, "intermediate.ply")
        
        try:
            # Create a simple PLY file as intermediate format
            # In a real implementation, this would convert actual PRJ geometry
            with open(intermediate_file, 'w') as f:
                f.write(f"""ply
format ascii 1.0
comment Converted from PRJ: {prj_info['file_path']}
element vertex {prj_info['vertices']}
property float x
property float y
property float z
element face {prj_info['faces']}
property list uchar int vertex_index
end_header
""")
                # Write dummy vertex data
                for i in range(prj_info['vertices']):
                    f.write(f"{i*0.1} {i*0.2} {i*0.3}\n")
                
                # Write dummy face data
                for i in range(prj_info['faces']):
                    f.write(f"3 {i} {(i+1)%prj_info['vertices']} {(i+2)%prj_info['vertices']}\n")

            self.log_message.emit(f"Created intermediate file: {intermediate_file}")
            return intermediate_file
        except Exception as e:
            self.log_message.emit(f"Error creating intermediate file: {e}")
            return None

    def convert_to_target(self, intermediate_file, output_format, output_dir):
        """Convert intermediate file to target format"""
        base_name = os.path.splitext(os.path.basename(self.prj_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.{output_format.lower()}")
        
        try:
            if output_format == "OBJ":
                # Convert to OBJ (simplified - just copy the intermediate with .obj extension)
                with open(intermediate_file, 'r') as src, open(output_file, 'w') as dst:
                    dst.write("# Converted from PRJ file\n")
                    dst.write(f"# Original: {self.prj_path}\n")
                    
                    # Simple conversion from PLY to OBJ format
                    lines = src.readlines()
                    in_vertices = False
                    in_faces = False
                    
                    for line in lines:
                        if line.strip() == "end_header":
                            in_vertices = True
                            continue
                        elif "element vertex" in line:
                            continue
                        elif "element face" in line:
                            in_vertices = False
                            in_faces = True
                            continue
                        
                        if in_vertices and line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                dst.write(f"v {parts[0]} {parts[1]} {parts[2]}\n")
                        elif in_faces and line.strip():
                            parts = line.strip().split()
                            if len(parts) > 1:
                                # Convert from PLY face format to OBJ
                                indices = parts[1:]
                                dst.write(f"f {' '.join([str(int(i)+1) for i in indices])}\n")
                
                self.log_message.emit(f"Created OBJ file: {output_file}")
                
            elif output_format == "FBX":
                # For FBX, we would use a library like FBX SDK or similar
                # This is a placeholder implementation
                with open(output_file, 'w') as f:
                    f.write(f"""; FBX 7.4.0 project file
; Created by PRJ Converter
; Original file: {self.prj_path}

Objects: {{
    Model: "ConvertedModel", "Mesh" {{
        Vertices: /* dummy vertices */
        PolygonVertexIndex: /* dummy indices */
    }}
}}
""")
                self.log_message.emit(f"Created FBX file (placeholder): {output_file}")
            
            return output_file
        except Exception as e:
            self.log_message.emit(f"Error creating output file: {e}")
            return None


class PRJConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hash Animation Master PRJ Converter")
        self.setGeometry(100, 100, 800, 600)
        
        self.prj_file = None
        self.converter_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Hash Animation Master .PRJ Converter")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # PRJ file selection
        prj_layout = QHBoxLayout()
        self.prj_label = QLabel("No file selected")
        self.prj_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        self.prj_label.setMinimumHeight(30)
        prj_layout.addWidget(self.prj_label)
        
        browse_btn = QPushButton("Browse PRJ File")
        browse_btn.clicked.connect(self.browse_prj_file)
        browse_btn.setMinimumWidth(100)
        prj_layout.addWidget(browse_btn)
        
        file_layout.addLayout(prj_layout)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Conversion settings group
        settings_group = QGroupBox("Conversion Settings")
        settings_layout = QVBoxLayout()
        
        # Output format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["OBJ", "FBX"])
        self.format_combo.setCurrentText("OBJ")
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        settings_layout.addLayout(format_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output to same directory as input")
        self.output_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        output_layout.addWidget(self.output_label)
        
        output_btn = QPushButton("Browse Output")
        output_btn.clicked.connect(self.browse_output_dir)
        output_btn.setMinimumWidth(100)
        output_layout.addWidget(output_btn)
        
        settings_layout.addLayout(output_layout)
        
        # Conversion options
        options_layout = QVBoxLayout()
        self.include_normals = QCheckBox("Include vertex normals")
        self.include_normals.setChecked(True)
        options_layout.addWidget(self.include_normals)
        
        self.include_uvs = QCheckBox("Include UV coordinates")
        self.include_uvs.setChecked(True)
        options_layout.addWidget(self.include_uvs)
        
        self.include_materials = QCheckBox("Include materials")
        self.include_materials.setChecked(True)
        options_layout.addWidget(self.include_materials)
        
        # Animation options
        anim_layout = QHBoxLayout()
        anim_layout.addWidget(QLabel("Animation Frames:"))
        self.anim_spin = QSpinBox()
        self.anim_spin.setRange(0, 1000)
        self.anim_spin.setValue(0)
        self.anim_spin.setToolTip("0 = no animation, export only first frame")
        anim_layout.addWidget(self.anim_spin)
        anim_layout.addStretch()
        options_layout.addLayout(anim_layout)
        
        settings_layout.addLayout(options_layout)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.progress_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.setEnabled(False)
        button_layout.addWidget(self.convert_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Log window
        log_group = QGroupBox("Conversion Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def browse_prj_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PRJ File",
            "",
            "Hash Animation Master Files (*.prj);;All Files (*.*)"
        )
        
        if file_path:
            self.prj_file = file_path
            self.prj_label.setText(file_path)
            self.output_label.setText(os.path.dirname(file_path))
            self.convert_btn.setEnabled(True)
            self.log_text.append(f"Selected PRJ file: {file_path}")
            
    def browse_output_dir(self):
        if self.prj_file:
            default_dir = os.path.dirname(self.prj_file)
        else:
            default_dir = ""
            
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            default_dir
        )
        
        if dir_path:
            self.output_label.setText(dir_path)
            
    def start_conversion(self):
        if not self.prj_file or not os.path.exists(self.prj_file):
            QMessageBox.warning(self, "Error", "Please select a valid PRJ file.")
            return
            
        # Prepare conversion options
        options = {
            'include_normals': self.include_normals.isChecked(),
            'include_uvs': self.include_uvs.isChecked(),
            'include_materials': self.include_materials.isChecked(),
            'animation_frames': self.anim_spin.value()
        }
        
        # Get output directory
        output_dir = self.output_label.text()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                QMessageBox.warning(self, "Error", "Cannot create output directory.")
                return
        
        # Update UI
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.append(f"\nStarting conversion to {self.format_combo.currentText()}...")
        
        # Create and start conversion thread
        self.converter_thread = PRJConverterThread(
            self.prj_file,
            self.format_combo.currentText(),
            output_dir,
            options
        )
        self.converter_thread.progress_update.connect(self.update_progress)
        self.converter_thread.conversion_complete.connect(self.conversion_finished)
        self.converter_thread.log_message.connect(self.add_log_message)
        self.converter_thread.start()
        
    def cancel_conversion(self):
        if self.converter_thread and self.converter_thread.isRunning():
            self.converter_thread.running = False
            self.converter_thread.terminate()
            self.converter_thread.wait()
            self.log_text.append("Conversion cancelled.")
            self.reset_ui()
            
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        
    def add_log_message(self, message):
        self.log_text.append(message)
        
    def conversion_finished(self, success, message):
        if success:
            self.log_text.append(f"Successfully converted to: {message}")
            QMessageBox.information(self, "Success", f"File converted successfully!\n\nOutput: {message}")
        else:
            self.log_text.append(f"Conversion failed: {message}")
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")
            
        self.reset_ui()
        
    def reset_ui(self):
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Ready")
        self.statusBar().showMessage("Ready")
        

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    window = PRJConverterApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()