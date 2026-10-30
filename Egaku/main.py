from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QFileDialog, QMessageBox, QFrame, QProgressBar,
    QSpinBox,
)

from egaku.sketch_engine import PencilRenderer


class RenderWorker(QThread):
    progress = Signal(str)
    finished_image = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, output: str, model: str, strength: float, detail: int):
        super().__init__()
        self.source, self.output, self.model = source, output, model
        self.strength, self.detail = strength, detail

    def run(self):
        try:
            renderer = PencilRenderer(self.model, self.progress.emit)
            renderer.render(self.source, self.output, self.strength, self.detail)
            self.finished_image.emit(self.output)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class EgakuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Egaku — Image to Art Studio")
        self.resize(1440, 900)
        self.settings = QSettings("Egaku", "LocalAIPencil")
        self.source_path: str | None = None
        self.output_path: str | None = None
        self.worker: RenderWorker | None = None
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        left = QFrame()
        left.setObjectName("panel")
        left.setFixedWidth(260)
        lv = QVBoxLayout(left)
        title = QLabel("EGAKU")
        title.setObjectName("brand")
        subtitle = QLabel("IMAGE TO ART STUDIO")
        subtitle.setObjectName("muted")
        self.import_btn = QPushButton("＋  Import image")
        self.original_btn = QPushButton("◉  Original image")
        self.import_btn.clicked.connect(self.import_image)
        self.original_btn.clicked.connect(self.show_original)
        lv.addWidget(title)
        lv.addWidget(subtitle)
        lv.addSpacing(18)
        lv.addWidget(self.import_btn)
        lv.addWidget(self.original_btn)

        self.model_label = QLabel("MODEL")
        self.model_label.setObjectName("section")
        self.model_value = QLabel("No base model selected")
        self.model_value.setWordWrap(True)
        choose_folder = QPushButton("Choose model folder")
        choose_file = QPushButton("Choose checkpoint file")
        choose_folder.clicked.connect(self.choose_model_folder)
        choose_file.clicked.connect(self.choose_model_file)
        lv.addSpacing(14)
        lv.addWidget(self.model_label)
        lv.addWidget(self.model_value)
        lv.addWidget(choose_folder)
        lv.addWidget(choose_file)
        lv.addStretch(1)
        privacy = QLabel("LOCAL INFERENCE
Photos stay on this device.")
        privacy.setObjectName("muted")
        lv.addWidget(privacy)

        center = QFrame()
        center.setObjectName("canvas")
        cv = QVBoxLayout(center)
        self.filename = QLabel("Import a photograph to begin")
        self.filename.setObjectName("filename")
        self.preview = QLabel("Your pencil illustration will appear here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(600, 500)
        self.preview.setObjectName("preview")
        self.status = QLabel("Ready")
        self.status.setObjectName("muted")
        cv.addWidget(self.filename)
        cv.addWidget(self.preview, 1)
        cv.addWidget(self.status)

        right = QFrame()
        right.setObjectName("panel")
        right.setFixedWidth(280)
        rv = QVBoxLayout(right)
        section = QLabel("REFINE PENCIL ILLUSTRATION")
        section.setObjectName("section")
        help_text = QLabel("Generate a graphite-style illustration locally.")
        help_text.setWordWrap(True)
        self.strength_label = QLabel("Artistic strength  58")
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(35, 78)
        self.strength.setValue(58)
        self.strength.valueChanged.connect(lambda v: self.strength_label.setText(f"Artistic strength  {v}"))
        self.detail_label = QLabel("Fine detail  70")
        self.detail = QSlider(Qt.Horizontal)
        self.detail.setRange(0, 100)
        self.detail.setValue(70)
        self.detail.valueChanged.connect(lambda v: self.detail_label.setText(f"Fine detail  {v}"))
        self.render_btn = QPushButton("Render pencil sketch")
        self.render_btn.setObjectName("primary")
        self.render_btn.clicked.connect(self.render)
        self.reset_btn = QPushButton("Reset adjustments")
        self.reset_btn.clicked.connect(self.reset)
        self.export_btn = QPushButton("Export artwork ↗")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        rv.addWidget(section)
        rv.addWidget(help_text)
        rv.addSpacing(12)
        rv.addWidget(self.strength_label)
        rv.addWidget(self.strength)
        rv.addWidget(self.detail_label)
        rv.addWidget(self.detail)
        rv.addSpacing(16)
        rv.addWidget(self.render_btn)
        rv.addWidget(self.reset_btn)
        rv.addWidget(self.export_btn)
        rv.addWidget(self.progress)
        rv.addStretch(1)
        rv.addWidget(QLabel("EGAKU · LOCAL AI
Single pencil medium · CUDA inference"))
        self.setCentralWidget(root)

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background: #191b22; color: #f1f3f8; font-family: 'Segoe UI'; font-size: 10pt; }
            QFrame#panel, QFrame#canvas { background: #20232c; border: 1px solid #343846; border-radius: 14px; }
            QLabel#brand { font-size: 25pt; font-weight: 800; letter-spacing: 1px; }
            QLabel#muted { color: #aab5c8; }
            QLabel#section { color: #c7b9ff; font-weight: 700; }
            QLabel#filename { font-size: 11pt; font-weight: 600; }
            QLabel#preview { background: #111319; border: 1px solid #303442; border-radius: 8px; color: #7d8798; }
            QPushButton { background: #303544; border: 1px solid #454c5e; border-radius: 9px; padding: 11px; }
            QPushButton:hover { background: #3a4052; }
            QPushButton#primary { background: #7961e8; border: 0; font-weight: 700; }
            QPushButton#primary:hover { background: #8b77f3; }
            QSlider::groove:horizontal { height: 6px; background: #41485a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #8b78f2; width: 16px; margin: -5px 0; border-radius: 8px; }
            QProgressBar { border: 1px solid #41485a; border-radius: 5px; padding: 3px; }
            QProgressBar::chunk { background: #7961e8; }
        """)

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import photograph", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self.source_path = path
        self.output_path = None
        self.filename.setText(Path(path).name)
        self.export_btn.setEnabled(False)
        self.show_path(path)

    def show_path(self, path: str):
        pix = QPixmap(path)
        if not pix.isNull():
            self.preview.setPixmap(pix.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.output_path and Path(self.output_path).exists():
            self.show_path(self.output_path)
        elif self.source_path:
            self.show_path(self.source_path)

    def show_original(self):
        if self.source_path:
            self.show_path(self.source_path)

    def choose_model_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Stable Diffusion 1.5 model folder")
        if path:
            self.settings.setValue("model_path", path)
            self.model_value.setText(path)

    def choose_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SD 1.5 checkpoint", "", "Model checkpoints (*.safetensors *.ckpt)"
        )
        if path:
            self.settings.setValue("model_path", path)
            self.model_value.setText(path)

    def render(self):
        if not self.source_path:
            QMessageBox.information(self, "Import image", "Import a photograph first.")
            return
        model = str(self.settings.value("model_path", ""))
        if not model or not Path(model).exists():
            QMessageBox.information(
                self, "Choose model",
                "Select your local Stable Diffusion 1.5 model folder or checkpoint first."
            )
            return
        if self.worker and self.worker.isRunning():
            return
        self.output_path = str(Path(self.source_path).with_name(Path(self.source_path).stem + "_egaku.png"))
        self.render_btn.setEnabled(False)
        self.progress.show()
        self.status.setText("Starting local AI renderer…")
        self.worker = RenderWorker(
            self.source_path, self.output_path, model,
            self.strength.value() / 100.0, self.detail.value()
        )
        self.worker.progress.connect(self.status.setText)
        self.worker.finished_image.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, path: str):
        self.progress.hide()
        self.render_btn.setEnabled(True)
        self.output_path = path
        self.show_path(path)
        self.export_btn.setEnabled(True)
        self.status.setText("Pencil illustration ready")

    def on_failed(self, message: str):
        self.progress.hide()
        self.render_btn.setEnabled(True)
        self.status.setText("Render failed")
        QMessageBox.critical(self, "Egaku render failed", message)

    def reset(self):
        self.strength.setValue(58)
        self.detail.setValue(70)

    def export(self):
        if not self.output_path or not Path(self.output_path).exists():
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export artwork", str(Path(self.output_path)),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        image = Image.open(self.output_path)
        image.save(path)
        self.status.setText(f"Exported: {Path(path).name}")


def main():
    app = QApplication(sys.argv)
    window = EgakuWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
