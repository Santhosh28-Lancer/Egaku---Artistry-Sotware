from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QFileDialog,
    QMessageBox,
    QFrame,
    QProgressBar,
)

try:
    from egaku.sketch_engine import PencilRenderer
    ENGINE_IMPORT_ERROR: Exception | None = None
except Exception as import_error:
    PencilRenderer = None
    ENGINE_IMPORT_ERROR = import_error


class RenderWorker(QThread):
    progress = Signal(str)
    finished_image = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source: str,
        output: str,
        model: str,
        strength: float,
        detail: int,
    ):
        super().__init__()
        self.source = source
        self.output = output
        self.model = model
        self.strength = strength
        self.detail = detail

    def run(self) -> None:
        try:
            if PencilRenderer is None:
                raise RuntimeError(
                    "The rendering engine could not be imported. "
                    "Close Egaku, run setup_gpu.bat, and launch run.bat again. "
                    f"Import details: {ENGINE_IMPORT_ERROR}"
                )
            renderer = PencilRenderer(self.model, self.progress.emit)
            renderer.render(
                self.source,
                self.output,
                self.strength,
                self.detail,
            )
            self.finished_image.emit(self.output)
        except Exception as exc:
            log_path = None
            try:
                log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Egaku" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "last-render-error.log"
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            message = f"{type(exc).__name__}: {exc}"
            if log_path:
                message += f"\n\nFull diagnostic log:\n{log_path}"
            self.failed.emit(message)


class EgakuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Egaku — Image to Art Studio")
        self.resize(1440, 900)

        self.settings = QSettings("Egaku", "ImageToArtStudio")
        self.source_path: str | None = None
        self.output_path: str | None = None
        self.preview_path: str | None = None
        self.worker: RenderWorker | None = None

        self._build_ui()
        self._apply_style()
        self._restore_model_setting()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        # LEFT PANEL
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

        model_label = QLabel("BASE MODEL")
        model_label.setObjectName("section")
        self.model_value = QLabel("No Stable Diffusion 1.5 model selected")
        self.model_value.setWordWrap(True)

        choose_folder = QPushButton("Choose model folder")
        choose_file = QPushButton("Choose checkpoint file")
        choose_folder.clicked.connect(self.choose_model_folder)
        choose_file.clicked.connect(self.choose_model_file)

        lv.addSpacing(14)
        lv.addWidget(model_label)
        lv.addWidget(self.model_value)
        lv.addWidget(choose_folder)
        lv.addWidget(choose_file)
        lv.addStretch(1)

        privacy = QLabel("LOCAL INFERENCE\nPhotos stay on this device.")
        privacy.setObjectName("muted")
        lv.addWidget(privacy)

        # CENTER CANVAS
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

        # RIGHT PANEL
        right = QFrame()
        right.setObjectName("panel")
        right.setFixedWidth(280)
        rv = QVBoxLayout(right)

        section = QLabel("REFINE PENCIL ILLUSTRATION")
        section.setObjectName("section")

        help_text = QLabel(
            "Generate a graphite-style illustration locally. "
            "Rendering uses a low-VRAM GPU profile."
        )
        help_text.setWordWrap(True)

        self.strength_label = QLabel("Artistic strength  58")
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(35, 78)
        self.strength.setValue(58)
        self.strength.valueChanged.connect(
            lambda value: self.strength_label.setText(f"Artistic strength  {value}")
        )

        self.detail_label = QLabel("Fine detail  70")
        self.detail = QSlider(Qt.Horizontal)
        self.detail.setRange(0, 100)
        self.detail.setValue(70)
        self.detail.valueChanged.connect(
            lambda value: self.detail_label.setText(f"Fine detail  {value}")
        )

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
        rv.addWidget(
            QLabel("EGAKU · IMAGE TO ART STUDIO\nSingle pencil medium · Local model rendering")
        )

        # These were missing from the previous source: without them the
        # three panels were constructed but never shown in the main window.
        root_layout.addWidget(left)
        root_layout.addWidget(center, 1)
        root_layout.addWidget(right)
        self.setCentralWidget(root)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background: #191b22;
                color: #f1f3f8;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QFrame#panel, QFrame#canvas {
                background: #20232c;
                border: 1px solid #343846;
                border-radius: 14px;
            }
            QLabel#brand {
                font-size: 25pt;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#muted { color: #aab5c8; }
            QLabel#section { color: #c7b9ff; font-weight: 700; }
            QLabel#filename { font-size: 11pt; font-weight: 600; }
            QLabel#preview {
                background: #111319;
                border: 1px solid #303442;
                border-radius: 8px;
                color: #7d8798;
            }
            QPushButton {
                background: #303544;
                border: 1px solid #454c5e;
                border-radius: 9px;
                padding: 11px;
            }
            QPushButton:hover { background: #3a4052; }
            QPushButton:disabled { color: #778092; background: #272b35; }
            QPushButton#primary {
                background: #7961e8;
                border: 0;
                font-weight: 700;
            }
            QPushButton#primary:hover { background: #8b77f3; }
            QSlider::groove:horizontal {
                height: 6px;
                background: #41485a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #8b78f2;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #41485a;
                border-radius: 5px;
                padding: 3px;
            }
            QProgressBar::chunk { background: #7961e8; }
        """)

    def _restore_model_setting(self) -> None:
        saved = str(self.settings.value("model_path", ""))
        if saved and Path(saved).exists():
            self.model_value.setText(saved)

    def import_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import photograph",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Invalid image", "Egaku could not open this image.")
            return

        self.source_path = path
        self.output_path = None
        self.filename.setText(Path(path).name)
        self.export_btn.setEnabled(False)
        self.show_path(path)

    def show_path(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self.preview_path = path
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview_path and Path(self.preview_path).is_file():
            self.show_path(self.preview_path)

    def show_original(self) -> None:
        if self.source_path:
            self.show_path(self.source_path)

    def choose_model_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Stable Diffusion 1.5 model folder"
        )
        if path:
            self.settings.setValue("model_path", path)
            self.model_value.setText(path)

    def choose_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SD 1.5 checkpoint",
            "",
            "Model checkpoints (*.safetensors *.ckpt)",
        )
        if path:
            self.settings.setValue("model_path", path)
            self.model_value.setText(path)

    def render(self) -> None:
        if not self.source_path:
            QMessageBox.information(self, "Import image", "Import a photograph first.")
            return

        model = str(self.settings.value("model_path", ""))
        if not model or not Path(model).exists():
            QMessageBox.information(
                self,
                "Choose model",
                "Select a local Stable Diffusion 1.5 model folder or checkpoint first.",
            )
            return

        if self.worker and self.worker.isRunning():
            return

        self.output_path = str(
            Path(self.source_path).with_name(Path(self.source_path).stem + "_egaku.png")
        )
        self.render_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.show()
        self.status.setText("Starting local AI renderer…")

        self.worker = RenderWorker(
            self.source_path,
            self.output_path,
            model,
            self.strength.value() / 100.0,
            self.detail.value(),
        )
        self.worker.progress.connect(self.status.setText)
        self.worker.finished_image.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, path: str) -> None:
        self.progress.hide()
        self.render_btn.setEnabled(True)
        self.output_path = path
        self.show_path(path)
        self.export_btn.setEnabled(Path(path).is_file())
        self.status.setText("Pencil illustration ready")

    def on_failed(self, message: str) -> None:
        self.progress.hide()
        self.render_btn.setEnabled(True)
        self.export_btn.setEnabled(bool(self.output_path and Path(self.output_path).is_file()))
        self.status.setText("Render failed; original image is preserved")
        QMessageBox.critical(self, "Egaku render failed", message)

    def reset(self) -> None:
        self.strength.setValue(58)
        self.detail.setValue(70)

    def export(self) -> None:
        if not self.output_path or not Path(self.output_path).is_file():
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export artwork",
            str(Path(self.output_path)),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return

        target = Path(path)
        if not target.suffix:
            extension = ".png"
            if selected_filter.startswith("JPEG"):
                extension = ".jpg"
            elif selected_filter.startswith("WebP"):
                extension = ".webp"
            elif selected_filter.startswith("TIFF"):
                extension = ".tiff"
            target = target.with_suffix(extension)

        try:
            with Image.open(self.output_path) as image:
                image.save(target)
            self.status.setText(f"Exported: {target.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Could not export the artwork:\n{exc}")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Rendering in progress",
                "Wait for the current render to finish before closing Egaku.",
            )
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Egaku")
    window = EgakuWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
