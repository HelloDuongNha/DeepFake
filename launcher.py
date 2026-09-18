"""Simple macOS launcher for Deep-Live-Cam webcam mode."""

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin/python"

# VS Code's Run button may use macOS's system Python. Reopen this launcher with
# the project's interpreter before importing packages installed in .venv.
if Path(sys.prefix).resolve() != VENV_DIR.resolve():
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Missing project Python: {VENV_PYTHON}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)


SOURCE_DIR = Path(os.environ.get("DLC_SOURCE_DIR", PROJECT_DIR / "source_faces"))


class Launcher(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Deep-Live-Cam · Live với OBS")
        self.resize(680, 700)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_DIR))
        self.process.setProgram(str(PROJECT_DIR / ".venv/bin/python"))
        self.process.setArguments([
            "run.py", "--execution-provider", "auto",
            "--frame-processor", "face_swapper", "--live-resizable",
        ])
        self._stdout_pending = ""
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)

        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("Deep-Live-Cam")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QLabel("1. Chọn ảnh khuôn mặt bạn được phép sử dụng"))

        image_row = QHBoxLayout()
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Chọn ảnh JPG, PNG hoặc WEBP")
        image_row.addWidget(self.image_path, 1)
        browse = QPushButton("Chọn ảnh…")
        browse.clicked.connect(self._choose_image)
        image_row.addWidget(browse)
        layout.addLayout(image_row)

        self.preview = QLabel("Chưa chọn ảnh")
        self.preview.setObjectName("preview")
        self.preview.setFixedHeight(120)
        layout.addWidget(self.preview)

        layout.addWidget(QLabel("2. Chọn webcam"))
        self.camera = QComboBox()
        self.camera.addItem("Camera 0 (thường là webcam mặc định)", 0)
        self.camera.addItem("Camera 1 (nếu có)", 1)
        layout.addWidget(self.camera)

        layout.addWidget(QLabel("3. Chọn chất lượng hình"))
        self.quality = QComboBox()
        self.quality.addItem("Mac mượt 360p — không dùng enhancer (khuyên dùng)", (640, 360, "None", 3))
        self.quality.addItem("Mac 360p + GPEN-256 mỗi 3 frame", (640, 360, "GPEN-256", 3))
        self.quality.addItem("HD 720p — không dùng enhancer", (1280, 720, "None", 3))
        self.quality.addItem("HD 720p + GPEN-512 mỗi 3 frame", (1280, 720, "GPEN-512", 3))
        self.quality.addItem("360p — ưu tiên tốc độ", (640, 360, "None", 3))
        layout.addWidget(self.quality)
        quality_note = QLabel(
            "Nếu mặt bị mịn như ảnh AI, dùng HD 720p cân bằng. Model hiện tại không thay kiểu tóc."
        )
        quality_note.setWordWrap(True)
        layout.addWidget(quality_note)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Mở Live")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self._start)
        actions.addWidget(self.start_button)
        self.stop_button = QPushButton("Đóng Live")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.process.terminate)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)

        self.status = QLabel("Sẵn sàng")
        layout.addWidget(self.status)
        obs = QLabel(
            "4. Trong OBS: Sources → + → macOS Screen Capture → "
            "Method: Window Capture → chọn cửa sổ preview của Deep-Live-Cam. "
            "Cấp quyền Screen Recording cho OBS nếu macOS hỏi. "
            "Nếu cần đưa cảnh OBS vào Zoom/Meet, bấm Start Virtual Camera trong OBS."
        )
        obs.setWordWrap(True)
        obs.setObjectName("obs")
        layout.addWidget(obs)
        layout.addStretch()

        self.image_path.textChanged.connect(self._update_preview)
        self.setStyleSheet("""
            QMainWindow { background: #171b22; color: #f2f4f8; }
            QLabel { color: #f2f4f8; font-size: 14px; }
            QLabel#title { font-size: 26px; font-weight: 700; }
            QLabel#preview { background: #242b36; border: 1px solid #455064;
                             border-radius: 8px; qproperty-alignment: AlignCenter; }
            QLabel#obs { background: #242b36; border-radius: 8px; padding: 12px; }
            QLineEdit, QComboBox { background: #242b36; color: #f2f4f8;
                                  border: 1px solid #455064; padding: 8px; }
            QPushButton { background: #38465a; color: white; border: 0;
                          border-radius: 7px; padding: 10px 16px; }
            QPushButton:hover { background: #50627c; }
            QPushButton:disabled { background: #2b313b; color: #7d8795; }
            QPushButton#start { background: #246db3; font-weight: 700; }
            QPushButton#start:hover { background: #3285d4; }
        """)

    def _choose_image(self) -> None:
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh khuôn mặt", str(SOURCE_DIR),
            "Ảnh (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if path:
            self.image_path.setText(path)

    def _update_preview(self) -> None:
        pixmap = QPixmap(self.image_path.text())
        if pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Chưa chọn ảnh hợp lệ")
        else:
            self.preview.setText("")
            self.preview.setPixmap(pixmap.scaled(
                180, 110, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def _start(self) -> None:
        image = Path(self.image_path.text()).expanduser()
        if not image.is_file() or QPixmap(str(image)).isNull():
            QMessageBox.warning(self, "Ảnh chưa hợp lệ", "Hãy chọn một file ảnh hợp lệ.")
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        if not (PROJECT_DIR / ".venv/bin/python").is_file():
            QMessageBox.critical(self, "Thiếu môi trường", "Không tìm thấy .venv/bin/python.")
            return

        env = QProcessEnvironment.systemEnvironment()
        env.insert("DLC_SOURCE_IMAGE", str(image.resolve()))
        env.insert("DLC_SOURCE_DIR", str(image.resolve().parent))
        env.insert("DLC_CAMERA_INDEX", str(self.camera.currentData()))
        env.insert("DLC_AUTO_LIVE", "1")
        width, height, enhancer, interval = self.quality.currentData()
        env.insert("DLC_CAPTURE_WIDTH", str(width))
        env.insert("DLC_CAPTURE_HEIGHT", str(height))
        env.insert("DLC_ENHANCER", enhancer)
        env.insert("DLC_ENHANCER_INTERVAL", str(interval))
        env.insert("DLC_MASK_BLUR", "3")
        env.insert("DLC_MASK_EROSION", "3")
        env.insert("DLC_BLEND_MODE", "alpha")
        env.insert("DLC_PROFILE", "mac-smooth")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
        env.insert("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
        env.insert("TMPDIR", str(PROJECT_DIR / ".tmp") + os.sep)
        for name in (".cache", ".cache/matplotlib", ".tmp"):
            (PROJECT_DIR / name).mkdir(parents=True, exist_ok=True)
        self.process.setProcessEnvironment(env)
        self.process.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Đang mở Deep-Live-Cam. Preview có thể mất 10–30 giây…")

    def _on_stdout(self) -> None:
        self._stdout_pending += bytes(self.process.readAllStandardOutput()).decode(
            errors="replace"
        )
        lines = self._stdout_pending.split("\n")
        self._stdout_pending = lines.pop()
        for line in lines:
            if "[webcam] Camera running at" in line:
                self.status.setText(line.strip())

    def _on_error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(f"Không mở được Live: {self.process.errorString()}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if code:
            output = bytes(self.process.readAllStandardError()).decode(errors="replace")
            self.status.setText(f"Live đã dừng (mã {code}). {output[-180:]}")
        else:
            self.status.setText("Live đã đóng.")

    def closeEvent(self, event) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            self.process.waitForFinished(3000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
