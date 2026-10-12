"""Entry point used by the packaged desktop application."""

import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from modules.desktop_bootstrap import (
    download_core_model, missing_core_models, prepare_desktop,
)
from modules.paths import CACHE_DIR, DATA_DIR, MODELS_DIR, SETTINGS_PATH


class ModelDownloadWorker(QThread):
    progress = Signal(int, str)
    result = Signal(bool, str)

    def __init__(self, missing: list[str]):
        super().__init__()
        self.missing = missing

    def run(self) -> None:
        for index, name in enumerate(self.missing):
            self.progress.emit(index, name)
            if not download_core_model(name):
                self.result.emit(False, name)
                return
        self.result.emit(True, "")


def _obtain_first_run_models() -> bool:
    missing = missing_core_models()
    if not missing:
        return True
    app = QApplication.instance() or QApplication(sys.argv)
    answer = QMessageBox.question(
        None, "Deep Live Cam – models",
        "Lần chạy đầu cần tải khoảng 800 MB model về thư mục dữ liệu của bạn. "
        "Các model InsightFace/inswapper chỉ được công bố cho nghiên cứu phi thương mại; "
        "hãy kiểm tra quyền sử dụng trước khi tải. Tiếp tục?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return False

    dialog = QProgressDialog("Đang chuẩn bị model...", "", 0, len(missing))
    dialog.setCancelButton(None)
    dialog.setWindowTitle("Deep Live Cam")
    dialog.setMinimumDuration(0)
    dialog.show()
    worker = ModelDownloadWorker(missing)
    outcome = {"ok": False, "name": ""}
    worker.progress.connect(lambda index, name: (
        dialog.setValue(index), dialog.setLabelText(f"Đang tải {name} ({index + 1}/{len(missing)})...")
    ))
    worker.result.connect(lambda ok, name: outcome.update(ok=ok, name=name))
    worker.start()
    while worker.isRunning():
        app.processEvents()
        worker.wait(50)
    app.processEvents()
    dialog.close()
    if not outcome["ok"]:
        QMessageBox.critical(None, "Deep Live Cam", f"Không tải được {outcome['name']}. Kiểm tra mạng rồi mở lại ứng dụng.")
    return outcome["ok"]


def main() -> None:
    prepare_desktop()
    if "--package-smoke-test" in sys.argv:
        print(f"[desktop] missing core models={missing_core_models()}")
        print(f"[desktop] data={DATA_DIR}")
        print(f"[desktop] models={MODELS_DIR}")
        print(f"[desktop] cache={CACHE_DIR}")
        print(f"[desktop] settings={SETTINGS_PATH}")
        print("[desktop] package smoke test OK")
        return
    desktop_app = QApplication.instance() or QApplication(sys.argv)
    if not _obtain_first_run_models():
        return
    import run
    run.core.run()


if __name__ == "__main__":
    main()
