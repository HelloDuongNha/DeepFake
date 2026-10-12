"""Render the transparent vector icon for macOS and Windows packages."""

from pathlib import Path
import shutil

from PIL import Image
from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


ROOT = Path(__file__).resolve().parent


def main() -> None:
    renderer = QSvgRenderer(QByteArray((ROOT / "icon.svg").read_bytes()))
    iconset = ROOT / "DeepLiveCam.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(exist_ok=True)
    rendered = {}
    for size in (16, 32, 64, 128, 256, 512, 1024):
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        rendered[size] = image
    mapping = {
        "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
    }
    for name, size in mapping.items():
        target = iconset / name
        rendered[size].save(str(target))
    rendered[1024].save(str(ROOT / "DeepLiveCam.png"))
    Image.open(ROOT / "DeepLiveCam.png").save(
        ROOT / "DeepLiveCam.ico", sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)]
    )
    Image.open(ROOT / "DeepLiveCam.png").save(ROOT / "DeepLiveCam.icns", format="ICNS")


if __name__ == "__main__":
    main()
