"""Create a clean source-only ZIP to transfer the Windows build to a PC."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "dist" / "DeepLiveCam-Windows-build-kit.zip"
TOP = "DeepLiveCam-Windows-build-kit"

TOP_LEVEL = (
    "desktop_entry.py", "run.py", "requirements.txt", "LICENSE", "README.md",
)
PACKAGING = (
    "DeepLiveCam.spec", "DeepLiveCam.iss", "DeepLiveCam.ico", "icon.svg",
    "make_icons.py", "requirements-build.txt", "build_windows.ps1",
    "sign_windows.ps1", "README.md", "WINDOWS_BUILD_KIT.md",
)


def members() -> list[Path]:
    files = [ROOT / name for name in TOP_LEVEL]
    files.extend((ROOT / "modules").rglob("*.py"))
    files.extend((ROOT / "locales").glob("*.json"))
    files.extend(ROOT / "packaging" / name for name in PACKAGING)
    return sorted(set(files))


def main() -> None:
    paths = members()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing build inputs: {missing}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(DESTINATION, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, f"{TOP}/{path.relative_to(ROOT).as_posix()}")
    print(f"Created {DESTINATION} ({len(paths)} source/config files)")


if __name__ == "__main__":
    main()
