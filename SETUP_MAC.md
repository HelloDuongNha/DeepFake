# Deep-Live-Cam on Apple Silicon

## Launch

Double-click `start.command` to open the original Deep-Live-Cam 2.1.5
interface, or run `./run_mac.sh` in Terminal. Select the source image and
camera in that interface, then use its **Live** button. The first model load
may take 10–30 seconds. macOS may ask for camera access.

The Mac profile starts at 640×360 and keeps the preview window resizable. The
GPEN options run every third live frame to reduce model load, but may still
reduce frame rate or make skin look too smooth. See [PROFILES.md](PROFILES.md)
for mask, blend, and Windows GPU controls. The swap model changes the face area;
it does not replace hairstyle or hair silhouette. For better results, use a
sharp source portrait in similar lighting and keep the webcam face well lit.

The original `run.sh` is also available if you prefer the upstream source
folder chooser: `./run.sh` opens the standard Deep-Live-Cam interface.

## OBS on macOS

1. Leave the Deep-Live-Cam preview window open.
2. In OBS, choose **Sources → + → macOS Screen Capture**.
3. In the source properties, set **Method → Window Capture** and choose the
   Deep-Live-Cam preview window. Resize or crop it in the OBS canvas.
4. If the picture is blank, grant OBS **Screen Recording** permission in
   macOS System Settings, then restart OBS. OBS also provides
   **OBS Studio → Review App Permissions**.
5. To use the OBS scene as a webcam in Zoom, Meet, or another app, click
   **Start Virtual Camera** in OBS and select **OBS Virtual Camera** there.

Deep-Live-Cam outputs a preview window; OBS captures that window. OBS does not
need to open the physical webcam separately for this workflow.

Passing `--source` to `run.py` enters headless file-processing mode, so `run.sh`
does not use that flag for live webcam mode.

## Models

The app downloads missing models when first needed. To download them manually,
use the following exact names and destinations:

| Purpose | Source | Destination |
| --- | --- | --- |
| Face swap: `inswapper_128.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx | `models/inswapper_128.onnx` |
| Face analysis: `1k3d68.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/1k3d68.onnx | `models/buffalo_l/1k3d68.onnx` |
| Face analysis: `2d106det.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/2d106det.onnx | `models/buffalo_l/2d106det.onnx` |
| Face analysis: `det_10g.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/det_10g.onnx | `models/buffalo_l/det_10g.onnx` |
| Face analysis: `genderage.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/genderage.onnx | `models/buffalo_l/genderage.onnx` |
| Face analysis: `w600k_r50.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/w600k_r50.onnx | `models/buffalo_l/w600k_r50.onnx` |
| Optional face enhancement: `gfpgan-1024.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan-1024.onnx | `models/gfpgan-1024.onnx` |
| Optional alternate face swap: `inswapper_128_fp16.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx | `models/inswapper_128_fp16.onnx` |

All six required files above were downloaded and verified against the sizes
expected by the project's model downloader. The README also links
`inswapper_128_fp16.onnx`, but the current Apple Silicon swapper
selects `inswapper_128.onnx` first. The FP16 model is an alternative when the
FP32 model is absent.

## Environment

Current upstream recommends Python 3.14 and supports Python 3.11 through 3.14;
Python 3.10 does not meet its current ONNX Runtime requirement. The environment
for this checkout is `.venv` with `onnxruntime==1.28.0` on Apple Silicon. It was
installed with a project-local `uv` and Python 3.14 because Homebrew was blocked
by an unaccepted Xcode license on this machine. Check
CoreML availability with:

```sh
.venv/bin/python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
```

The result should contain `CoreMLExecutionProvider`. The face swapper and face
analysis models were also loaded successfully with CoreML on this Mac. Actual
webcam frame rate depends on the camera and model operations.
