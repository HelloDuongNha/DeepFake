# Deep-Live-Cam on Apple Silicon

## Launch

Double-click `start.command` to open the original Deep-Live-Cam 2.1.5
interface, or run `./run_mac.sh` in Terminal. Select the source image and
camera in that interface, then use its **Live** button. The first model load
may take 10–30 seconds. macOS may ask for camera access.

The Mac profile starts at 640×360 and keeps the preview window resizable. The
GPEN-256 runs on every live frame to avoid alternating sharp/cached looks, but may
reduce frame rate or make skin look too smooth. See [PROFILES.md](PROFILES.md)
for mask, blend, and Windows GPU controls. The swap model changes the face area;
it does not replace hairstyle or hair silhouette. For better results, use a
sharp source portrait in similar lighting and keep the webcam face well lit.

The original `run.sh` is also available if you prefer the upstream source
folder chooser: `./run.sh` opens the standard Deep-Live-Cam interface.

The Mac preset uses **GPEN-256**. Larger enhancers may lower FPS or make
skin look too smooth. To try GPEN-512 from Terminal:

```sh
DLC_ENHANCER=GPEN-512 DLC_ENHANCER_INTERVAL=3 DLC_DETAIL_STRENGTH=0.35 ./run_mac.sh
```

`DLC_DETAIL_STRENGTH` restores fine camera texture after enhancement, while
`DLC_MASK_BLUR=1.5` and `DLC_MASK_EROSION=4` control the landmark-aligned swap
edge. See [PROFILES.md](PROFILES.md) for the full tuning table. The source
portrait is passed to InsightFace unchanged and the swap mask has no fixed
side crop. GPEN paste-back covers the complete aligned crop and uses a border
equal to 1/16 of that crop, so the transition scales with near and distant faces.
`DLC_FILM_GRAIN=0.35` adds camera-matched microtexture
after GPEN/GFPGAN; `DLC_COLOR_MATCH=1` applies masked LAB tone matching in the
enhancer and before Poisson when that blend mode is enabled. The live swap no
longer fades out at a preset yaw angle. Between detector hits, optical flow
tracks the face landmarks; small distant faces are refreshed by the detector
every frame, and after a brief detector loss the last face is held for up to
1.25 seconds.

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
| Optional face enhancement: `GPEN-BFR-256.onnx` | https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx | `models/GPEN-BFR-256.onnx` |
| Optional face enhancement: `GPEN-BFR-512.onnx` | https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-512.onnx | `models/GPEN-BFR-512.onnx` |
| Optional hairline parsing: `face_parsing_resnet18.onnx` | https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx | `models/face_parsing_resnet18.onnx` |
| Optional alternate face swap: `inswapper_128_fp16.onnx` | https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx | `models/inswapper_128_fp16.onnx` |

The six core files (the swap model plus the five `buffalo_l` files) are
verified against the sizes expected by the project's model downloader. The README also links
`inswapper_128_fp16.onnx`, but the current Apple Silicon swapper
selects `inswapper_128.onnx` first. The FP16 model is an alternative when the
FP32 model is absent.

The main `./run_mac.sh` enables the 19-class parser above and keeps GPEN-256.
The parser only lets skin and inner facial features receive swapped/enhanced
pixels. The source image's outer cheek strips are also removed before identity
extraction and at paste-back. Parsing adds a 512px inference pass and may
substantially lower FPS. To use a model stored elsewhere, set
`DLC_HAIR_PARSER_MODEL=/absolute/path/model.onnx`.

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
