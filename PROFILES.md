# Live profiles

## macOS Apple Silicon

Run `./run_mac.sh` or double-click `start.command` to open the original
Deep-Live-Cam interface. The Mac profile requests 640×360 at 60 fps, CoreML,
alpha blending, and GPEN-256 on every frame by default. This is a balanced preset, not a
guaranteed output frame rate. The actual camera mode and processing FPS depend
on hardware.

To disable GPEN-256 if the machine cannot sustain the desired frame rate:

```sh
DLC_ENHANCER=None ./run_mac.sh
```

GPEN/GFPGAN can be selected in the Face Enhancer control. GPEN-256 runs on
every live frame; its last restored crop is used only after a transient model
failure. GPEN-512/GFPGAN can still cache when configured with a longer
interval. If FPS is low, leave Face Enhancer set to None and keep the 640×360
capture profile.

The Mac script enables the 19-class parser in
`models/face_parsing_resnet18.onnx` while keeping GPEN-256. It excludes hair,
ears, neck and background from the pasted face. The outer 15% on each side
of the source portrait is removed with a strict radial mask before identity
extraction. Paste-back uses a separate 18% lateral cheek erosion. No yaw threshold disables
the swap; brief detector misses use optical-flow tracking of the last face.
Face parsing adds another inference pass and can lower FPS.

## Windows NVIDIA

On the Windows computer, install Python 3.11–3.14, FFmpeg, NVIDIA drivers,
and the CUDA/cuDNN runtime compatible with the version of `onnxruntime-gpu`
in `requirements.txt`. Create a new Windows virtual environment in this
checkout and install requirements:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
run_windows.bat
```

The Windows profile chooses CUDA when an NVIDIA GPU and CUDA provider are
available. It runs GPEN-512 every frame and enables Poisson blending. Select
GFPGAN or GPEN-256 from the Face Enhancer menu, or override the default before
launch, for example `set DLC_ENHANCER=GFPGAN`. Each enhancer is a separate
choice; they are not all stacked in one pass.

TensorRT is optional. When its matching runtime is installed and ONNX Runtime
lists `TensorrtExecutionProvider`, set `DLC_PREFER_TENSORRT=1` before running
the batch file. The app registers TensorRT first, CUDA second, and caches
compiled TensorRT engines under `.cache/tensorrt`. ONNX Runtime documentation:
[CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
[TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

`run_windows.bat` does not change GPU or VRAM clock frequencies. Clock control
belongs to GPU driver/vendor tools and is not a safe application-level tuning
knob. The script lets ONNX Runtime manage GPU memory and caches TensorRT engines.

CodeFormer is not bundled or wired into this ONNX pipeline. Its official
implementation uses a separate PyTorch model and dependencies. The supported
high-resolution choices in this checkout are GPEN-512 and GFPGAN.

## Edge controls on both platforms

Set these environment variables before launch to tune the paste-back mask:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLC_MASK_BLUR` | `1.5` | Gaussian sigma in the aligned 128 px face crop; smaller means a narrower feather. Range: 0–16. |
| `DLC_MASK_EROSION` | `4` | Pixels to erode the aligned mask; larger keeps the swapped area farther inside the face. Range: 0–16. |
| `DLC_DETAIL_STRENGTH` | `0.35` | High-frequency camera texture mixed back after GPEN/GFPGAN. `0` disables it; `1` is strongest. |
| `DLC_FILM_GRAIN` | `0.35` | Adaptive fine grain added after enhancement. It measures camera noise and uses a 2–4 level field; `0` disables it. |
| `DLC_HAIRLINE_GUARD` | `0.16` | Forehead guard used by the legacy GFPGAN paste-back path; GPEN uses its complete aligned crop. |
| `DLC_HAIR_PARSING` | `1` in `run_mac.sh` | Run the ONNX parser as a subtractive hair-only guard. Other labels cannot punch holes in the face. |
| `DLC_HAIR_PARSER_MODEL` | `models/face_parsing_resnet18.onnx` | Override the parser model path. |
| `DLC_DET_THRESHOLD` | `0.40` | Face detector confidence threshold. The lower default helps retain small, distant webcam faces. |
| `DLC_COLOR_MATCH` | `1` | Match LAB colour statistics in the aligned enhancer crop and before Poisson blending. Set `0` to disable. |
| `DLC_BLEND_MODE` | `alpha` on Mac, `poisson` on Windows script | `alpha` is faster; `poisson` uses OpenCV seamlessClone for color-adaptive blending. |
| `DLC_ENHANCER_INTERVAL` | `1` on Mac and Windows | Run GPEN-512/GFPGAN every N live frames if changed. GPEN-256 always runs on each live frame to prevent alternating appearance. |
| `DLC_POISSON_MAX_LAB_DISTANCE` | `32` | If average LAB face colour differs more than this, use alpha blending for that frame. |

Start with the defaults. If a halo remains, try `DLC_MASK_BLUR=1` and
`DLC_MASK_EROSION=5`. If the mask cuts into the cheeks, reduce erosion. If
hair is touched at the forehead, increase `DLC_HAIRLINE_GUARD` to `0.20`.
Poisson can help with skin-tone seams but is more expensive and cannot
guarantee a perfect match.

## Detail preservation

GPEN-512 and GFPGAN are loaded from `models/` when selected in the original
Face Enhancer control. The restoration result is blended with a configurable
high-frequency residual from the live camera crop, which keeps fine eyebrow
hairs and expression lines without reintroducing broad blur. For a sharper
result use `DLC_DETAIL_STRENGTH=0.45`; lower it to `0.15` if the camera is
noisy. The feature is also applied when an enhancer is cached between live
frames.

The live five-point detector uses the aligned hairline guard. When 106-point
landmarks are available, `face_masking.create_hairline_safe_mask` adds an
eyebrow/chin-aware skin mask. The parser model lives in the ignored `models/`
directory and is loaded when the main Mac script runs.

Mouth and eye masks use separate landmark groups. For the InsightFace 106-point
model the outer mouth is `52:64` and the two eyes are `33:43`/`87:97`; for a
68-point face they fall back to the conventional `48:68` mouth and `36:48`
eyes. The mouth slider is clipped below the lowest eye point, so increasing it
cannot erase the eyes or expand their bounding box.
