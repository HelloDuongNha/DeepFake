# Live profiles

## macOS Apple Silicon

Run `./run_mac.sh` or double-click `start.command` to open the original
Deep-Live-Cam interface. The Mac profile requests 640×360 at 60 fps, CoreML,
alpha blending, and no enhancer by default. This is a speed preset, not a
guaranteed output frame rate. The actual camera mode and processing FPS depend
on hardware.

To try GPEN-256 every third frame from Terminal:

```sh
DLC_ENHANCER=GPEN-256 DLC_ENHANCER_INTERVAL=3 ./run_mac.sh
```

GPEN/GFPGAN can be selected in the Face Enhancer control. A cached enhanced
face is aligned to the current face position on skipped frames, so the entire
camera frame is never frozen. If FPS is still low, leave Face Enhancer set to
None and keep the 640×360 capture profile.

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
| `DLC_HAIRLINE_GUARD` | `0.16` | Fraction of the aligned crop protected above the forehead. Increase if hair is being touched; range 0–0.35. |
| `DLC_BLEND_MODE` | `alpha` on Mac, `poisson` on Windows script | `alpha` is faster; `poisson` uses OpenCV seamlessClone for color-adaptive blending. |
| `DLC_ENHANCER_INTERVAL` | `3` on Mac, `1` on Windows script | Run an enabled enhancer every N live frames. File processing always runs every frame. |

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

The live five-point detector uses the aligned hairline guard for speed. When
106-point landmarks are available, `face_masking.create_hairline_safe_mask`
adds an eyebrow/chin-aware skin mask before paste-back. A separate BiSeNet
hair parser is intentionally not bundled: it would add another model and
inference pass on every Mac frame, reducing the smoothness target.
