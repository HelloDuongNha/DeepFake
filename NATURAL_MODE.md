# Natural 256 camera rendering

For source launches (`run.py` or `run_mac.sh`), choose **Face model: Natural 256**,
**Face Enhancer: GPEN-256**, and keep **Natural mouth** enabled. Stop Live before
changing models. INSwapper 128 remains available in the same selector.
`DLC_SWAP_MODEL=inswapper_128` explicitly selects the previous engine at startup.

Natural 256 uses HyperSwap-1a's native 256px output and the existing tracker,
anatomical mask, color matching and paste-back. GPEN contributes 40% after a
successful native swap. Eye detail now applies bounded local contrast to the
current eyes/brows; Live no longer runs a second GFPGAN eye restoration. Camera
microtexture is reduced to avoid bringing back the camera subject's features.
Tiny cheek marks detected in the camera but absent from the aligned source
portrait receive local inpainting; eyes, nostrils and lips are excluded. This
heuristic is not a semantic skin analysis and may miss or misclassify marks.

## Model provenance

The first use downloads `hyperswap_1a_256.onnx` (402,742,682 bytes) into the existing
user model directory. Its SHA-256 is checked before opening an inference session:

`c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add`

- [Official weights](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.3.0)
- [Official preprocessing/model registry](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py)

The published model metadata identifies the license as **ResearchRAIL**. This
integration does not change the weights' license. If loading fails, the log
reports the error and attempts the existing INSwapper fallback.

## Validation and limits

On the development Mac, the real CoreML models were tested with the supplied
source portrait, raw camera image and 12 consecutive frames from the raw video.
All 12 video frames applied the swap. Warm processing in that sample took
185–215 ms per frame, excluding capture/detection/UI; this is **not Live FPS**.
The first model initialization includes CoreML compilation and is much slower.
The Qt model selector and its protection against switching during Live were
also exercised in offscreen mode. The automated suite passes 131 tests.

Review artifacts are in `.tmp/natural-face-review/`: `before-after.png` uses the
same source and raw frame for both pipelines, `sequence.png` shows the consecutive
video frames, and `measurements.json` records the offline processing times.

This remains a 256px face model. It cannot reproduce every pore, wrinkle, age cue
or hairstyle of the full-resolution portrait. Close-up rendering and long Live
sessions still need visual evaluation. Existing packaged apps have not been
rebuilt by this change.
