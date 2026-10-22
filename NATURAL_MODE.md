# Natural 256 camera rendering

For source launches (`run.py` or `run_mac.sh`), choose **Face model: Natural 256**,
**Face Enhancer: GPEN-256**, and keep **Natural teeth** enabled. Stop Live before
changing models. INSwapper 128 remains available in the same selector.
HyperSwap 1B/1C are also selectable; AlphaFace has been removed after the user's
comparison. Natural eyes preserves the
camera iris/gaze inside the tracked eyelids, while brows and surrounding skin
remain swapped. It can be disabled independently for comparison.
Natural nose is opt-in: it restores the current camera nose tip/nostrils, rather
than retaining the source portrait's lower nose shape. RestoreFormer++ is now
selectable as a quality experiment, but measured roughly twice the processing
time of GPEN-256 on the supplied still image. GPEN-256 remains recommended for Live.
Hand protection is enabled by default and restores current camera foreground
after all face processing using XSeg 2. It adds inference time (44ms median in
the latest raw hand-video sample) and can be disabled for comparison. Natural
nose now extends its feather below the nostril shadow to reduce the soft seam.
Natural mouth is now labeled Natural teeth; its saved setting is unchanged.
`DLC_SWAP_MODEL=inswapper_128` explicitly selects the previous engine at startup.

Natural 256 uses HyperSwap-1a's native 256px output and the existing tracker,
anatomical mask, color matching and paste-back. GPEN contributes 40% after a
successful native swap. The Mask switch replaces Eye detail. OFF shows the original camera and bypasses
AI processing; ON resumes the selected model. Eye detail is removed from Live. Camera
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
