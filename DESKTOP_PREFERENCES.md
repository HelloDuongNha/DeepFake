# Source history, saved preferences and feature contours

## Live orientation

The Dọc / Ngang button beside Live switches the preview immediately and saves
the choice for the next launch. Dọc keeps the existing portrait fill/crop.
Ngang sizes the window to the actual camera aspect ratio and fits the complete
frame without cropping or stretching, including after manual window resizing.
Camera capture, mask processing and upstream frame dimensions are unchanged.

## Source images and restart

Select a face once. The next desktop launch restores that image and its preview.
Recent faces opens a thumbnail album of up to 24 unique recently chosen paths,
most recent first. Double-click a thumbnail or press Use selected. Missing or
unreadable files are disabled. History stores file paths, not duplicate photos.
Clear history forgets the selected source and the album; original photos are
not deleted. Selecting a face again begins a new history.

Saved preferences include all current Options switches, enhancer/model choices,
Transparency, Sharpness, camera device index and main/Live window geometry.
Settings are validated and written atomically through modules/user_settings.py.
Missing/corrupt settings use defaults. Existing user settings are retained.
CLI/source and explicit environment overrides continue to take precedence.
run_mac.sh no longer injects DLC_ENHANCER=GPEN-256 on every launch, so a saved
None/GFPGAN/RestoreFormer choice is not overwritten. The existing explicit
Poisson-disable profile still applies when launching through that script.

Installed apps use their user data directory; source launches use
switch_states.json in the project, unless DLC_DATA_DIR is set. This change is
in the source app; existing .app/.exe packages have not been rebuilt.

## Feature contour options

- Natural eyes + Eyelid repair: copy a small eyelid rim from the current camera
  as well as the eyeball. This addresses the mismatch between camera eyes and
  generated lids. Camera eyelid shape is preserved; brows stay generated.
- Nose edge repair: suppress narrow lateral ridges in the native generated nose,
  block camera nose contours in microtexture transfer, and preserve the corrected
  contour after enhancement. It does not substitute the real camera nose. The
  native correction also works without an enhancer. See reports/nose-ridges-2026-10-15/README.md.
- Natural nose: remains the separate option for camera nose anatomy. When this
  is enabled, the model-nose repair switch is disabled because it is superseded.

Each repair can be switched off independently. No extra AI model or temporal
frame cache is added. The initial camera-nose-edge experiment changed nose
identity too strongly and was discarded in favor of the pre-enhancer reference.

### Forehead at turns

Forehead guard uses the current camera forehead, eyebrows and glabella when
turning, with a smooth pose transition. Frontal rendering stays unchanged. This
preserves actual camera brow shape at oblique angles; it is a fallback, not a
new source-identity reconstruction model. Nose and mouth processing stay intact.
The option is saved and can be disabled independently. See
reports/upper-face-2026-10-15/README.md for comparisons and validation.

## Verification, 2026-10-15

- Twelve focused unit tests passed: malformed/missing settings, exact roundtrip,
  bounded/deduplicated history, atomic failure, local contour support, current
  iris response, invalid geometry and existing natural-eye/nose/hand behavior.
- Actual Qt windows checked: select two source files, reselect/deduplicate,
  change sliders, enhancer=None, camera and repair switches, close and initialize
  again. Source preview/history/controls restored. Album and Clear checked;
  original source files remained on disk. The test used an isolated settings file.
  Clear/no-source also bypasses AI and returns exact current camera pixels;
  stale results from a previous source selection are discarded.
- Natural 256 + GPEN-256 compared on three raw camera-video frames (78,234,625)
  with the new source /Users/helloduongnha/Downloads/images.jpeg. The local repair
  passes took 3.5–5.7ms in this small CPU sample, excluding AI inference. This is
  not a sustained FPS benchmark. Artifacts are in reports/feature-contours-2026-10-15.
- CoreML compilation was restricted in the tool sandbox, so these image comparisons
  use CPU inference. They do not establish real-camera quality under the exact
  lighting in the user's screenshot. The image/model quality limitation remains;
  contour repair is not a new face-swap model.

A 14-frame replay through the actual processing worker with images.jpeg also completed.
