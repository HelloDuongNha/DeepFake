# Nose ridge correction — 2026-10-15

## Diagnosis

Compared original camera, native HyperSwap-1A output, paste-back without camera
microtexture, normal paste-back, GPEN, and the previous contour repair. The narrow
bright lateral ridges are already visible in the native generated nose. They do
not originate solely in GPEN or the final contour blend. Copying the pre-GPEN
nose therefore retained the defect. See stages.png (three different poses).

## Source changes

- Natural256Swapper: detect narrow positive lateral ridges in the native generated
  crop using dense landmarks and a horizontal contrast response. Fill only those
  small detected strips from neighboring generated pixels, with bounded per-channel
  correction. No camera nose texture is substituted. Skip invalid geometry/strong yaw.
- Camera microtexture: optional nose exclusion gate prevents the actual subject's
  alar edges being added over a narrower generated nose. Existing direct helper
  callers retain their default behavior; Live enables the gate through Nose edge repair.
- Existing final nose contour repair: retain the corrected pre-enhancer model
  contour, then check final remaining lateral ridges on frontal/mild-yaw frames.
- Keep using the existing Nose edge repair switch. Natural nose remains the separate
  camera-anatomy option and supersedes model nose correction. No new AI model or
  new dropdown added; history/preferences are retained.

## What was checked

- Three raw camera-video frames (526, 701, 965) from the user's 2026-10-07
  10-46-47.mov, selected at different measured nose/eye pitch ratios. Source is
  /Users/helloduongnha/Downloads/images.jpeg; Natural 256 + GPEN-256.
- comparison.png: left is the prior repair captured before these changes; right
  is the updated complete swap/GPEN/contour pipeline on the same frame.
  The bright ridge is less pronounced; some model-generated side contour remains.
  This is an improvement, not evidence of a completely reconstructed natural nose.
- Replayed 14 video frames through the real processing worker, all completed
  without losing the swap. This is video replay on CPU, not a physical-camera or
  sustained performance benchmark. No additional neural inference was introduced.
- 23 targeted tests passed, including bright-ridge reduction, exact dark-nostril
  preservation in the synthetic case, unchanged bridge/mouth outside the repair
  support, camera-nose residual blocking, and existing detail/eye/hand/settings tests.
- Also examined cleanup on both supplied screenshots. Those are already-swapped
  outputs and were used only as local filter probes, never as raw driving frames
  or proof of full-pipeline behavior.

The implementation reduces detected highlight ridges and duplicate texture;
base-model nose geometry and fine alar detail are still limited. Aggressive wide
inpainting and broad smoothing prototypes were discarded because they flattened
nose shading. No guarantee of eliminating all contour artifacts in every pose.
