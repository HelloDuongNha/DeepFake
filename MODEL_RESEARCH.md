# Face model comparison — 2026-10-12

The desktop keeps **Natural 256 (HyperSwap 1A)** and **INSwapper 128**, and now
also offers **HyperSwap 1B** and **HyperSwap 1C**. AlphaFace was removed after
the user's Live comparison.
Natural 256 remains the default. Stop Live before selecting another engine.

## Real model trials on this Mac

The supplied older-man source portrait and the same raw camera frame were used
for every render. Models ran through CoreML/CPU with the existing paste-back,
GPEN-256 at 40%, Natural mouth and Eye detail at sharpness 0.8. Release CRC32
hashes were verified. Each measurement is the median of three warm renders,
excluding detection, model/session startup, capture and UI.

| Model | Native face size | Warm render | Visual observation on this sample |
| --- | --- | --- | --- |
| HyperSwap 1A | 256×256 | 331 ms | Accepted baseline; softer age cues |
| HyperSwap 1B | 256×256 | 297 ms | Close to 1A; no clear frontal detail advantage |
| HyperSwap 1C | 256×256 | 280 ms | More visible under-eye/cheek folds; worth a video comparison |
| AlphaFace | 256×256 | 752 ms | Different eyes/nose, softer nose shadows; substantially slower here |

These timings are not Live FPS or a statistically reliable speed ranking of
the three HyperSwap variants. The comparison is one identity pair and one
expression. It does not establish blink/mouth stability or quality at other
angles. 1C is the most interesting next candidate for this older source face,
but its stronger age cues can also exaggerate shadows. Keeping 1A as the default
avoids replacing the result the user already accepted without video evidence.

Artifacts in `.tmp/model-research/`:

- `comparison.png`: four renders from the same input.
- `1a.png`, `1b.png`, `1c.png`, `alpha.png`: full-size renders.
- `results.json`, `alpha-results.json`: timings and release checksums.
- `evaluate.py`, `evaluate_alpha.py`: local evaluation scripts.
- `ui.png`: offscreen Qt rendering of the adjusted interface.

The three additional weights have been copied into the ignored user model
directory on the development Mac. Other installations download only the chosen
engine. Each adapter verifies the registered SHA-256 before session creation.
Real production-adapter runs applied all three trial models successfully;
AlphaFace is no longer selectable.

## Other directions researched

- **Ghost**: the [authors' implementation](https://github.com/ai-forever/ghost)
  is available. FaceFusion's exported 256px variant needs an identity embedding
  converter in addition to the swap model. Not run in this comparison.
- **HifiFace**: the [paper](https://arxiv.org/abs/2106.09965) investigates 3D shape
  and identity; [maum-ai's implementation](https://github.com/maum-ai/hififace)
  explicitly identifies itself as unofficial. FaceFusion's downloadable entry
  is also named `hififace_unofficial_256` and needs a separate converter. A
  paper's 512px experiment does not establish a ready-to-use 512px Mac model.
- **UniFace**: FaceFusion provides a 256px source-image-based model with a
  different alignment contract. It needs its own adapter; not tested here.
- **SimSwap 512**: omitted because the user previously rejected its changed
  facial appearance.
- **Pixel boost**: increasing render dimensions is not the same as changing the
  learned native face resolution. The prior INSwapper boost experiment was
  slower and did not sufficiently improve this subject's eyes/brows.

Model dimensions, input contracts, weight locations and license metadata were
checked against the [FaceFusion model registry](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py).
HyperSwap metadata lists ResearchRAIL; AlphaFace lists Non-Commercial. The
research downloads do not change those licenses.

## UI changes and verification

The dropdowns now use compact 38px rows. The centered source thumbnail is 128px;
target selection and the swap arrow are removed from the camera interface.
Camera/Live controls are pinned outside the scrollable options, so Live remains
visible in the default 640×820 window. GPEN-512 is removed from the desktop menu;
a saved GPEN-512 selection migrates to GPEN-256. Natural mouth remains.

**Natural eyes**, enabled by default, restores the current camera eye aperture
inside the 106-point eyelid contours after restoration. Its feather runs inward,
so the camera iris cannot spill outside that contour. The surrounding generated
lids/brows stay swapped. This preserves the user's iris color and gaze, rather
than transferring the source portrait's eyeballs. Invalid or closed apertures
are skipped. Tracking accuracy still limits the result at strong angles.
The new helper passed boundary/current-frame tests; it was visually compared
on the supplied frontal image. A live side-angle session has not been verified.

Qt offscreen checks exercised model choices and pinned Live geometry.
The suite now passes 135 tests. Packaged apps were not rebuilt.

## Further enhancer research

[CodeFormer](https://github.com/sczhou/CodeFormer) has an explicit fidelity weight:
higher values favor fidelity to the input; lower values favor restoration.
[RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) is another restoration
architecture. FaceFusion publishes 512px ONNX entries for CodeFormer,
RestoreFormer++ and GFPGAN 1.4 in its
[enhancer registry](https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_enhancer/core.py).
CodeFormer remains untested here. RestoreFormer++ has now been tested and added
as an optional quality choice; GPEN-256 remains the Live default.

`eyes-comparison.png` shows the four trials with Natural eyes; `final-ui.png`
shows the compact interface. Earlier images/measurements remain available.

## Latest measured comparison and Natural nose

Ten consecutive frames from the supplied original video were processed with
each HyperSwap variant, GPEN-256, Natural eyes/mouth and Natural nose. All three
applied 10/10 swaps. Warm median rendering times excluding detection/capture/UI
were 1A: **199 ms**, 1B: **191 ms**, 1C: **195 ms**. This small difference does not
establish a reliable speed winner; it also excludes the first CoreML compile.
The user's Live preference and frontal appearance favor keeping Natural 256
(HyperSwap 1A) as the main choice. The variants do not increase native resolution.

On the supplied still frame, GPEN-256 took **269 ms** through swap/restoration,
and RestoreFormer++ **556 ms** with the same 40% restoration mix. Both produced
nonzero output changes and passed the restoration guard. These are offline
render times, not Live FPS. RestoreFormer++ is not claimed to be faster/better.
Its published ONNX file (294,264,232 bytes) was release-CRC checked; the production
loader verifies SHA-256 `164818b70d1745da4108a45c9263b05452236d14ac0ffab2202af0ac4a438195`.

Natural nose uses the camera's lower nose contour from landmarks 74–86, matches
broad rim color and feathers inward. It changes that anatomy to the camera
subject's own nose, so it defaults off and can be compared independently.
The complete 1A/GPEN-256 pipeline with nose enabled applied 12 consecutive video
frames; boundary/current-frame tests also pass. A long real Live session and
strong profile angles have not been validated.

The UI is now 640×700 by default. Group padding is reduced, cards no longer
stretch their internal rows, and the status/footer area is removed from view.
Live remains pinned. Qt tests verified new-enhancer processor loading/removal,
the nose toggle, compact cards and the removed AlphaFace/footer controls.

Review `.tmp/nose-review/comparison.png`, `nose-closeup.png`, `sequence.png`,
`ui.png`, `measurements.json` and `variant-timings.json` for the actual outputs.

## Nose edge, foreground occlusion and compact restoration trial

The Natural nose contour now extends slightly below the nostril shadow and
uses a narrower inward feather. Its lower boundary no longer crosses the
shadow itself. A regression test checks that dark lower-nostril pixels are
retained rather than blended halfway with the generated face.

**Hand protection** (default on, independently switchable) runs XSeg 2 on the
current raw camera crop and restores foreground pixels after all restorers and
Natural eyes/nose/mouth passes. It does not reuse a previous hand image. The
[FaceFusion occlusion registry](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
publishes the BGR NHWC input contract and DeepFaceLab/GPL-3.0 metadata. The
70,324,286-byte weights were release-CRC checked, and the runtime verifies SHA-256
`cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042`.
XSeg 1/2/3 were tried; 2 found more of the finger than 1 in the supplied processed
screenshot. Screenshot alignment was manually annotated where the generic
leftmost detector could not detect its heavily cropped face.

The new **real raw hand video**, `2026-10-12 15-41-00.mov`, contains 1,640 frames
at 1920×1080/60fps. The camera's largest-face detection path was used for the
offline comparison (the generic leftmost selector otherwise selects the poster
in the background). Twenty-three selected frames across the clip, including a
series four frames apart around the nose crossing, all applied the swap. Six
manually selected 5×5 finger patches in two rendered comparisons differed from
the raw camera before protection and matched it exactly afterward. These probes
are local evidence, not a complete segmentation accuracy measurement. Warm
protection overhead had a 44ms median on those frames, excluding detection.
No claim is made that every occlusion or a long Live session is solved.

The [Real-ESRGAN general x4v3 compact model](https://github.com/xinntao/Real-ESRGAN)
was downloaded from the authors' release and reconstructed as an ONNX graph
following its SRVGG architecture, without installing PyTorch into the app.
The experiment's restricted tensor loader accepts only the required storage,
OrderedDict and tensor reconstruction entries. The published checkpoint is
about 4.7MB. At a 256px restoration crop, the local full render trial measured
GPEN-256 at 318ms and compact SR at 304ms; the difference is small and the compact
result did not show a convincing eye/brow detail improvement. Using a larger
restoration canvas was slower (369ms). This candidate is not added to the app
menu and is not claimed to surpass GPEN-256. RestoreFormer++ remains optional;
GPEN-256 is still the practical Live recommendation from the trials so far.

Review `.tmp/occlusion-review/hand-frame-894.png` and `hand-frame-1341.png`
(camera / before / after), `real-hand-results.json`, `finger-probes.json`,
`nose-before-after.png`, `enhancer-comparison.png` and `ui.png`. The current
suite passes 138 tests; Qt checks verified the new toggle and pinned Live.
