# Oblique forehead and glabella repair

The supplied screenshot shows an upper forehead seam plus distorted skin/brow
texture between the eyebrows. On three previously supplied camera recordings,
oblique samples reproduce upper-face geometry/colour artifacts in the swap,
before GPEN. Restricting the mask or protecting a generated eyebrow as a separate
island does not repair the model's distorted upper-face texture.

## Change

Added the optional, default-on **Forehead guard** in the empty Options cell beside
Hand protection. It uses the current unprocessed camera forehead, eyebrows and
skin between brows during turns. It fades in continuously with estimated yaw;
frontal frames stay unchanged. Pose is measured in the facial eye axis, so pure
roll is not mistaken for yaw. The support follows that axis, protects eye
apertures, and stops above the eyes/nose. Broad colour matching is restricted to
the transition. No cached frame or additional AI inference is used.

A prototype that retained generated brows inside separate masks left visible
islands, so it was discarded. The final guard keeps a continuous camera upper
face when oblique. This intentionally retains the user's actual brow shape at
turns, rather than claiming to reconstruct the source portrait's brows. Turn
Forehead guard off if that identity tradeoff is unsuitable.

The pass runs after nose/teeth/eye repairs and before foreground/hand protection.
The previously accepted nose repair code was not changed. The switch is saved
through the existing preferences/history system. No extra row or dropdown was
added to the Options layout.

## Verification

- compared three oblique samples from the supplied recordings (10-07 10-46-47,
  10-12 16-00-45, and 10-12 15-41-00), with the current images.jpeg source and
  Natural 256 + GPEN-256. See comparison.png: camera / previous / guarded.
- Lower-face pixels below the eye line plus a small safety margin are exactly
  unchanged in all three direct comparisons. The forehead seam and brow islands
  are visibly reduced by restoring the current camera region.
- The actual Live worker completed nine frames spanning these three poses with
  the guard enabled. This is replay, not a physical-camera trial.
- Tested frontal bypass, invalid/zero/NaN geometry, current-camera response,
  preservation of eyes/nose/mouth, pure-roll bypass and a continuous yaw ramp.
- Qt toggle and restart persistence verified with an isolated settings file;
  history/sliders/model/enhancer and the existing controls still restore.
- Direct guard calls took about 12–21ms on these large sample frames, excluding
  AI inference. These are isolated CPU measurements, not sustained camera FPS.

Tests and comparisons used CPU inference inside the tool sandbox. No guarantee
of a perfect source-identity reconstruction at profile angles is implied; this
is an explicit current-camera fallback for the upper-face failure.
