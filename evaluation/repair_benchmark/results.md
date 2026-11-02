# Repair benchmark

Masks: ground-truth stand-in. Point tracks: OpenCV Lucas-Kanade on the reference video. Errors: injected (focal, floor position, size, timing). Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).
Reprojection error is a fraction of the image diagonal.

| clip | loss (silhouette + tracks) | focal error | reprojection error | mask IoU | timing error (frames) | accepted |
|---|---|---|---|---|---|---|
| syn_01 | 0.147 → 0.061 | 15.7% → 0.1% | 1.5% → 0.6% | 0.644 → 0.894 | 3.000 → 2.000 | 7/8 |
| syn_02 | 0.554 → 0.153 | 10.9% → 0.1% | 7.2% → 2.1% | 0.285 → 0.715 | 6.000 → 3.000 | 12/14 |
| syn_03 | 0.322 → 0.144 | 15.9% → 0.2% | 3.7% → 1.5% | 0.357 → 0.729 | 4.000 → 4.000 | 5/8 |
| syn_04 | 0.498 → 0.154 | 13.0% → 0.3% | 2.7% → 0.3% | 0.281 → 0.883 | 2.000 → 0.500 | 12/14 |
| syn_05 | 0.628 → 0.238 | 17.4% → 0.2% | 3.7% → 0.9% | 0.376 → 0.770 | 4.000 → 2.000 | 17/20 |
| syn_06 | 0.286 → 0.083 | 14.9% → 0.3% | 3.8% → 0.8% | 0.565 → 0.876 | 3.500 → 2.000 | 14/14 |
