# Repair benchmark

Sources: syn_01 masks=ground_truth_stand_in tracks=opencv_lk; syn_02 masks=ground_truth_stand_in tracks=opencv_lk; syn_03 masks=ground_truth_stand_in tracks=opencv_lk; syn_04 masks=ground_truth_stand_in tracks=opencv_lk; syn_05 masks=ground_truth_stand_in tracks=opencv_lk; syn_06 masks=ground_truth_stand_in tracks=opencv_lk. Errors: injected (focal, floor position, size, timing). Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).
Reprojection error is a fraction of the image diagonal.

| clip | loss (silhouette + tracks) | focal error | reprojection error | mask IoU | timing error (frames) | accepted |
|---|---|---|---|---|---|---|
| syn_01 | 0.147 → 0.041 | 15.7% → 0.1% | 1.5% → 0.2% | 0.644 → 0.925 | 3.000 → 1.000 | 6/8 |
| syn_02 | 0.554 → 0.081 | 10.9% → 0.1% | 7.2% → 0.8% | 0.285 → 0.858 | 6.000 → 0.500 | 12/14 |
| syn_03 | 0.322 → 0.066 | 15.9% → 0.2% | 3.7% → 0.3% | 0.357 → 0.941 | 4.000 → 1.000 | 4/8 |
| syn_04 | 0.498 → 0.235 | 13.0% → 0.3% | 2.7% → 0.7% | 0.281 → 0.714 | 2.000 → 0.000 | 12/14 |
| syn_05 | 0.628 → 0.196 | 17.4% → 0.2% | 3.7% → 0.9% | 0.376 → 0.822 | 4.000 → 0.333 | 18/20 |
| syn_06 | 0.286 → 0.038 | 14.9% → 0.3% | 3.8% → 0.6% | 0.565 → 0.942 | 3.500 → 0.000 | 13/14 |
