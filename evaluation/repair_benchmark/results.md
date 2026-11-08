# Repair benchmark

Sources: syn_01 masks=ground_truth_stand_in tracks=opencv_lk; syn_02 masks=ground_truth_stand_in tracks=opencv_lk; syn_03 masks=ground_truth_stand_in tracks=opencv_lk; syn_04 masks=ground_truth_stand_in tracks=opencv_lk; syn_05 masks=ground_truth_stand_in tracks=opencv_lk; syn_06 masks=ground_truth_stand_in tracks=opencv_lk. Errors: injected (focal, floor position, size, timing). Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).
Reprojection error is a fraction of the image diagonal.

| clip | loss (silhouette + tracks) | focal error | reprojection error | mask IoU | timing error (frames) | accepted |
|---|---|---|---|---|---|---|
| syn_01 | 0.147 → 0.042 | 15.7% → 0.1% | 1.5% → 0.3% | 0.644 → 0.938 | 3.000 → 0.000 | 8/10 |
| syn_02 | 0.554 → 0.083 | 10.9% → 0.1% | 7.2% → 0.7% | 0.285 → 0.869 | 6.000 → 0.500 | 13/18 |
| syn_03 | 0.322 → 0.059 | 15.9% → 0.2% | 3.7% → 0.2% | 0.357 → 0.961 | 4.000 → 0.000 | 6/10 |
| syn_04 | 0.498 → 0.156 | 13.0% → 0.3% | 2.7% → 0.6% | 0.281 → 0.868 | 2.000 → 0.000 | 16/18 |
| syn_05 | 0.628 → 0.183 | 17.4% → 0.2% | 3.7% → 0.7% | 0.376 → 0.844 | 4.000 → 0.000 | 21/26 |
| syn_06 | 0.286 → 0.029 | 14.9% → 0.3% | 3.8% → 0.4% | 0.565 → 0.957 | 3.500 → 0.000 | 18/18 |
