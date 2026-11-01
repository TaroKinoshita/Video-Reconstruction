# Repair benchmark

Masks: ground-truth stand-in. Errors: injected (focal, floor position, size, timing). Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).
Reprojection error is a fraction of the image diagonal.

| clip | silhouette loss | focal error | reprojection error | mask IoU | timing error (frames) | accepted |
|---|---|---|---|---|---|---|
| syn_01 | 0.346 → 0.137 | 15.7% → 7.4% | 1.5% → 0.8% | 0.644 → 0.856 | 3.000 → 2.000 | 7/8 |
| syn_02 | 0.708 → 0.277 | 10.9% → 4.6% | 7.2% → 2.1% | 0.285 → 0.718 | 6.000 → 3.000 | 14/14 |
| syn_03 | 0.623 → 0.172 | 15.9% → 8.2% | 3.7% → 0.9% | 0.357 → 0.796 | 4.000 → 2.000 | 8/8 |
| syn_04 | 0.705 → 0.448 | 13.0% → 13.9% | 2.7% → 1.9% | 0.281 → 0.521 | 2.000 → 2.000 | 11/14 |
| syn_05 | 0.641 → 0.206 | 17.4% → 1.7% | 3.7% → 1.0% | 0.376 → 0.764 | 4.000 → 2.000 | 17/20 |
| syn_06 | 0.437 → 0.207 | 14.9% → 14.3% | 3.8% → 1.5% | 0.565 → 0.790 | 3.500 → 2.500 | 13/14 |
