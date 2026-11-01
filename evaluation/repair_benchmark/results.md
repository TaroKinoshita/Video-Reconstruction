# Repair benchmark

Masks: ground-truth stand-in. Errors: injected (focal, floor position, size, timing). Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).
Reprojection error is a fraction of the image diagonal.

| clip | silhouette loss | focal error | reprojection error | mask IoU | timing error (frames) | accepted |
|---|---|---|---|---|---|---|
| syn_01 | 0.122 → 0.032 | 15.7% → 0.1% | 1.5% → 0.6% | 0.644 → 0.894 | 3.000 → 2.000 | 7/8 |
| syn_02 | 0.387 → 0.173 | 10.9% → 10.9% | 7.2% → 2.5% | 0.285 → 0.674 | 6.000 → 4.000 | 10/14 |
| syn_03 | 0.225 → 0.075 | 15.9% → 15.9% | 3.7% → 1.4% | 0.357 → 0.786 | 4.000 → 1.000 | 6/8 |
| syn_04 | 0.363 → 0.054 | 13.0% → 0.3% | 2.7% → 0.3% | 0.281 → 0.883 | 2.000 → 0.500 | 12/14 |
| syn_05 | 0.404 → 0.155 | 17.4% → 0.5% | 3.7% → 1.1% | 0.376 → 0.742 | 4.000 → 2.667 | 16/20 |
| syn_06 | 0.239 → 0.101 | 14.9% → 14.9% | 3.8% → 1.6% | 0.565 → 0.818 | 3.500 → 2.500 | 12/14 |
