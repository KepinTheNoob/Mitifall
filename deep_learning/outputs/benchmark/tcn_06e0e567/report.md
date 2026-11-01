# Benchmark: ml_inference vs deep_learning (TCN)

- Windows: 9683 test windows over 19 LOSO folds (2689 Fall / 6994 ADL)
- Soft-fall subset: impact windows of fall recordings with peak |a| <= 5.04 g (140 windows)
- Deep model: tcn, channels ax, ay, az, acc_mag, tilt_deg, ensemble size 1, median best epoch 5

| Model | Accuracy | F1 macro | F1 Fall | Sensitivity | Specificity | ROC-AUC | PR-AUC | Soft-fall sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ml_inference RF @ 0.30 (deployed) | 0.842 | 0.816 | 0.747 | 0.837 | 0.844 | 0.906 | 0.854 | 0.914 |
| ml_inference RF @ val-tuned | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | 0.854 | 0.771 |
| deep_learning TCN @ val-tuned | 0.862 | 0.833 | 0.763 | 0.799 | 0.887 | 0.926 | 0.861 | 0.843 |

## DL minus RF, subject-level bootstrap (95 % interval)

vs **ml_inference RF @ 0.30 (deployed)**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | +0.020 | [-0.013, +0.046] | no clear difference |
| F1 macro | +0.017 | [-0.020, +0.046] | no clear difference |
| F1 Fall | +0.017 | [-0.035, +0.058] | no clear difference |
| Sensitivity | -0.036 | [-0.104, +0.022] | no clear difference |
| Specificity | +0.041 | [-0.002, +0.073] | no clear difference |
| ROC-AUC | +0.020 | [-0.005, +0.040] | no clear difference |
| PR-AUC | +0.007 | [-0.034, +0.041] | no clear difference |
| Soft-fall sens. | -0.068 | [-0.162, +0.023] | no clear difference |

vs **ml_inference RF @ val-tuned**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | -0.016 | [-0.048, +0.009] | no clear difference |
| F1 macro | -0.014 | [-0.053, +0.015] | no clear difference |
| F1 Fall | -0.016 | [-0.072, +0.024] | no clear difference |
| Sensitivity | +0.023 | [-0.045, +0.079] | no clear difference |
| Specificity | -0.030 | [-0.067, -0.002] | RF higher |
| ROC-AUC | +0.020 | [-0.005, +0.040] | no clear difference |
| PR-AUC | +0.007 | [-0.034, +0.041] | no clear difference |
| Soft-fall sens. | +0.072 | [-0.041, +0.171] | no clear difference |

## Same operating point (ROC curves compared at the forest's threshold)

The deep model's threshold is read off the pooled test ROC curve to match the forest's specificity (or sensitivity). A curve comparison like ROC-AUC, not a deployable threshold.

| Forest configuration | Metric | RF | DL | Delta | 95 % interval | Verdict |
|---|---|:---:|:---:|:---:|:---:|---|
| ml_inference RF @ 0.30 (deployed) | sensitivity at equal specificity | 0.837 | 0.863 | +0.026 | [-0.025, +0.070] | no clear difference |
| ml_inference RF @ 0.30 (deployed) | specificity at equal sensitivity | 0.844 | 0.880 | +0.036 | [-0.046, +0.075] | no clear difference |
| ml_inference RF @ val-tuned | sensitivity at equal specificity | 0.777 | 0.788 | +0.011 | [-0.072, +0.060] | no clear difference |
| ml_inference RF @ val-tuned | specificity at equal sensitivity | 0.915 | 0.921 | +0.006 | [-0.036, +0.033] | no clear difference |
