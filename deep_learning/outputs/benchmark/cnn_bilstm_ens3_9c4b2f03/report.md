# Benchmark: ml_inference vs deep_learning (CNN-BILSTM x3 seeds)

- Windows: 9683 test windows over 19 LOSO folds (2689 Fall / 6994 ADL)
- Soft-fall subset: impact windows of fall recordings with peak |a| <= 5.04 g (140 windows)
- Deep model: cnn_bilstm, channels ax, ay, az, acc_mag, tilt_deg, ensemble size 3, median best epoch 6

| Model | Accuracy | F1 macro | F1 Fall | Sensitivity | Specificity | ROC-AUC | PR-AUC | Soft-fall sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ml_inference RF @ 0.30 (deployed) | 0.842 | 0.816 | 0.747 | 0.837 | 0.844 | 0.906 | 0.854 | 0.914 |
| ml_inference RF @ val-tuned | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | 0.854 | 0.771 |
| deep_learning CNN-BILSTM x3 seeds @ val-tuned | 0.867 | 0.836 | 0.764 | 0.775 | 0.902 | 0.925 | 0.863 | 0.879 |

## DL minus RF, subject-level bootstrap (95 % interval)

vs **ml_inference RF @ 0.30 (deployed)**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | +0.024 | [-0.004, +0.046] | no clear difference |
| F1 macro | +0.020 | [-0.012, +0.045] | no clear difference |
| F1 Fall | +0.018 | [-0.026, +0.055] | no clear difference |
| Sensitivity | -0.059 | [-0.123, +0.003] | no clear difference |
| Specificity | +0.056 | [+0.021, +0.085] | DL higher |
| ROC-AUC | +0.020 | [-0.007, +0.043] | no clear difference |
| PR-AUC | +0.010 | [-0.031, +0.044] | no clear difference |
| Soft-fall sens. | -0.034 | [-0.097, +0.031] | no clear difference |

vs **ml_inference RF @ val-tuned**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | -0.011 | [-0.036, +0.008] | no clear difference |
| F1 macro | -0.011 | [-0.043, +0.010] | no clear difference |
| F1 Fall | -0.015 | [-0.060, +0.015] | no clear difference |
| Sensitivity | +0.000 | [-0.058, +0.051] | no clear difference |
| Specificity | -0.015 | [-0.045, +0.009] | no clear difference |
| ROC-AUC | +0.020 | [-0.007, +0.043] | no clear difference |
| PR-AUC | +0.010 | [-0.031, +0.044] | no clear difference |
| Soft-fall sens. | +0.106 | [+0.014, +0.197] | DL higher |

## Same operating point (ROC curves compared at the forest's threshold)

The deep model's threshold is read off the pooled test ROC curve to match the forest's specificity (or sensitivity). A curve comparison like ROC-AUC, not a deployable threshold.

| Forest configuration | Metric | RF | DL | Delta | 95 % interval | Verdict |
|---|---|:---:|:---:|:---:|:---:|---|
| ml_inference RF @ 0.30 (deployed) | sensitivity at equal specificity | 0.837 | 0.847 | +0.010 | [-0.055, +0.058] | no clear difference |
| ml_inference RF @ 0.30 (deployed) | specificity at equal sensitivity | 0.844 | 0.858 | +0.014 | [-0.073, +0.064] | no clear difference |
| ml_inference RF @ val-tuned | sensitivity at equal specificity | 0.777 | 0.784 | +0.006 | [-0.064, +0.046] | no clear difference |
| ml_inference RF @ val-tuned | specificity at equal sensitivity | 0.915 | 0.917 | +0.002 | [-0.063, +0.039] | no clear difference |
