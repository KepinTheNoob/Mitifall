# Benchmark: ml_inference vs deep_learning (CNN-BILSTM)

- Windows: 9683 test windows over 19 LOSO folds (2689 Fall / 6994 ADL)
- Soft-fall subset: impact windows of fall recordings with peak |a| <= 5.04 g (140 windows)
- Deep model: cnn_bilstm, channels ax, ay, az, acc_mag, tilt_deg, ensemble size 1, median best epoch 6

| Model | Accuracy | F1 macro | F1 Fall | Sensitivity | Specificity | ROC-AUC | PR-AUC | Soft-fall sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ml_inference RF @ 0.30 (deployed) | 0.842 | 0.816 | 0.747 | 0.837 | 0.844 | 0.906 | 0.854 | 0.914 |
| ml_inference RF @ val-tuned | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | 0.854 | 0.771 |
| deep_learning CNN-BILSTM @ val-tuned | 0.870 | 0.840 | 0.770 | 0.784 | 0.903 | 0.927 | 0.861 | 0.850 |

## DL minus RF, subject-level bootstrap (95 % interval)

vs **ml_inference RF @ 0.30 (deployed)**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | +0.028 | [-0.000, +0.052] | no clear difference |
| F1 macro | +0.024 | [-0.007, +0.050] | no clear difference |
| F1 Fall | +0.024 | [-0.018, +0.061] | no clear difference |
| Sensitivity | -0.052 | [-0.106, -0.001] | RF higher |
| Specificity | +0.057 | [+0.017, +0.088] | DL higher |
| ROC-AUC | +0.021 | [+0.004, +0.041] | DL higher |
| PR-AUC | +0.007 | [-0.021, +0.036] | no clear difference |
| Soft-fall sens. | -0.066 | [-0.129, +0.000] | no clear difference |

vs **ml_inference RF @ val-tuned**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | -0.008 | [-0.032, +0.011] | no clear difference |
| F1 macro | -0.007 | [-0.035, +0.015] | no clear difference |
| F1 Fall | -0.009 | [-0.048, +0.022] | no clear difference |
| Sensitivity | +0.007 | [-0.043, +0.048] | no clear difference |
| Specificity | -0.013 | [-0.048, +0.011] | no clear difference |
| ROC-AUC | +0.021 | [+0.004, +0.041] | DL higher |
| PR-AUC | +0.007 | [-0.021, +0.036] | no clear difference |
| Soft-fall sens. | +0.074 | [-0.026, +0.162] | no clear difference |

## Same operating point (ROC curves compared at the forest's threshold)

The deep model's threshold is read off the pooled test ROC curve to match the forest's specificity (or sensitivity). A curve comparison like ROC-AUC, not a deployable threshold.

| Forest configuration | Metric | RF | DL | Delta | 95 % interval | Verdict |
|---|---|:---:|:---:|:---:|:---:|---|
| ml_inference RF @ 0.30 (deployed) | sensitivity at equal specificity | 0.837 | 0.859 | +0.022 | [-0.033, +0.060] | no clear difference |
| ml_inference RF @ 0.30 (deployed) | specificity at equal sensitivity | 0.844 | 0.873 | +0.029 | [-0.031, +0.061] | no clear difference |
| ml_inference RF @ val-tuned | sensitivity at equal specificity | 0.777 | 0.771 | -0.006 | [-0.078, +0.044] | no clear difference |
| ml_inference RF @ val-tuned | specificity at equal sensitivity | 0.915 | 0.909 | -0.006 | [-0.049, +0.033] | no clear difference |
