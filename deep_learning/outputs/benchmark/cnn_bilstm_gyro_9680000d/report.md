# Benchmark: ml_inference vs deep_learning (CNN-BILSTM +gyro)

- Windows: 9683 test windows over 19 LOSO folds (2689 Fall / 6994 ADL)
- Soft-fall subset: impact windows of fall recordings with peak |a| <= 5.04 g (140 windows)
- Deep model: cnn_bilstm, channels ax, ay, az, acc_mag, tilt_deg, gx, gy, gz, gyr_mag, ensemble size 1, median best epoch 6

| Model | Accuracy | F1 macro | F1 Fall | Sensitivity | Specificity | ROC-AUC | PR-AUC | Soft-fall sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ml_inference RF @ 0.30 (deployed) | 0.842 | 0.816 | 0.747 | 0.837 | 0.844 | 0.906 | 0.854 | 0.914 |
| ml_inference RF @ val-tuned | 0.877 | 0.846 | 0.778 | 0.777 | 0.915 | 0.906 | 0.854 | 0.771 |
| deep_learning CNN-BILSTM +gyro @ val-tuned | 0.880 | 0.851 | 0.784 | 0.783 | 0.918 | 0.928 | 0.867 | 0.757 |

## DL minus RF, subject-level bootstrap (95 % interval)

vs **ml_inference RF @ 0.30 (deployed)**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | +0.038 | [+0.012, +0.062] | DL higher |
| F1 macro | +0.035 | [+0.006, +0.063] | DL higher |
| F1 Fall | +0.039 | [-0.002, +0.080] | no clear difference |
| Sensitivity | -0.055 | [-0.106, -0.012] | RF higher |
| Specificity | +0.074 | [+0.043, +0.101] | DL higher |
| ROC-AUC | +0.022 | [+0.007, +0.039] | DL higher |
| PR-AUC | +0.013 | [-0.014, +0.039] | no clear difference |
| Soft-fall sens. | -0.155 | [-0.320, +0.018] | no clear difference |

vs **ml_inference RF @ val-tuned**

| Metric | Delta | 95 % interval | Verdict |
|---|:---:|:---:|---|
| Accuracy | +0.003 | [-0.017, +0.019] | no clear difference |
| F1 macro | +0.004 | [-0.020, +0.024] | no clear difference |
| F1 Fall | +0.006 | [-0.027, +0.035] | no clear difference |
| Sensitivity | +0.004 | [-0.037, +0.038] | no clear difference |
| Specificity | +0.003 | [-0.020, +0.023] | no clear difference |
| ROC-AUC | +0.022 | [+0.007, +0.039] | DL higher |
| PR-AUC | +0.013 | [-0.014, +0.039] | no clear difference |
| Soft-fall sens. | -0.015 | [-0.153, +0.112] | no clear difference |

## Same operating point (ROC curves compared at the forest's threshold)

The deep model's threshold is read off the pooled test ROC curve to match the forest's specificity (or sensitivity). A curve comparison like ROC-AUC, not a deployable threshold.

| Forest configuration | Metric | RF | DL | Delta | 95 % interval | Verdict |
|---|---|:---:|:---:|:---:|:---:|---|
| ml_inference RF @ 0.30 (deployed) | sensitivity at equal specificity | 0.837 | 0.863 | +0.026 | [-0.010, +0.062] | no clear difference |
| ml_inference RF @ 0.30 (deployed) | specificity at equal sensitivity | 0.844 | 0.871 | +0.027 | [-0.013, +0.068] | no clear difference |
| ml_inference RF @ val-tuned | sensitivity at equal specificity | 0.777 | 0.813 | +0.035 | [-0.014, +0.060] | no clear difference |
| ml_inference RF @ val-tuned | specificity at equal sensitivity | 0.915 | 0.936 | +0.021 | [-0.008, +0.043] | no clear difference |
