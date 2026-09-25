# LOSO-CV fall detection benchmark

- Windows: 9683 (2689 Fall / 6994 ADL)
- Features: 53
- Subjects (folds): 19 -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## Pooled performance (all folds combined)

| model               | sensitivity | specificity | f1     | accuracy | precision | bal_acc | missed_falls | false_alarms |
|---------------------|-------------|-------------|--------|----------|-----------|---------|--------------|--------------|
| random_forest       | 0.7196      | 0.9477      | 0.7756 | 0.8843   | 0.8409    | 0.8336  | 754          | 366          |
| svm_rbf             | 0.7605      | 0.9111      | 0.7636 | 0.8693   | 0.7668    | 0.8358  | 644          | 622          |
| logistic_regression | 0.7646      | 0.7322      | 0.6213 | 0.7412   | 0.5233    | 0.7484  | 633          | 1873         |
| svm_linear          | 0.7917      | 0.7070      | 0.6201 | 0.7306   | 0.5096    | 0.7494  | 560          | 2049         |

## Fold-averaged performance (mean +/- std over subjects, NaN-aware)

| model               | accuracy        | sensitivity     | specificity     | precision       | f1              | balanced_accuracy |
|---------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------|
| random_forest       | 0.882 +/- 0.078 | 0.701 +/- 0.135 | 0.946 +/- 0.035 | 0.868 +/- 0.100 | 0.765 +/- 0.103 | 0.820 +/- 0.062   |
| svm_rbf             | 0.861 +/- 0.073 | 0.743 +/- 0.151 | 0.894 +/- 0.056 | 0.799 +/- 0.098 | 0.765 +/- 0.118 | 0.816 +/- 0.080   |
| logistic_regression | 0.721 +/- 0.089 | 0.765 +/- 0.118 | 0.712 +/- 0.087 | 0.632 +/- 0.138 | 0.684 +/- 0.121 | 0.746 +/- 0.088   |
| svm_linear          | 0.710 +/- 0.089 | 0.801 +/- 0.123 | 0.683 +/- 0.094 | 0.620 +/- 0.131 | 0.689 +/- 0.115 | 0.750 +/- 0.076   |

## Aggregated confusion matrices

```
random_forest (aggregated over all LOSO folds)
                 pred ADL   pred Fall
  true ADL           6628         366
  true Fall           754        1935
  missed falls (FN): 754    false alarms (FP): 366
```

```
svm_rbf (aggregated over all LOSO folds)
                 pred ADL   pred Fall
  true ADL           6372         622
  true Fall           644        2045
  missed falls (FN): 644    false alarms (FP): 622
```

```
logistic_regression (aggregated over all LOSO folds)
                 pred ADL   pred Fall
  true ADL           5121        1873
  true Fall           633        2056
  missed falls (FN): 633    false alarms (FP): 1873
```

```
svm_linear (aggregated over all LOSO folds)
                 pred ADL   pred Fall
  true ADL           4945        2049
  true Fall           560        2129
  missed falls (FN): 560    false alarms (FP): 2049
```

## Per-subject folds for the best model (random_forest)

| test_subject | n_test | n_test_fall | sensitivity | specificity | f1     | accuracy | fn  | fp |
|--------------|--------|-------------|-------------|-------------|--------|----------|-----|----|
| 1            | 247    | 0           | n/a         | 0.9514      | n/a    | 0.9514   | 0   | 12 |
| 2            | 637    | 247         | 0.8583      | 0.9231      | 0.8671 | 0.8980   | 35  | 30 |
| 3            | 234    | 0           | n/a         | 0.9060      | n/a    | 0.9060   | 0   | 22 |
| 4            | 542    | 152         | 0.7368      | 0.9795      | 0.8235 | 0.9114   | 40  | 8  |
| 5            | 637    | 234         | 0.8077      | 0.9479      | 0.8514 | 0.8964   | 45  | 21 |
| 6            | 650    | 221         | 0.7873      | 0.9044      | 0.7982 | 0.8646   | 47  | 41 |
| 7            | 429    | 78          | 0.5256      | 0.9316      | 0.5734 | 0.8578   | 37  | 24 |
| 8            | 130    | 78          | 0.6667      | 0.9231      | 0.7761 | 0.7692   | 26  | 4  |
| 9            | 442    | 0           | n/a         | 0.9615      | n/a    | 0.9615   | 0   | 17 |
| 10           | 247    | 0           | n/a         | 1.0000      | n/a    | 1.0000   | 0   | 0  |
| 11           | 465    | 231         | 0.6623      | 0.9658      | 0.7806 | 0.8151   | 78  | 8  |
| 12           | 429    | 0           | n/a         | 0.9510      | n/a    | 0.9510   | 0   | 21 |
| 13           | 260    | 0           | n/a         | 0.9962      | n/a    | 0.9962   | 0   | 1  |
| 14           | 416    | 117         | 0.7521      | 0.8562      | 0.7097 | 0.8269   | 29  | 43 |
| 15           | 429    | 156         | 0.8397      | 0.9524      | 0.8733 | 0.9114   | 25  | 13 |
| 16           | 135    | 70          | 0.3714      | 0.9846      | 0.5361 | 0.6667   | 44  | 1  |
| 17           | 273    | 143         | 0.8252      | 0.9154      | 0.8676 | 0.8681   | 25  | 11 |
| 18           | 2457   | 728         | 0.6882      | 0.9549      | 0.7666 | 0.8759   | 227 | 78 |
| 19           | 624    | 234         | 0.5897      | 0.9718      | 0.7206 | 0.8285   | 96  | 11 |

## Note on undefined folds

Subjects with no Fall recordings produce folds without positive samples, so sensitivity/precision/F1 are undefined (NaN) there and are excluded from the fold averages: [1, 3, 9, 10, 12, 13].
