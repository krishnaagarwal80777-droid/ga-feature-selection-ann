# GA-based feature selection for an ANN classifier

This project asks whether a genetic algorithm (GA), written from scratch in NumPy, can pick a smaller set of input attributes for a neural-network classifier without losing accuracy. It also compares the GA with standard feature-selection methods, measures how well it rejects pure-noise attributes, and checks how sensitive it is to its own hyperparameters.

## Datasets

- **Breast cancer (Wisconsin diagnostic)**: 569 samples, 30 continuous attributes, binary target (malignant / benign). Loaded with `sklearn.datasets.load_breast_cancer`.
- **German credit (OpenML id 31)**: 1000 samples, 20 attributes (7 numeric, 13 categorical), binary target (bad / good). Each categorical attribute is one-hot encoded, giving 61 columns. The GA and the baselines select whole attributes, so all dummy columns of an attribute go in or out together.

Each seed uses a stratified 80/20 train/test split. A `StandardScaler` is fitted on the training split only.

For the noise experiment, each dataset gets as many extra columns of standard-normal noise as it has real attributes (30 for breast cancer, 20 for German credit), generated with a fixed seed.

## Method

### Genetic algorithm (`ga_fs/ga.py`)

| Component | Setting |
|---|---|
| Chromosome | one bit per attribute (group-level, not per one-hot column) |
| Population | 30 |
| Selection | tournament of size 3 (drawn with replacement) |
| Crossover | uniform, probability 0.8 |
| Mutation | bit flip, per-bit probability 1/n (n = number of attributes) |
| Elitism | best 2 copied unchanged |
| Stopping | 40 generations, or 10 generations without improvement in the best fitness |
| Empty chromosome | if all bits are 0, one random bit is set to 1 |
| Cache | fitness of each chromosome is stored and reused |

**Fitness** is the mean balanced accuracy from 3-fold cross-validation repeated twice on the training split. A size penalty is then subtracted:

```
fitness = CV balanced accuracy - alpha * (attributes selected / total attributes),   alpha = 0.01
```

**Surrogate network.** Training the final network for every candidate would be too slow, so fitness uses a small, fast MLP: one hidden layer of 16 units, 300 iterations. The selected attributes are then evaluated with the final network. The CV folds and the network's random seed are the same for every candidate within a run.

**Final classifier** (`ga_fs/ann.py`): scikit-learn `MLPClassifier` with hidden layers (64, 32), ReLU, Adam, L2 alpha 1e-4, 1000 iterations, no early stopping.

### Baselines (`ga_fs/baselines.py`)

Every baseline is given the same number of attributes k that the GA chose for that seed.

- **All features**: no selection.
- **MI**: mutual information between each column and the target. One-hot columns are treated as discrete. An attribute's score is the maximum over its columns, and the top k attributes are kept.
- **RFE**: recursive feature elimination with logistic regression, run down to a full ranking. An attribute's rank is the best rank among its columns, and the top k are kept.
- **PCA**: k principal components of all columns, standardised before the network. PCA reduces dimension but still uses every attribute.
- **Random**: k attributes drawn at random. Metrics are averaged over 10 draws per seed.

### Evaluation

The metrics are balanced accuracy (primary), macro F1, recall on the minority class (malignant / bad), ROC AUC from predicted probabilities, and accuracy.

The main experiment uses 10 seeds; the noise and sensitivity experiments use 5. GA-vs-baseline comparisons pair results by seed. Each comparison reports:
- a paired t-test with its 95% CI;
- the Nadeau-Bengio corrected resampled t-test (n_test/n_train = 0.25) with its CI;
- a Wilcoxon signed-rank test, together with the smallest p-value the Wilcoxon could reach for that sample size;
- Holm correction across the comparisons within each dataset.

## Installation

Python 3.10 or newer (developed on 3.13):

```
pip install -r requirements.txt
```

## Running

All commands run from the repository root. Runtimes were measured on a 12-thread desktop CPU, and cross-validation runs in parallel on all cores.

| Command | What it does | Approx. runtime |
|---|---|---|
| `python main.py` | Main experiment: GA plus all baselines, 2 datasets x 10 seeds. Writes `results/all_runs.csv`, `summary.csv`, `stats.csv`, the feature-frequency files and the GA history files | ~1 h |
| `python main.py --noise` | Noise experiment: 2 datasets x 5 seeds with added noise attributes. Writes `noise_all_runs.csv`, `noise_summary.csv`, `noise_stats.csv` | ~40 min |
| `python main.py --sensitivity` | Sweeps alpha, mutation rate and population size (plus alpha on the noisy data), 5 seeds each. Writes `sensitivity_runs.csv` and `sensitivity_summary.csv` | ~8-9 h |
| `python main.py --rerun-baselines` | Recomputes only the MI and PCA rows from the saved GA selections, then rebuilds the summaries and stats. Backs up the original CSVs first | ~2 min |
| `python main.py --plots-only` | Builds every figure in `figures/` and the report tables in `results/tables/` from the saved CSVs | ~15 s |

Every experiment can be resumed: completed dataset/seed combinations (or sensitivity settings) are skipped on a rerun. All seeds are fixed, so a rerun reproduces the saved results exactly.

## Main results

Mean ± standard deviation over 10 seeds on the held-out test sets (from `results/tables/main_comparison.csv`). For PCA the attribute count is the number of components; it still uses every attribute.

| Dataset | Method | Attributes | Balanced acc. | Macro F1 | Minority recall | ROC AUC |
|---|---|---|---|---|---|---|
| Breast cancer | All features | 30.0 ± 0.0 | 0.9815 ± 0.0149 | 0.9829 ± 0.0133 | 0.9714 ± 0.0270 | 0.9972 ± 0.0034 |
| | GA | 14.2 ± 2.2 | 0.9696 ± 0.0105 | 0.9689 ± 0.0099 | 0.9643 ± 0.0257 | 0.9956 ± 0.0043 |
| | MI | 14.2 ± 2.2 | 0.9623 ± 0.0184 | 0.9623 ± 0.0193 | 0.9524 ± 0.0317 | 0.9921 ± 0.0077 |
| | RFE | 14.2 ± 2.2 | 0.9760 ± 0.0121 | 0.9755 ± 0.0102 | 0.9714 ± 0.0270 | 0.9965 ± 0.0035 |
| | PCA (k components) | 14.2 ± 2.2 | 0.9670 ± 0.0219 | 0.9679 ± 0.0215 | 0.9548 ± 0.0326 | 0.9959 ± 0.0029 |
| | Random | 14.2 ± 2.2 | 0.9662 ± 0.0133 | 0.9658 ± 0.0116 | 0.9586 ± 0.0279 | 0.9945 ± 0.0043 |
| German credit | All features | 20.0 ± 0.0 | 0.6625 ± 0.0490 | 0.6662 ± 0.0507 | 0.5000 ± 0.0778 | 0.7286 ± 0.0486 |
| | GA | 11.4 ± 1.8 | 0.6380 ± 0.0346 | 0.6389 ± 0.0340 | 0.4767 ± 0.0771 | 0.6967 ± 0.0545 |
| | MI | 11.4 ± 1.8 | 0.6438 ± 0.0386 | 0.6467 ± 0.0395 | 0.4783 ± 0.0572 | 0.6999 ± 0.0557 |
| | RFE | 11.4 ± 1.8 | 0.6424 ± 0.0362 | 0.6442 ± 0.0367 | 0.4833 ± 0.0598 | 0.7129 ± 0.0461 |
| | PCA (k components) | 11.4 ± 1.8 | 0.6290 ± 0.0370 | 0.6295 ± 0.0374 | 0.4767 ± 0.0540 | 0.7013 ± 0.0326 |
| | Random | 11.4 ± 1.8 | 0.6127 ± 0.0249 | 0.6142 ± 0.0248 | 0.4330 ± 0.0465 | 0.6698 ± 0.0326 |

## Key findings

- **Fewer attributes.** The GA cut the attribute count by 53% on breast cancer (30 to 14.2) and 43% on German credit (20 to 11.4), at a balanced-accuracy cost of 1.2 and 2.5 points against using all features.
- **Small accuracy cost, not significant once corrected.** On breast cancer the GA is below all features under the plain paired t-test (p = 0.010). That difference is not significant after Holm correction (p = 0.052) or under the Nadeau-Bengio test (p = 0.12).
- **Indistinguishable from the standard selectors.** The GA's differences from MI, RFE and PCA are all within ±0.01 balanced accuracy, and none is significant under any test. With the same k, RFE is numerically ahead on breast cancer, and MI and RFE are slightly ahead on German credit.
- **Better than random only on German credit.** There the GA beats random subsets of the same size by 2.5 points (Holm-adjusted t-test p = 0.004, Wilcoxon p = 0.010). Under the more conservative corrected test this is not significant after Holm (p = 0.13).
- **Partial noise rejection.** At alpha 0.01 the GA kept 8.2 of 30 noise attributes on breast cancer and 7.8 of 20 on German credit. That is fewer than random selection, but more than MI on breast cancer (2.0) and RFE on German credit (3.8). Raising alpha to 0.1-0.2 brings the GA down to 0.6-3.8 noise attributes, but on clean breast cancer data the independent train CV already falls once alpha reaches 0.05. Mutation rate (0.5/n to 2/n) and population size (10 to 50) had no clear effect on independent train CV, while runtime grows roughly linearly with population size.

## Limitations

- **Surrogate network.** GA fitness uses a 16-unit network trained for 300 iterations, not the final (64, 32) network. The selected subsets are therefore optimised for the surrogate and may not be the best subsets for the final model. The GA's own CV score is also optimistic, because the same folds score every candidate: in the sensitivity runs it is 1-5 points above an independent 5x3 CV of the final network on the same training data.
- **Overlapping resampled test sets.** The seeds are repeated 80/20 splits of the same data, so the test sets overlap and the seeds are not independent. A plain paired t-test underestimates the variance. The Nadeau-Bengio corrected test is reported for this reason and should be treated as the primary result. The noise experiment has only 5 seeds, so a Wilcoxon test there cannot reach p < 0.0625.
- **Scaler fitted before the inner CV.** The `StandardScaler` is fitted on the whole training split before the GA's inner cross-validation runs, so the validation folds inside the GA have contributed to the scaling statistics. This affects the GA's fitness estimates only, not the test-set results, and uses no labels. The independent train CV in the sensitivity sweep fits its scaler within each fold and does not have this issue.
- **ANN configuration fixed before feature selection, using early test results.** The final network's configuration (early stopping off, 1000 iterations) was set once, before any feature selection, to fix convergence problems in the all-features model. That change was made after looking at all-features results on seeds 0-4, which are part of the final evaluation. It was a single convergence fix rather than a tuning search, but the network was not chosen blind to the test data.
