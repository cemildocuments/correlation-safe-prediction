# Selective Prediction under Correlated Failures

Reproducible experiments for moment-constrained selective classification. The method chooses when to issue a prediction when input blocks can fail together. It controls selective error over all joint failure distributions consistent with specified marginal or pairwise probabilities, under a stable complete-data distribution and failures independent of features and labels.

## Quick start

Use Python 3.12 and a virtual environment. The source archives and prediction caches are included, so analysis replay works offline after dependencies are installed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --workers 8
```

The command verifies all six dataset checksums, runs eight mathematical tests, replays 196 predictor configurations and 3,136 policy evaluations, regenerates the exact examples, repeats 1,200 independent synthetic calibration draws, runs 390 sensitivity evaluations, and checks all 199 numerical result files against `expected_results.json`. Each synthetic draw is shared by six policy/order combinations, producing 7,200 output rows. Wall-clock timings are excluded from numerical comparisons. A successful run ends with `Verified all 199 result files` and `Complete protocol finished`.

Replay uses the saved calibration and test prediction arrays and does not fit predictors again. For model training from the original source archives, install the neural dependency and request refitting:

```sh
python -m pip install -r requirements-neural.txt
python reproduce.py --workers 8 --retrain
```

Refitting replaces the prediction caches and result files in that working copy. Preserve an extracted original copy for exact replay comparisons. CPU models use eight independent workers with one numerical-library thread each. Neural fits run serially, with MPS acceleration when available and a CPU fallback. The recorded environment used 18 GB memory. Four-block analyses have 16 masks; the six-block extension has 64 masks. Smaller worker counts reduce concurrent memory use. Neural retraining may differ across accelerator backends or numerical libraries; the supplied caches fix the responses used for exact analysis replay.

The Waveform source archive uses legacy UNIX compression. The loader uses `gzip -dc`, available on common macOS and Linux installations. On Windows, a compatible Linux environment provides the same command.

## Files

| File or directory | Function |
|---|---|
| `reproduce.py` | Runs the full fixed experiment protocol. |
| `core.py` | Simultaneous binomial bounds, equality/interval moment sets, robust selectors, coverage optimization, and exact ratio audits. |
| `datasets.py` | Reads source archives, removes exact duplicate feature-label rows, and creates the declared splits. |
| `run_experiments.py` | Trains predictors or replays cached responses; compares eight selection policies at two error targets. |
| `neural.py` | Fixed 128/64-unit neural classifier, trained for 60 epochs with AdamW. |
| `analytical_examples.py` | Parity counterexample, exact optimal coverage frontiers, and label-dependent failure example. |
| `simulations.py` | Repeated calibration experiments with known population response probabilities. |
| `sensitivity.py` | Calibration-size, failure-rate, and uncertain-moment experiments. |
| `test_core.py` | Eight independent checks, including an exhaustive vertex comparison for the ratio audit. |
| `summarize_results.py` | Writes numerical CSV summaries and all benchmark rows. |
| `validate_results.py` | Checks input hashes and replay agreement, with absolute tolerance 1e-9 and relative tolerance 1e-8. |
| `data/` | Six original UCI dataset archives. |
| `data_manifest.json` | Source URLs, dataset DOIs, licenses, retrieval date, and SHA-256 checksums. |
| `DATA_LICENSES.md` | Dataset attribution and reuse terms. |
| `cache/` | Calibration/test scores, error indicators, and split metadata for 196 trained predictors. |
| `results/` | Per-fit policies, empirical audits, simulations, sensitivity analyses, and aggregate CSV files. |
| `expected_results.json` | Frozen numerical reference outputs, excluding machine-dependent timings. |
| `environment.json` | Recorded software versions and computing configuration. |
| `requirements.txt` | Pinned dependencies for replay and CPU experiments. |
| `requirements-neural.txt` | Adds the neural training dependency. |
| `LICENSE` | MIT terms for the experiment code. |
| `CITATION.cff` | Machine-readable software citation. |

## Experimental protocol

The main study uses Letter Recognition, Pen Digits, Dry Bean, Spambase, Satellite, and Waveform. After deduplication, seeds 0–9 randomly allocate 40% of rows to training, 30% to calibration, and the remainder to test. Standardization is fitted on training rows. Zero-based feature index modulo the number of blocks determines the block assignment. Unavailable standardized features are set to zero, and the observed mask is appended.

Extra Trees uses 200 trees with minimum leaf size 2 and all features considered at each split. Logistic regression uses C=1 and a 600-iteration limit. Four independently masked copies of each training row use an independent block-failure rate of 0.25. Calibration and test responses enumerate every mask. These repeated mask evaluations share the same observations: the calibration sample size is the number of independent complete rows, not rows multiplied by masks.

Sixteen fixed actions consist of accept-all, confidence thresholds 0.30 through 0.95 in steps of 0.05, and reject-all. Targets are 0.10 and 0.20, with calibration failure probability 0.05. The eight policies are pooled, nominal LP, marginal robust, pairwise robust, maskwise, pairwise plug-in, shared marginal, and shared pairwise. A mask-specific mixture of thresholds is evaluated by its expected acceptance probability, so additional simulated policy coins are unnecessary.

Additional fits comprise 18 neural configurations (six datasets, seeds 0–2), 18 six-block tree configurations, and 40 source-partition configurations (Pen Digits and Satellite, two CPU models, ten seeds). The last set deliberately tests a possible change in the complete-data population and is outside the stated guarantee.

Synthetic calibration uses sample sizes 100, 250, 500, 1,000, 2,500, and 5,000, with 200 independent repetitions at each size. For k absent blocks, the high-score probability is 0.8−0.12k, its error probability is 0.01+0.012k+0.25×1{k=4}, and the low-score error probability is 0.20+0.05k. Scores are 0.8 or 0.4. Independent row-level uniform variables induce dependence between mask evaluations. True response probabilities are known, allowing population error to be evaluated directly.

Sensitivity analyses use tree seeds 0–4. Calibration fractions are 0.25, 0.5, and 1.0; failure rates are 0.10, 0.25, and 0.40. Unlabeled telemetry sizes are 200, 1,000, 10,000, and 100,000, using independent failures with probability 0.25. The telemetry and response-bound failure budgets are 0.025 each.

## Interpreting outputs

A mask bit equal to one means the corresponding feature block is absent. Masks use lexicographic binary order. Cache arrays have shape `(independent_rows, masks)`; `cal_conf` and `test_conf` contain maximum class probabilities, while `cal_wrong` and `test_wrong` contain classification error indicators. Metadata records the retained sample count, split sizes, seed, model, grouping, and training time.

`worst_risk_1` and `min_coverage_1` optimize over all mask laws with the declared marginal probabilities. Suffix `_2` adds the declared pairwise probabilities. `cal_adversary_risk_*` evaluates a mask law chosen using calibration responses on held-out test responses. `certified_coverage` is the calibration lower bound under the policy's own optimization set; for the maskwise policy it refers to the full simplex. The plug-in policy's analogous objective is empirical and is not a statistical certificate.

Undefined risk under complete abstention is represented as `NaN` in Python numerical JSON outputs and as an empty field in CSV files. It is counted as abstention, not as a zero-error observation. The strict JSON reference file uses `null` for these entries.

At the 0.10 target, the main marginal audit has 60/120 pooled-rule exceedances and 0/120 marginal-robust exceedances. Mean minimum coverage is 0.61807 for marginal robust versus 0.56263 for maskwise. Under the pairwise set, pairwise robust yields 0.74682 versus 0.70142 for maskwise. Empirical audits are descriptive stress tests, not population upper confidence bounds. The guarantee requires independent calibration rows, stable complete-data responses, and a correctly specified failure-moment set.

## Data and citation

Source data use CC BY 4.0; attribution is in `DATA_LICENSES.md`. Code uses the MIT license. Cite the software with `CITATION.cff` and credit the original datasets when reusing their records.
