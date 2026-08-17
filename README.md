# GUARD — Robust Detection of Ansible Security Smells

GUARD is a static analyzer for Ansible playbooks that detects four security
smells. Three of them (improper IP address binding, use of HTTP, weak
cryptography) are matched with regular expressions. The fourth —
**hardcoded secrets** — is the contribution of this work: instead of matching
key names against a fixed keyword list, GUARD embeds the configuration key
name with a sentence encoder and classifies it with a logistic regression.

The motivation is robustness. Keyword-based detectors such as GLITCH collapse
when the vocabulary shifts: removing the 38 keywords that actually fire in
GLITCH's rule set drops its F1 from 0.66 to 0.00, while GUARD — trained on the
same data with those keys removed from the training set — still reaches 0.29
(precision 0.86). Under the standard evaluation GUARD scores F1 0.76 against
GLITCH's 0.67, and comes within about three points of zero-shot frontier LLMs
at a fraction of the inference cost.

---

## Repository layout

```
src/                              the tool itself
  guard.py                        CLI entry point (regex rules + ML classifier)
  model.py                        GUARD class, embedder, dataset loader, frozen constants
  trainer.py                      trains and pickles the hardcoded-secret model
  contexualizationPhase.py        resolves {{ jinja variables }} before analysis

resources/
  in/ground_truth.csv             oracle: Ansible module params with a `secret` flag
  in/GLITCH_results.csv           GLITCH output (baseline)
  in/intellisa_results.csv        IntellSia output (baseline)
  datasets/dataset_t3.csv         canonical dataset, one row per unique leaf
  pretrainedModels/               guard_model.pkl + dataset_embeddings.npy
  out/                            metrics CSVs and figures produced by the study

tests/                            the empirical study, in four phases
  Phase_0_Dataset/                dataset construction, stats, negative validation
  Phase_1_LogReg/                 embedding/hyperparameter ablation, threshold, held-out eval
  Phase_2_Comparison/             GUARD vs LLMs, classical ML, GLITCH, IntellSia
  Phase_3_Robustness/             keyword-removal degradation study + plots
```

## Requirements

Python 3.12+ and:

```
sentence-transformers  scikit-learn  pandas  numpy  scipy
pyyaml  matplotlib  seaborn  requests  python-dotenv  anthropic
```

The encoder (`sentence-transformers/all-mpnet-base-v2`, ~420 MB) is downloaded
on first use. Phase 2 additionally needs an `ANTHROPIC_API_KEY` in a `.env` at
the repo root, and a local [Ollama](https://ollama.com) serving
`mistral:7b-instruct-v0.3-q4_K_M` on port 11434.

---

## Running the analyzer

All scripts use paths relative to their own directory, so run them from where
they live — `guard.py` from `src/`:

```bash
cd src

# single file
python guard.py -p ../path/to/playbook.yml -o output.csv

# whole project
python guard.py -p ../path/to/repo -f -o output.csv

# whole project, resolving Jinja variables first (folders only)
python guard.py -p ../path/to/repo -f -c -o output.csv
```

| flag | meaning |
| --- | --- |
| `-p, --path` | file or folder to analyze (required) |
| `-f, --folder` | treat `--path` as a folder |
| `-c, --context` | run the contextualization phase (requires `-f`) |
| `-o, --output` | output CSV (default `output.csv`) |
| `--leaves` | extra CSV with one row per analyzed key, including the pipeline stage that decided it |

The output CSV has one row per detected smell: `file, smell, line, content`.
Files with no findings get a single `none` row. `--leaves` is the debugging
view: it records every `key: value` pair the parser saw and why it was skipped,
matched by regex, or sent to the classifier.

The tool copies the target into `./temp`, analyzes it there, and deletes it
afterwards — so contextualization never rewrites your sources.

### Training

`resources/pretrainedModels/guard_model.pkl` ships with the repo. To rebuild it:

```bash
cd src
python trainer.py \
  -d ../resources/datasets/dataset_t3.csv \
  -e ../resources/pretrainedModels/dataset_embeddings.npy \
  -o ../resources/pretrainedModels
```

Drop `-e` to recompute embeddings from scratch (add `--save-embeddings` to cache
them). The trainer prints an in-sample sanity check and warns if recall or max
probability fall well below the reference (~0.79 / ~0.99), which usually means
an encoder mismatch or missing standardization.

---

## Method

The constants in `src/model.py` define the method under evaluation and are not
tunable — changing them invalidates the reported numbers.

| constant | value |
| --- | --- |
| `MODEL_NAME` | `sentence-transformers/all-mpnet-base-v2` |
| `FIEX_C` | `0.01` (L2 logistic regression, lbfgs, no class weighting) |
| `DECISION_THRESHOLD` | `0.40` |

Only the **leaf** of the key is used — `mysql.root.password` becomes `password`.
The classifier sees the key name and nothing else: no value, no surrounding
task, no file context. The scaler is fitted on the full corpus (labeled and
unlabeled), the classifier only on labeled rows.

### Dataset

`Phase_0/0_generate_datasets.py` collapses the oracle into one row per unique
leaf and assigns a label by majority vote over its occurrences (`--theta 0.5`;
`--theta-mode any_positive` is available as a sensitivity check). Leaves that
never carry an annotation but occur at least `--min-occ` times are promoted to
**heuristic negatives** — the assumption being that a key name appearing
repeatedly without ever being flagged is not a secret.

`dataset_t3.csv` is the `--min-occ 3` variant:

| | count |
| --- | --- |
| unique leaves | 31,486 |
| positive | 658 |
| negative | 4,985 (201 annotated + 4,784 heuristic) |
| unlabeled | 25,843 |

`2_negative_validation.py` audits that heuristic: sample 200 heuristic negatives,
label them by hand in `resources/out/negatives_review.csv`, then rerun with
`MODE="report"` for the error rate and a Jeffreys 95% CI. All metrics are also
reported on annotated-only negatives (`f1_annotated_only`) so the heuristic
never carries the headline result alone.

---

## Reproducing the study

```bash
# Phase 0 — build the dataset and inspect it
cd tests/Phase_0_Dataset
python 0_generate_datasets.py --min-occ 3 --out ../../resources/datasets/dataset_t3.csv
python 1_dataset_stats.py
python 2_negative_validation.py

# Phase 1 — model selection on SELECT, confirmation on HOLDOUT
cd ../Phase_1_LogReg
python 0_embedding_ablation.py      # 3 encoders x 7 C values x 2 class weights, 30 sub-splits
python 1_threshold_analysis.py      # threshold sweep 0.05-0.95
python 2_evaluation_chosen_model.py # single pass on the held-out split

# Phase 2 — comparison against every baseline
cd ../Phase_2_Comparison
python 0_LLMs_classification.py                      # caches LLM predictions (resumable)
python 1_normalizzation_Glitch_and_intellisa.py      # maps GLITCH/IntellSia output onto leaves
python 2_method_comparison.py                        # 30 group-disjoint runs, all methods

# Phase 3 — keyword-removal robustness
cd ../Phase_3_Robustness
python 0_robustness_test.py \
  -d ../../resources/datasets/dataset_t3.csv \
  -e ../../resources/pretrainedModels/dataset_embeddings.npy \
  --glitch-file ../../resources/in/GLITCH_results.csv \
  -o ../../resources/out/ablation_regex.csv
python 1_plotter.py
```

`splits.py` holds the partition protocol: a 30% held-out split (seed 7), never
touched during model selection, and 30 inner sub-splits for stability. Rows in
the held-out set stay in the corpus but have their labels masked to `-1`, so
they contribute to the scaler and to the unlabeled pool without leaking
supervision.

### Results

Phase 2, mean over 30 runs, positive-class F1 (`resources/out/comparison_summary.csv`):

| method | F1 | precision | recall |
| --- | --- | --- | --- |
| LLM claude-fable-5 (zero-shot) | 0.789 ± 0.022 | 0.824 | 0.758 |
| LLM claude-sonnet-4-6 (zero-shot) | 0.789 ± 0.020 | 0.821 | 0.760 |
| **GUARD (ours)** | **0.762 ± 0.019** | 0.763 | 0.763 |
| kNN (k=21, distance) | 0.742 ± 0.024 | 0.747 | 0.738 |
| Random Forest (d=20, n=300) | 0.735 ± 0.027 | 0.747 | 0.724 |
| LabelPropagation (labeled-only) | 0.733 ± 0.023 | 0.727 | 0.739 |
| LabelSpreading (k=13, α=0.5) | 0.695 ± 0.018 | 0.684 | 0.707 |
| LabelPropagation (+unlabeled) | 0.683 ± 0.026 | 0.743 | 0.633 |
| GLITCH RegEx | 0.675 ± 0.026 | 0.660 | 0.691 |
| LLM mistral 7b-instruct q4 (local) | 0.560 ± 0.020 | 0.456 | 0.729 |
| IntellSia | 0.000 | — | — |

Phase 3, keyword-removal curve (`resources/out/ablation_regex.csv`, plotted in
`fig_ablation_*.pdf`): with all 38 keywords available GLITCH reaches F1 0.662
against GUARD's 0.807; at full removal GLITCH goes to 0.000 while GUARD holds
0.287 at precision 0.857. The degradation is graceful rather than catastrophic
because the embedding generalizes across morphological variants the keyword
list never enumerates.

---

## Citation

The hardcoded-secret definition used in the LLM prompts follows Rahman et al.,
*The Seven Sins: Security Smells in Infrastructure as Code Scripts*, ICSE 2019.
