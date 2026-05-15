# Extending GUARD: Detecting Security Smells Across IaC Languages

Bachelor's thesis project — University of Twente, 2026.

GUARD is a machine learning-based tool for detecting security smells in Ansible Infrastructure as Code files. This project extends GUARD to support **Terraform** and **Puppet**.

## Project Structure

```
ThesisCode/
├── src/
│   ├── guard.py                    # Main detection pipeline (dispatcher + shared detection logic)
│   ├── parsers/
│   │   ├── __init__.py             # ParsedLine dataclass + get_parser()
│   │   ├── ansible_parser.py       # YAML-based Ansible parser
│   │   ├── puppet_parser.py        # Puppet .pp parser (=> and $var = assignments)
│   │   └── terraform_parser.py     # Terraform HCL parser (= assignments, multi-line lists)
│   ├── contextualization/
│   │   ├── puppet_context.py       # Puppet $variable resolution
│   │   └── terraform_context.py    # Terraform var/local resolution
│   ├── contexualizationPhase.py    # Original Ansible {{ variable }} resolution
│   ├── vectorizer.py               # Word2Vec vectorization for key names
│   ├── models_trainer.py           # ML model training (Label Propagation + SetFit)
│   └── pretrainedModels/           # Trained models + Google News Word2Vec embeddings
├── datasets/
│   ├── puppet/                     # Puppet oracle dataset (GLITCH ASE 2022)
│   ├── terraform/                  # Terraform oracle dataset (GLITCH replication package)
│   ├── trainingSetComments.csv     # SetFit training data (suspicious comments)
│   └── testSetComments.csv         # SetFit test data
├── evaluation/
│   ├── evaluate_puppet.py          # Puppet: compare GUARD output against oracle
│   └── evaluate_terraform.py       # Terraform: compare GUARD output against oracle
├── results/
│   ├── baseline_puppet.csv         # RQ1: unmodified GUARD on Puppet
│   ├── baseline_terraform.csv      # RQ1: unmodified GUARD on Terraform
│   ├── extended_puppet.csv         # RQ2: extended GUARD on Puppet
│   ├── extended_terraform.csv      # RQ2: extended GUARD on Terraform
│   └── figures/
├── retrain_setfit.py               # SetFit retraining script
├── hardcoded_secrets_dataset.csv   # Label Propagation training data
├── susp_comments_dataset.csv       # Suspicious comments dataset (~15k lines)
├── requirements.txt
└── README.md
```

## Architecture

The original GUARD processes only Ansible (YAML) files through a monolithic pipeline. This project refactors the architecture into a dispatcher pattern:

1. **Dispatcher** (`guard.py`) — routes files to the correct parser based on extension (`.yml`/`.yaml` → Ansible, `.pp` → Puppet, `.tf` → Terraform)
2. **Language-specific parsers** — convert raw file lines into standardised `ParsedLine` objects with key, value, comment, and parameterization metadata
3. **Shared detection logic** — operates on `ParsedLine` objects regardless of source language, applying both RegEx pattern matching and ML-based semantic analysis

This design means the detection logic (RegEx + ML) is written once and works across all three languages. Only the parsing and variable resolution layers differ per language.

### Key parser adaptations

- **Puppet**: handles `=>` (resource attributes) and `$variable = value` (parameter defaults); detects `$var`, `${var}`, and `$module::var` as parameterized values
- **Terraform**: handles `=` assignments with comparison operator exclusion (`==`, `!=`, `>=`, `<=`); joins multi-line lists (`cidr_blocks = [\n"0.0.0.0/0"]`) into single values; detects `var.`, `local.`, `${...}` interpolation, and function calls as parameterized
- **Detection extensions**: `protocol = "HTTP"` rule for Terraform load balancer configurations; empty password detection for keys with empty or `undef` values

## Setup

1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Unzip the pretrained Word2Vec model:
   ```bash
   cd src/pretrainedModels
   tar -xzf googleNews-vectors-negative300.bin.tar.gz
   ```

3. Download the external datasets (see below).

## Usage

Run GUARD on a folder of IaC files:
```bash
cd src
python guard.py -p <path-to-folder> -f -o output.csv
```

Run with contextualization (Ansible projects only):
```bash
python guard.py -p <path-to-folder> -f -c -o output.csv
```

Evaluate against oracle datasets:
```bash
python evaluation/evaluate_puppet.py -g results/extended_puppet.csv -o datasets/puppet/oracle-dataset-analysis/oracle-dataset.csv
python evaluation/evaluate_terraform.py -g results/extended_terraform.csv ...
```

## Model Reproduction

The retrained SetFit suspicious comments model is not included in the repository due to size constraints.

To reproduce the model:
```bash
python retrain_setfit.py
```

## Results

### Puppet

| Metric    | Baseline (RQ1) | Extended (RQ2) |
|-----------|----------------|----------------|
| TP        | 5              | 33             |
| FP        | 138            | 169            |
| FN        | 35             | 7              |
| Precision | 0.035          | 0.163          |
| Recall    | 0.125          | 0.825          |
| F1        | 0.055          | 0.273          |

Oracle: 65 total smells, 40 GUARD-compatible, 25 out-of-scope.

### Terraform

| Metric    | Baseline (RQ1) | Extended (RQ2) |
|-----------|----------------|----------------|
| TP        | 0              | 52             |
| FP        | 38             | 63             |
| FN        | 66             | 9              |
| Precision | 0.000          | 0.452          |
| Recall    | 0.000          | 0.852          |
| F1        | 0.000          | 0.591          |

Oracle: 376 total smells, only 66 GUARD-compatible (~17.5%), 310 out-of-scope.

### Key findings

1. **Parser extension is the primary driver of improvement.** Both languages went from near-zero detection to >0.8 recall through parsing alone — the underlying detection logic (RegEx + ML) transfers well once preprocessing is correct.
2. **Taxonomy mismatch is fundamental for Terraform.** Only 17.5% of Terraform oracle smells overlap with GUARD's 7-smell taxonomy. The majority are infrastructure-level concerns (logging, IAM, encryption, firewall) absent from configuration management tools.
3. **ML models require language-specific retraining.** The Ansible-trained SetFit and Label Propagation models produce significant false positives on Puppet and Terraform due to differing comment styles and key naming conventions. Puppet is particularly affected (~117 suspicious comment FPs).

## External Datasets

The evaluation datasets are too large to commit. Download them from the following sources and place them in the `datasets/` directory:

| Dataset | Source | Size | Link |
|---------|--------|------|------|
| Terraform (GLITCH) | Figshare | ~94 MB | [terraform-GLITCH-replication-package.zip](https://figshare.com/articles/thesis/terraform-GLITCH-replication-package_zip/24574234) |
| Puppet (GLITCH ASE 2022) | Figshare | ~1.7 GB | [GLITCH Replication Package](https://doi.org/10.6084/m9.figshare.19726603.v2) |

These are the same datasets used in the GLITCH papers, enabling direct comparison of detection performance (RQ3).

## References

- **GUARD**: Grella et al., *Guard: Robust Detection of Ansible Security Smells*, 2024 (unpublished).
- **GLITCH**: Saavedra & Ferreira, *GLITCH: Automated Polyglot Security Smell Detection in Infrastructure as Code*, ASE 2022.
- **GLITCH extended**: Saavedra et al., *Polyglot Code Smell Detection for Infrastructure as Code with GLITCH*, ASE 2023.
- **Seven Sins**: Rahman, Parnin & Williams, *The Seven Sins: Security Smells in Infrastructure as Code Scripts*, ICSE 2019.