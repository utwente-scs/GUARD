# GUARD — Multi-Language IaC Security Smell Detection

GUARD is a machine learning-based tool for detecting security smells in Infrastructure as Code files. Originally built for Ansible, this extended version also supports **Puppet**, **Chef**, and **Terraform**.

## Supported Languages

| Language  | Extensions     | Parser | ML Models | Contextualization |
|-----------|---------------|--------|-----------|-------------------|
| Ansible   | `.yml`, `.yaml` | YAML   | Base      | Jinja2 `{{ }}`    |
| Puppet    | `.pp`          | `=>` / `$var =` | Per-language | `$var`, `${var}`, `$module::var` |
| Chef      | `.rb`          | Ruby `key 'value'` | Per-language | — |
| Terraform | `.tf`          | HCL `=` assignments | Base (Ansible) | `var.`, `local.`, `${...}` |

## Detected Security Smells

1. **Hardcoded secret** — passwords, tokens, keys assigned as literal values
2. **Admin by default** — default admin/root usernames
3. **Empty password** — empty string or `undef` password values
4. **Suspicious comment** — TODO/FIXME/HACK comments indicating unfinished security work
5. **Use of HTTP** — unencrypted HTTP URLs
6. **Unrestricted IP address binding** — `0.0.0.0` bindings
7. **Use of weak cryptography** — MD5, SHA1, arcfour references

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

3. Download the oracle datasets for evaluation (see [External Datasets](#external-datasets)).

## Usage

**All commands must be run from the GUARD root directory using module syntax (`python -m`).**

### Basic detection

```bash
# Scan a folder of IaC files (auto-detects language by extension)
python -m src.guard -p <path-to-folder> -f -o output.csv

# Scan a single file
python -m src.guard -p <path-to-file> -o output.csv
```

### Per-language models

Per-language models improve precision for Puppet and Chef by using training data specific to that language:

```bash
python -m src.guard -p <path> -f -o output.csv --lang puppet
python -m src.guard -p <path> -f -o output.csv --lang chef
```

Without `--lang`, GUARD uses the base Ansible-trained models (which still work across languages, just with lower precision).

### Contextualization

Contextualization resolves variable references to check whether a value is parameterized (and therefore not hardcoded):

```bash
python -m src.guard -p <path> -f -c -o output.csv --lang puppet
```

### Output format

GUARD outputs a CSV with three columns (no header):

```
filename,smell_type,line_number
```

Example:
```
myfile.pp,hardcoded-secret,42
myfile.pp,use of http,17
myfile.pp,suspicious comment,5
```

## Evaluation

Evaluate GUARD output against the GLITCH oracle datasets:

```bash
# Puppet
python evaluation/evaluate_puppet.py \
    -g results/retrained_puppet_v2.csv \
    -o datasets/puppet/oracle-dataset-analysis/oracle-dataset.csv

# Chef
python evaluation/evaluate_chef.py \
    -g results/retrained_chef_v2.csv \
    -o datasets/chef-data/oracle-dataset-analysis/oracle_dataset.csv

# Terraform
python evaluation/evaluate_terraform.py \
    -g results/extended_terraform.csv \
    -o datasets/terraform/oracle-dataset/oracle-dataset.csv

# GLITCH comparison (RQ3)
python evaluation/evaluate_glitch.py \
    -g datasets/chef-data/oracle-dataset-analysis/GLITCH-chef-oracle.csv \
    -o datasets/chef-data/oracle-dataset-analysis/oracle_dataset.csv \
    --guard results/retrained_chef_v2.csv \
    -l chef
```

### Chef evaluation path

Point GUARD to the subdirectory containing the `.rb` files directly, not the parent — otherwise output filenames will have a prefix the evaluator can't match:

```bash
# Correct
python -m src.guard -p datasets/chef-data/oracle-dataset/recipes -f -o results/retrained_chef_v2.csv --lang chef

# Wrong — output will have 'recipes/' prefix
python -m src.guard -p datasets/chef-data/oracle-dataset -f -o results/retrained_chef_v2.csv --lang chef
```

## Model Training

The retrained per-language models are not included due to size. To reproduce them:

### 1. Build per-language datasets

```bash
cd data_extraction
python merge_per_language.py --lang puppet --balance
python merge_per_language.py --lang chef --balance
```

### 2. Train SetFit (suspicious comments)

```bash
python retrain_setfit.py --lang puppet
python retrain_setfit.py --lang chef
```

### 3. Train Label Propagation (hardcoded secrets)

```bash
python -m src.models_trainer -l --lang puppet \
    -htrain data_extraction/per_language/puppet/hardcoded_secrets_dataset.csv \
    -htest tests/hardcoded_secrets/parameters/sets/test_set.csv

python -m src.models_trainer -l --lang chef \
    -htrain data_extraction/per_language/chef/hardcoded_secrets_dataset.csv \
    -htest tests/hardcoded_secrets/parameters/sets/test_set.csv
```

### 4. Extract new training data (optional)

To extract training data from your own Chef/Puppet repositories:

```bash
python data_extraction/extract_training_data.py --tech chef <path-to-rb-files>
python data_extraction/extract_training_data.py --tech puppet <path-to-pp-files>
```

### Dependency pinning

SetFit retraining requires pinned versions:

```bash
pip install setfit==0.7.0 huggingface_hub==0.20.3 transformers==4.36.0 sentence-transformers==2.2.2
```

## Architecture

```
guard.py (dispatcher)
  ├── parsers/
  │   ├── ansible_parser.py   →  ParsedLine
  │   ├── puppet_parser.py    →  ParsedLine
  │   ├── chef_parser.py      →  ParsedLine
  │   └── terraform_parser.py →  ParsedLine
  ├── Detection logic (shared)
  │   ├── RegEx patterns (HTTP, 0.0.0.0, weak crypto)
  │   ├── Label Propagation (hardcoded secrets via Word2Vec)
  │   └── SetFit (suspicious comments)
  └── contextualization/
      ├── ansible_context.py   (Jinja2)
      ├── puppet_context.py    ($var resolution)
      └── terraform_context.py (var/local resolution)
```

The dispatcher routes files to the correct parser based on extension. Each parser converts raw lines into standardised `ParsedLine` objects. The shared detection logic then operates on `ParsedLine` objects regardless of source language — written once, works across all four languages.

## Project Structure

```
ThesisCode/
├── src/
│   ├── guard.py                    # Main detection pipeline
│   ├── parsers/                    # Language-specific parsers
│   ├── contextualization/          # Variable resolution per language
│   ├── vectorizer.py               # Word2Vec vectorization
│   ├── models_trainer.py           # ML model training
│   └── pretrainedModels/           # Trained models + embeddings
│       ├── puppet/                 # Puppet-only models
│       └── chef/                   # Chef-only models
├── datasets/                       # Oracle datasets (external, see below)
├── data_extraction/                # Training data extraction + merging
├── evaluation/                     # Evaluation scripts per language
├── results/                        # Detection output CSVs
├── tests/                          # LP evaluation test set
├── retrain_setfit.py               # SetFit retraining script
└── requirements.txt
```

## External Datasets

Download and place in `datasets/`:

| Dataset | Source | Link |
|---------|--------|------|
| Terraform oracle | Figshare | [terraform-GLITCH-replication-package.zip](https://figshare.com/articles/thesis/terraform-GLITCH-replication-package_zip/24574234) |
| Puppet oracle | Figshare | [GLITCH Replication Package](https://doi.org/10.6084/m9.figshare.19726603.v2) |

## Known Issues

- `vectorizer.py` line 11 needs `str(word)` cast to handle NaN values from CSVs with empty rows
- `src/parsers/__init__.py` must use `from . import X` (not `from .parsers import X`)
- `guard.py` importlib paths must be prefixed with `src.`
- `models_trainer.py` `trainedModel_path` must point to `./src/pretrainedModels`

## References

- **GUARD**: Grella et al., *Guard: Robust Detection of Ansible Security Smells*, 2024.
- **GLITCH**: Saavedra & Ferreira, *GLITCH: Automated Polyglot Security Smell Detection in Infrastructure as Code*, ASE 2022.
- **GLITCH extended**: Saavedra et al., *Polyglot Code Smell Detection for Infrastructure as Code with GLITCH*, ASE 2023.
- **Seven Sins**: Rahman, Parnin & Williams, *The Seven Sins: Security Smells in Infrastructure as Code Scripts*, ICSE 2019.