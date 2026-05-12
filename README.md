# Extending GUARD: Detecting Security Smells Across IaC Languages

Bachelor's thesis project — University of Twente, 2026.

GUARD is a machine learning-based tool for detecting security smells in Ansible Infrastructure as Code files. This project extends GUARD to support **Terraform** and **Puppet**.

## Project Structure

```
ThesisCode/
├── GUARD/                          # GUARD codebase (original + extensions)
│   ├── src/
│   │   ├── requirements.txt.       # All the required versions to run the python files successfully and not run into dependency issues
│   │   ├── guard.py                # Main detection pipeline
│   │   ├── contexualizationPhase.py # Variable reference resolution
│   │   ├── vectorizer.py           # Word2Vec vectorization for key names
│   │   └── models_trainer.py       # ML model training (Label Propagation + SetFit)
│   ├── datasets/                   # GUARD's original Ansible training data
│   └── tests/                      # Evaluation experiments
├── datasets/                       # External datasets (not committed — see below)
│   ├── glitch-terraform/
│   └── glitch-puppet/
└── README.md
```

## Setup

1. Clone this repo.
2. Unzip the pretrained Word2Vec model before running:
   ```
   cd GUARD/src/pretrainedModels
   tar -xzf googleNews-vectors-negative300.bin.tar.gz
   ```
3. Download the external datasets (see below).

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