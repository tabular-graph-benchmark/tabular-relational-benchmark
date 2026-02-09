# Anonymous Submission Repository (KDD 2026)

This repository contains an anonymized implementation for the paper
currently under double-blind review at KDD 2026.

To preserve anonymity at submission time, we provide a minimal
reproducible version of the method. The repository will be updated
during the review period.

## Scope of Release
We release the core methodology rather than the full experimental
benchmark. Specifically, this repository will include:

- graph injection module for tabular models
- row-level graph construction
- feature-level graph construction
- example backbone integration
- runnable training script on a public dataset

The large-scale benchmark infrastructure (extensive dataset collection,
hyperparameter sweeps, and plotting utilities) used in the paper is not
required to verify the proposed method and therefore is not included in
this anonymized release.

## Reproducibility
The provided code will allow reviewers and researchers to execute a
complete training pipeline and verify the effect of graphification on a
representative tabular dataset.

All datasets used in the example are publicly available.

The repository will be expanded after the review process.
