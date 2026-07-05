# Assignment 4 - Generative Models with GANs

Students:
- Yehonatan Segal, 209359801
- Omer Onn, 318910759

## Overview

This project implements generative models for the Adult tabular dataset:

1. Standard GAN
2. Conditional GAN (cGAN)
3. Discrete categorical generation experiments
4. Mode-collapse experiment
5. Open-ended Feature Matching GAN experiment

The implementation is written in Python using PyTorch.

## Project structure

```text
Assignment4/
├── data/
│   └── raw/
│       └── adult.arff
├── src/
│   └── reusable Python modules
├── scripts/
│   └── executable scripts for each stage
├── results/
│   └── generated CSV/TXT/PNG outputs
├── models/
│   └── trained model artifacts and training plots
├── Assignment4_pipeline.ipynb
├── Assignment4_Report.pdf
├── requirements.txt
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the project

Run commands from the `Assignment4/` root directory.

The full pipeline can be executed from the notebook:

```text
Assignment4_pipeline.ipynb
```

Alternatively, the scripts can be run directly:

```bash
python scripts/01_inspect_dataset.py
python scripts/02_create_splits.py
python scripts/03_preprocess_splits.py
python scripts/04_train_gan.py
python scripts/05_train_cgan.py
python scripts/06_evaluate_models.py
python scripts/07_discrete_feature_experiments.py
python scripts/08_mode_collapse.py
python scripts/09_open_ended_feature_matching.py
```

## Outputs

Generated outputs are saved under:

```text
results/
```

Training curves and model artifacts are saved under:

```text
models/
```

The final report is included as:

```text
Assignment4_Report.pdf
```

## Notes

The submitted ZIP already contains the generated results and figures used in the report.  
Re-running all scripts may take time because the pipeline includes GAN training and Random Forest evaluation.