# Kaggle House Prices

This project trains a local baseline for the Kaggle House Prices competition.

## Run

Use the project virtual environment:

```bash
.venv/bin/python train_model.py
```

The script writes:

- `outputs/submission.csv` - Kaggle submission file
- `outputs/cv_results.csv` - 5-fold RMSLE validation results by Ridge alpha
- `outputs/ridge_coefficients.csv` - fitted coefficients for inspection

The model uses log-transformed `SalePrice`, house-specific feature engineering,
`scikit-learn` preprocessing pipelines, one-hot encoding, standardized numeric
features, K-fold validation, and Ridge regression.

## AI Usage Statement

AI assistance was used to help draft, refactor, document, and validate the
training code and dataset summary workflow. The modeling approach, generated
files, and results should still be reviewed before Kaggle submission.
