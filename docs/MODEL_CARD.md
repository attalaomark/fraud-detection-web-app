# Model Card - FINS Fraud Detection

## Model Summary

The application uses a saved **Decision Tree classifier** to classify digital transactions as fraud or non-fraud.

- Task: supervised binary classification
- Positive class: fraud (`1`)
- Negative class: non-fraud (`0`)
- Inference artifact: `model.pkl`
- Scaler artifact: `scaler.pkl`
- Default application threshold: `0.80`

## Input Features

The application builds seven features for each transaction:

1. `amount`
2. `oldBalInitiator`
3. `newBalInitiator`
4. `oldBalRecipient`
5. `newBalRecipient`
6. `isFlaggedFraud`
7. `waktu`

The first five numerical features are transformed with the saved scaler.

## Training Workflow

The final academic report describes the following pipeline:

```text
EDA
 -> cleaning
 -> time feature engineering
 -> encoding
 -> numerical scaling
 -> 70/15/15 train-validation-test split
 -> SMOTE on train only (sampling_strategy=0.2)
 -> base-model comparison
 -> Decision Tree selection
 -> Optuna TPE tuning
 -> final test evaluation
 -> export model for Flask integration
```

## Reported Base-Model Result

Decision Tree:

- Binary F1-score: `0.860632`
- Accuracy: `0.999204`

## Reported Final Test Result

Fraud class:

- Precision: `0.13`
- Recall: `1.00`
- F1-score: `0.23`
- Support: `614`

Overall accuracy: `0.98`.

The final model therefore prioritizes detecting fraud but produces a substantial number of false positives.

## Threshold

The report discusses `0.80` as a threshold intended to improve the balance between fraud sensitivity and false positives. The web application exposes this value as:

```text
FRAUD_THRESHOLD=0.8
```

## Feature Importance

The report describes `amount` as the most important feature (~53%) and `newBalanceRecipient` as the second most important (~24%).

## Intended Use

This model is intended for academic and portfolio demonstration of machine-learning-to-web integration.

## Out-of-Scope Use

Do not use this model as the sole basis for real account blocking, fraud accusations, transaction rejection, or other production financial-risk decisions.

## Known Limitations

- Highly imbalanced problem.
- Low final fraud precision.
- Dataset is synthetic.
- Performance may not transfer to real financial traffic.
- Saved preprocessing/model artifacts must remain aligned with the feature order in `app.py`.
- The application is a prototype rather than a hardened financial system.
