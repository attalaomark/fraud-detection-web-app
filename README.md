# FINS - Machine Learning Fraud Detection Web App

A portfolio-ready academic group project that integrates a **machine learning fraud-detection model** into a simulated digital-banking transaction workflow. The application evaluates each transfer, flags suspicious transactions, and adds an OTP verification step before a flagged transaction can proceed.

> **Academic Group Project**  
> S1 Teknologi Sains Data, Universitas Airlangga - 2025  
> Developed collaboratively by four students from the Data Science Technology Program.  
> This repository is intended for education, demonstration, and portfolio purposes. It is **not a production banking system**.

---

## Project Overview

Digital transaction fraud is difficult to handle using static rules alone because fraud patterns can change and highly imbalanced data can make detection unreliable. This project explores a machine-learning-based approach and integrates the resulting model into a Flask web application.

The final project workflow covers:

- exploratory data analysis (EDA);
- data cleaning and feature engineering;
- handling class imbalance with SMOTE;
- comparison of multiple classification algorithms;
- Decision Tree model selection and Optuna tuning;
- fraud probability scoring during transactions;
- configurable fraud thresholding;
- OTP-based user verification for suspicious transfers;
- transaction history and account information; and
- relational data storage through Supabase.

---

## Application Flow

```text
Login
  |
Dashboard
  |
Create Transfer
  |
Confirm Transaction
  |
Build Transaction Features
  |
StandardScaler -> Decision Tree -> Fraud Probability
  |
  +---------------------------+
  |                           |
Below threshold          At/above threshold
  |                           |
Transaction proceeds     Fraud warning
                              |
                     "Was this you?"
                       /            \
                     No              Yes
                     |                |
                  Cancel          Send OTP
                                      |
                                Verify OTP
                                      |
                             Transaction proceeds
```

The deployed application uses a configurable fraud threshold. The default value in this cleaned portfolio version is **0.80**.

---

# Machine Learning

## Machine Learning Task

The final project is formulated as a **supervised binary classification** problem:

- `0` - Non-Fraud Transaction
- `1` - Fraud Transaction

The model is designed to score a transaction during the transfer flow so the application can respond immediately when a transaction appears suspicious.

---

## Dataset

The final report identifies the modelling dataset as:

**`MoMTSim_20240722202413_1000_dataset`**

The dataset was obtained from **Mendeley Data** and represents labelled synthetic mobile-money transactions.

The raw dataset itself is **not included in this repository**. This avoids unnecessarily redistributing external data and keeps the repository focused on the application and trained artifacts.

> **Important project-history note:** an earlier Machine Learning Canvas described a different Kaggle dataset and an unsupervised clustering concept. The implementation and final report evolved into the supervised Decision Tree pipeline documented here. This README follows the **final implementation and report**.

---

## Features Used by the Web App

The inference path in `app.py` constructs these seven model inputs:

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `oldBalInitiator` | Sender balance before the transaction |
| `newBalInitiator` | Sender balance after the transaction |
| `oldBalRecipient` | Recipient balance before the transaction |
| `newBalRecipient` | Recipient balance after the transaction |
| `isFlaggedFraud` | Binary flag derived from a high transaction amount |
| `waktu` | Time-of-day category derived from transaction time |

The numerical balance and amount features are transformed using the saved `StandardScaler` before prediction.

---

## Time Feature Engineering

The report groups transaction time into four categories:

- **Pagi:** 05:00-10:59
- **Siang:** 11:00-14:59
- **Sore:** 15:00-18:59
- **Malam:** 19:00-04:59

The web application recreates this feature from the transaction time before inference.

---

## Data Preparation

The modelling workflow documented in the final report includes:

1. **Data Cleaning** - removing financially inconsistent observations.
2. **Time Feature Engineering** - transforming the original time/step information into time-of-day categories.
3. **Encoding** - converting the time category into numerical form.
4. **Currency Conversion** - the academic analysis used a fixed assumption of `1 USD = IDR 16,000`.
5. **Scaling** - numerical features were standardized using `StandardScaler`.
6. **Train/Validation/Test Split** - `70% / 15% / 15%`.
7. **SMOTE** - applied **only to the training data** with `sampling_strategy=0.2`.

The final cleaned dataset reported **1,625,692 observations**.

---

# Model Development

## Model Comparison

The final report compares the following classification algorithms:

| Model | Binary F1-score | Accuracy |
|---|---:|---:|
| **Decision Tree** | **0.860632** | **0.999204** |
| Extra Trees | 0.831224 | 0.999016 |
| Random Forest | 0.825243 | 0.998967 |
| CatBoost | 0.717340 | 0.998048 |
| LightGBM | 0.711346 | 0.997987 |
| K-Nearest Neighbors | 0.691415 | 0.997818 |
| XGBoost | 0.667412 | 0.997560 |
| Logistic Regression | 0.263945 | 0.987283 |
| Naive Bayes | 0.050446 | 0.955076 |

Because fraud cases are a very small minority, **F1-score for the fraud class is more informative than accuracy alone**.

Decision Tree was selected as the final model because it achieved the strongest reported base-model F1-score while remaining simple and interpretable.

---

## Hyperparameter Tuning

The selected Decision Tree was tuned using:

- **Optuna**
- Tree-structured Parzen Estimator (**TPE**)
- **10 trials**
- **10-fold cross-validation**
- Binary F1-score as the optimization metric

The report identifies **Trial 6** as the best Optuna trial, with a mean cross-validation F1-score of **0.2467**.

---

# Final Model Evaluation

The final report gives the following test-set classification results:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-Fraud (`0`) | 1.00 | 0.98 | 0.99 | 243,240 |
| Fraud (`1`) | 0.13 | 1.00 | 0.23 | 614 |
| **Accuracy** |  |  | **0.98** | **243,854** |

### Interpretation

The final model achieved **recall = 1.00 for fraud**, meaning all fraud observations in the reported test set were detected.

However, fraud precision was only **0.13**, indicating many false positives.

This limitation is important. For a real financial system, the model should not be considered production-ready without further threshold calibration, model improvement, probability calibration, cost-sensitive evaluation, and validation on representative real-world data.

The report evaluated the **0.80 fraud threshold** as a practical trade-off to reduce false positives. This cleaned application therefore uses `0.8` as its default threshold while allowing it to be changed through an environment variable.

---

# Feature Importance

The Decision Tree feature-importance analysis found:

- `amount` contributed roughly **53%** of model importance;
- `newBalanceRecipient` contributed roughly **24%**;
- sender balance features also contributed to the prediction; and
- `oldBalanceRecipient`, `isFlaggedFraud`, and `waktu` had comparatively little importance in the reported final model.

These results describe the trained academic model and should not be generalized beyond the dataset used in this project.

---

# Web Application

## Application Features

The Flask application implements a simulated banking experience with:

- User login
- Dashboard with account balance and transaction overview
- Transfer form
- Transaction confirmation
- Real-time machine learning inference
- Suspicious-transaction warning
- OTP delivery through SMTP
- OTP verification and resend flow
- Transaction success/result page
- Profile/account information
- Transaction history
- Supabase-backed account and transaction data
- Model/database health-check endpoints

---

# Database Design

The academic application uses three main relational entities.

### `nasabah`

Stores user/account-holder information such as:

- account ID
- password/PIN
- first name
- last name
- email
- birth date
- location
- occupation

### `rekening`

Stores bank-account information such as:

- account number
- account balance
- account holder relationship

### `transaction`

Stores:

- transaction ID
- transaction timestamp
- IP address
- sender account
- recipient account
- transaction amount
- transaction message/detail

The application connects to the database layer through the **Supabase Python client**.

---

# Tech Stack

## Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn / SMOTE
- Optuna
- Decision Tree Classifier

## Web Application

- Flask
- Jinja2
- HTML
- CSS
- Supabase
- python-dotenv
- SMTP / Gmail for OTP delivery
- Joblib / Pickle

---

# Repository Structure

```text
.
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── improved_fraud_detection_ui.html
├── docs/
│   ├── MODEL_CARD.md
│   └── PROJECT_NOTES.md
├── static/
│   ├── style.css
│   └── asset/
│       └── logo.png
└── templates/
    ├── confirm_transaction.html
    ├── dashboard.html
    ├── enter_otp.html
    ├── fraud.html
    ├── history.html
    ├── info.html
    ├── login.html
    ├── result.html
    └── transaction.html
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/fraud-detection-web-app.git
cd fraud-detection-web-app
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy `.env.example` to `.env`.

### Windows

```bash
copy .env.example .env
```

### macOS/Linux

```bash
cp .env.example .env
```

Then configure the required values inside `.env`.

| Variable | Required | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | Recommended | Flask session secret |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase API key |
| `EMAIL_HOST` | Optional | SMTP host |
| `EMAIL_PORT` | Optional | SMTP port |
| `EMAIL_USERNAME` | For email OTP | SMTP sender account |
| `EMAIL_APP_PASSWORD` | For email OTP | SMTP app password |
| `FRAUD_THRESHOLD` | Optional | Fraud probability threshold, default `0.8` |
| `FLASK_DEBUG` | Optional | Enable/disable Flask debug mode |

## 5. Run the Application

```bash
python app.py
```

Then open the local address displayed by Flask.

---

# Security

This public version intentionally excludes:

- `.env`;
- hard-coded email credentials;
- hard-coded Supabase credentials;
- local absolute model paths;
- Python cache files; and
- previous `.git` history that may contain sensitive values.

**Before publishing publicly, rotate/revoke any credential that has ever appeared in the original project files or Git history.**

Do not commit `.env`.

Only `.env.example` should be tracked.

---

# Project Team

This project was developed collaboratively as a **group project by four students** from the S1 Teknologi Sains Data Program, Universitas Airlangga.

### Kelompok 11

| Name | Student ID |
|---|---|
| **Reinhart Ananda Siswadi** | 164221046 |
| **Andreas Hendra Herwanto** | 164221064 |
| **Giovanni Eki Pamungkas** | 164221090 |
| **Attala Omar Kareem** | 164221107 |

**Program:** S1 Teknologi Sains Data  
**Faculty:** Fakultas Teknologi Maju dan Multidisiplin  
**University:** Universitas Airlangga  
**Course:** Machine Learning

---

# Project Resources

The original academic project also includes:

- deployed web application;
- Google Colab modelling notebook;
- original group GitHub repository; and
- Machine Learning Canvas.

For public portfolio purposes, only include links that have been approved by all group members.

For Google Docs and Canva, use **view-only links** rather than editable links.

---

# Limitations and Future Work

Potential improvements include:

- reducing false positives while preserving high fraud recall;
- comparing calibrated probability thresholds using precision-recall curves;
- evaluating PR-AUC and cost-sensitive metrics;
- testing alternative imbalance-handling strategies;
- probability calibration;
- retraining on newer or real-world transaction data;
- data-drift and model-performance monitoring;
- stronger authentication and credential handling;
- automated tests and CI; and
- deployment using production-grade WSGI infrastructure.

---

# Disclaimer

This project is an **academic prototype**.

Predictions must not be used as the sole basis for real financial decisions, account blocking, or fraud accusations.

---

# License

No open-source license is included by default.

Add a license only after all group members agree on reuse and redistribution terms.
