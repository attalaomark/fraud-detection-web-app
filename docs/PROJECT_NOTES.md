# Project Notes

## Source Alignment

This repository reflects the **final project implementation**, not every idea that appeared during early planning.

### Early Machine Learning Canvas

The early canvas proposed:

- Kaggle's "Bank Transaction Dataset for Fraud Detection";
- clustering;
- K-Means / Isolation Forest;
- Firebase / Firestore; and
- anomaly-oriented evaluation such as Silhouette Score and Davies-Bouldin Index.

### Final Report and Implementation

The final report and current application instead use:

- labelled mobile-money transaction data from Mendeley Data;
- supervised binary classification;
- a Decision Tree final model;
- StandardScaler;
- SMOTE;
- Optuna tuning;
- Flask; and
- a Supabase-backed transaction application with OTP verification.

For that reason, the public README documents the **final report and code path**. The early canvas is retained only as project-history context.

## Academic Results vs Application Behaviour

The report records several modelling stages with different metric values. In particular, the base Decision Tree result is much stronger than the final post-tuning fraud F1-score. The repository therefore keeps both results visible rather than presenting only the best-looking number.

## Portfolio Guidance

Before publishing:

1. rotate any credentials used by the original project;
2. verify `.env` is ignored;
3. add personal/group-approved GitHub profile links;
4. replace editable Google Docs or Canva links with view-only links;
5. optionally add screenshots or a short demo GIF; and
6. keep the academic disclaimer visible.
