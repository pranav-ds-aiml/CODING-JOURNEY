----------------------------------
CHALLENGE: EXTREME CLASS IMBALANCE
---------------------------------

Real credit card fraud:
- Legitimate: 99.8%
- Fraud: 0.2%

This creates unique challenges!
-------------------------------
IMPORTANT: For fraud detection:
-------------------------------
- Accuracy is MISLEADING (99% by always predicting legitimate)
- Recall is CRITICAL (must catch fraud!)
- Precision is important (don't block too many good transactions)
- F1-Score balances both
- AUC gives overall performance

-----------------
THRESHOLD TUNING:
-----------------

Default threshold = 0.5
But for fraud detection, we might want different threshold:

- Lower threshold (0.3): Catch more fraud, more false alarms
- Higher threshold (0.7): Fewer false alarms, miss some fraud

FRAUD DETECTION SUMMARY:

CHALLENGE:
- Extreme class imbalance (99:1)
- Must catch fraud (high recall)
- Minimize false alarms (good precision)

MODELS TESTED:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

BEST MODEL: {best_model_name}
- Recall: {best_recall:.4f} (catches {best_recall*100:.1f}% of fraud)
- Optimal Threshold: {best_threshold}
- Final F1-Score: {final_f1:.4f}

KEY LEARNINGS:
━━━━━━━━━━━━━━
✓ Accuracy is misleading on imbalanced data
✓ Recall is critical for fraud detection
✓ class_weight='balanced' helps with imbalance
✓ Threshold tuning is essential
✓ Ensemble methods (RF, GB) perform best
✓ Always use stratify=y when splitting

BUSINESS VALUE:
━━━━━━━━━━━━━━
- Catching {fraud_caught}/{total_fraud} fraudulent transactions
- Only {legitimate_blocked} false alarms
- Can be deployed in production!

FILES CREATED:
━━━━━━━━━━━━━━
1. credit_card_data.csv
2. fraud_01_eda.png
3. fraud_02_confusion_matrices.png
4. fraud_03_roc_curves.png
5. fraud_04_feature_importance.png
6. fraud_05_threshold_analysis.png
7. fraud_detection_model.pkl