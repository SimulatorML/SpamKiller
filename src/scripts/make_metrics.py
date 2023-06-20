import pandas as pd
import yaml
import matplotlib.pyplot as plt
from metrics import recall_at_precision, recall_at_specificity, bootstrap_metric, curves
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# Read data
with open("./config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)
    df_cleaned_spam_and_not_spam_path = config["df_cleaned_spam_and_not_spam"]

# Load data
df_cleaned_spam_and_not_spam = pd.read_csv(
    df_cleaned_spam_and_not_spam_path, sep=";"
)  # Поменяйте на реальный путь к вашему файлу

# Split data(80% train, 20% test plus shuffle)
train_df, test_df = train_test_split(
    df_cleaned_spam_and_not_spam, test_size=0.2, random_state=42, shuffle=True
)

# Load test data
true_labels = test_df["label"].to_numpy()
pred_scores = test_df["pred_scores"].to_numpy()

# Calculate metrics
recall_at_prec = recall_at_precision(true_labels, pred_scores)
recall_at_spec = recall_at_specificity(true_labels, pred_scores)

# Calculate confidence bounds
lcb, ucb = bootstrap_metric(recall_at_precision, true_labels, pred_scores)

# Plot curves
_, _ = curves(true_labels, pred_scores)

# Precision-Recall curve
pr_display = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
plt.figure(figsize=(12, 6))
plt.suptitle("Precision-Recall Curve")
pr_display.plot()
plt.show()

# ROC curve
roc_display = RocCurveDisplay.from_predictions(true_labels, pred_scores)
plt.figure(figsize=(12, 6))
plt.suptitle("ROC Curve")
roc_display.plot()
plt.show()

# Print metrics
print(f"Recall at precision: {recall_at_prec}")
print(f"Recall at specificity: {recall_at_spec}")
print(f"Confidence bounds: {lcb}, {ucb}")
