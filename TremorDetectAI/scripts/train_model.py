import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import product
import os

# === CONFIG ===
DATA_FILE = '../data/processed_features.csv'
MODEL_OUTPUT_DIR = '../models'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(DATA_FILE)
df_binary = df.copy()
df_binary['label'] = df_binary['label'].apply(lambda x: 'tremor' if x in ['rest_tremor', 'intention_tremor'] else 'no_tremor')

X = df_binary.drop(columns=['label'])
y = df_binary['label']

# === Calculate Class Weights ===
class_counts = y.value_counts()
total_samples = len(y)
class_weights = {
    'no_tremor': total_samples / (len(class_counts) * class_counts['no_tremor']),
    'tremor': total_samples / (len(class_counts) * class_counts['tremor'])
}
print(f"Class distribution: {class_counts.to_dict()}")
print(f"Class weights: {class_weights}")

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Hyperparameter Grid ===
n_estimators_list = [50, 100, 200]
max_depth_list = [None, 10, 20]
min_samples_split_list = [2, 5, 10]

# To store results
results = []

# === Train Models Across Grid ===
for n, d, s in product(n_estimators_list, max_depth_list, min_samples_split_list):
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=d,
        min_samples_split=s,
        class_weight=class_weights,  # Add class weights
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    # Save model
    model_name = f"rf_n{n}_d{d if d is not None else 'None'}_s{s}_weighted.joblib"
    joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, model_name))

    results.append({
        'n_estimators': n,
        'max_depth': d,
        'min_samples_split': s,
        'train_acc': acc_train,
        'test_acc': acc_test,
        'model_file': model_name
    })

# === Convert Results to DataFrame ===
results_df = pd.DataFrame(results)
results_df.to_csv('gridsearch_results_weighted.csv', index=False)

# ---- Evaluate on TRAIN set ----
print("Train Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Train Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

# ---- Evaluate on TEST set ----
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# === Plot Heatmap of Test Accuracy ===
pivot = results_df.pivot_table(
    index='n_estimators',
    columns='min_samples_split',
    values='test_acc',
    aggfunc='max'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title("Test Accuracy Heatmap (max across max_depth) - With Class Weights")
plt.ylabel("n_estimators")
plt.xlabel("min_samples_split")
plt.tight_layout()
plt.savefig("accuracy_heatmap_weighted.png")
plt.show()

print("\nTraining complete with class weights. Models saved in 'saved_models/' and results in 'gridsearch_results_weighted.csv'")