from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_FILE = 'processed_features.csv'

# === Load Data ===
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=['label'])
y = df['label']

# Convert labels to numeric
label_map = {'rest_tremor': 0, 'intention_tremor': 1, 'no_tremor': 2}
df['label_num'] = df['label'].map(label_map)

# Perform PCA
X = df.drop(columns=['label', 'label_num'])
y = df['label_num']
X_pca = PCA(n_components=2).fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA Projection of Feature Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
plt.show()
