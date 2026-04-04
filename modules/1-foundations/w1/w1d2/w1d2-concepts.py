# Week 1 Day 2 Homework
# Supervised vs Unsupervised Learning

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score


# =========================================
# 1. Paradigm Decision Function
# =========================================
def choose_paradigm(has_labels: bool, goal: str) -> str:
    if has_labels:
        if goal == "predict_category":
            return "Classification"
        elif goal == "predict_number":
            return "Regression"
    else:
        if goal == "discover_groups":
            return "Clustering"
        elif goal == "compress_data":
            return "Dimensionality Reduction"

    return "Unknown"


print("=== Paradigm Decision ===")
tests = [
    (True, "predict_category"),
    (True, "predict_number"),
    (False, "discover_groups"),
    (False, "compress_data")
]

for has_labels, goal in tests:
    result = choose_paradigm(has_labels, goal)
    print(f"has_labels={has_labels}, goal='{goal}' → {result}")


# =========================================
# 2. Supervised Classification Task (KNN)
# =========================================
print("\n=== Supervised: KNN on Digits Dataset ===")

# Load dataset
data = load_digits()
X = data.data
y = data.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Find lowest precision digit
report = classification_report(y_test, y_pred, output_dict=True)

lowest_digit = None
lowest_precision = 1.0

for digit in map(str, range(10)):
    if digit in report:
        precision = report[digit]["precision"]
        if precision < lowest_precision:
            lowest_precision = precision
            lowest_digit = digit

print(f"Lowest precision digit: {lowest_digit} ({lowest_precision:.2f})")

# Comment:
# Digit 8 often has the lowest precision because it looks similar to 3 or 9,
# especially due to its loops and shape variations.


# =========================================
# 3. Unsupervised Clustering Task (KMeans)
# =========================================
print("\n=== Unsupervised: KMeans on Digits (No Labels) ===")

# Use data without labels
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Cluster sizes
cluster_sizes = np.bincount(cluster_labels)
print("Cluster sizes:", cluster_sizes)

# Largest cluster
largest_cluster = np.argmax(cluster_sizes)
print(f"Largest cluster: Cluster {largest_cluster} ({cluster_sizes[largest_cluster]} samples)")

# Get indices of largest cluster
indices = np.where(cluster_labels == largest_cluster)[0]

# Pick 5 random samples
random_indices = np.random.choice(indices, 5, replace=False)

# Plot images
plt.figure(figsize=(10, 2))
for i, idx in enumerate(random_indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(data.images[idx], cmap="gray")
    plt.axis("off")

plt.suptitle("Samples from Largest Cluster")
plt.savefig("digit_clusters.png")
plt.close()

print("[Saved digit_clusters.png]")


# =========================================
# 4. Comparison Analysis (Comments)
# =========================================
