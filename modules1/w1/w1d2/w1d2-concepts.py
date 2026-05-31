# Week 1 Day 2 Homework
# Exploring Supervised and Unsupervised Learning

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.cluster import KMeans


def choose_paradigm(has_labels: bool, goal: str) -> str:
    if has_labels and goal == "predict_category":
        return "Classification"
    elif has_labels and goal == "predict_number":
        return "Regresion"
    elif not has_labels and goal == "discover_groups":
        return "Clustering"
    elif not has_labels and goal == "compress_data":
        return "Dimensionality Reduction"
    else:
        return "Unknown"


print("=== Paradigm Decision ===")
examples = [
    (True, "predict_category"),
    (True, "predict_number"),
    (False, "discover_groups"),
    (False, "compress_data")
]

for has_labels, goal in examples:
    result = choose_paradigm(has_labels, goal)
    print(f"has_labels={has_labels}, goal='{goal}' -> {result}")


print("\n=== Supervised: KNN on Digits Dataset ===")

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
lowest_digit = np.argmin(precisions)
lowest_precision = precisions[lowest_digit]
print(f"Lowest precision digit: {lowest_digit} ({lowest_precision:.2f})")


print("\n=== Unsupervised: KMeans on Digits (No Labels) ===")

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

cluster_sizes = np.bincount(cluster_labels)
print("Cluster sizes:", cluster_sizes)

largest_cluster = np.argmax(cluster_sizes)
print(f"Largest cluster: Cluster {largest_cluster} ({cluster_sizes[largest_cluster]} samples)")

cluster_indexes = np.where(cluster_labels == largest_cluster)[0]

np.random.seed(42)
random_indexes = np.random.choice(cluster_indexes, size=5, replace=False)

plt.figure(figsize=(8, 2))
plt.gray()

for i, index in enumerate(random_indexes):
    plt.subplot(1, 5, i + 1)
    plt.imshow(digits.images[index])
    plt.axis("off")
    plt.title(f"C{largest_cluster}")

plt.tight_layout()
plt.savefig("digit_clusters.png")
print("[Saved digit_clusters.png]")

