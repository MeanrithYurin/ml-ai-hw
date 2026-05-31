# Week 3 Day 1 homework

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def distribution_stats(data: np.ndarray) -> dict:
    q3 = np.percentile(data, 75)
    q1 = np.percentile(data, 25)

    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "variance": np.var(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "range": np.max(data) - np.min(data),
        "iqr": q3 - q1,
    }


def print_stats_table(name, stats):
    print(f"\n=== Distribution Stats: {name} ===")
    print(f"Mean      : {stats['mean']:.4f}")
    print(f"Median    : {stats['median']:.4f}")
    print(f"Variance  : {stats['variance']:.4f}")
    print(f"Std Dev   : {stats['std']:.4f}")
    print(f"Min       : {stats['min']:.4f}")
    print(f"Max       : {stats['max']:.4f}")
    print(f"Range     : {stats['range']:.4f}")
    print(f"IQR       : {stats['iqr']:.4f}")


diabetes = load_diabetes()
X_diabetes = diabetes.data
feature_names = diabetes.feature_names

bmi_index = feature_names.index("bmi")
age_index = feature_names.index("age")

bmi_stats = distribution_stats(X_diabetes[:, bmi_index])
age_stats = distribution_stats(X_diabetes[:, age_index])

print_stats_table("BMI Feature", bmi_stats)
print_stats_table("Age Feature", age_stats)


print("\n=== Normal Distribution Check ===")
print("Feature          | Within mean+-1std | Within mean+-2std")
print("-----------------+-------------------+------------------")

normal_results = []

for i, name in enumerate(feature_names):
    col = X_diabetes[:, i]
    mu = np.mean(col)
    sigma = np.std(col)

    within_1 = np.mean(np.abs(col - mu) <= sigma) * 100
    within_2 = np.mean(np.abs(col - mu) <= 2 * sigma) * 100

    normal_results.append((name, within_1, within_2))
    print(f"{name:15s} | {within_1:8.1f}%          | {within_2:8.1f}%")

print("\nExpected normal data is about 68.3% and 95.4%")


print("\n=== Naive Bayes Spam Filter ===")

p_spam = 0.25
p_ham = 0.75

word_probs = {
    "offer": {"spam": 0.60, "ham": 0.05},
    "meeting": {"spam": 0.05, "ham": 0.35},
    "free": {"spam": 0.55, "ham": 0.02},
}


def naive_bayes_spam(words_present: list) -> float:
    spam_score = p_spam
    ham_score = p_ham

    for word in word_probs:
        if word in words_present:
            spam_score = spam_score * word_probs[word]["spam"]
            ham_score = ham_score * word_probs[word]["ham"]
        else:
            spam_score = spam_score * (1 - word_probs[word]["spam"])
            ham_score = ham_score * (1 - word_probs[word]["ham"])

    final_prob = spam_score / (spam_score + ham_score)
    return final_prob


tests = [
    ["offer", "free"],
    ["meeting"],
    ["offer", "meeting", "free"],
    ["free"],
]

for words in tests:
    prob = naive_bayes_spam(words)
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"Words {words} -> P(spam) = {prob * 100:.1f}% -> {label}")

print("\n=== Classifier Calibration ===")

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)

benign_probs = probabilities[:, 1]

avg_for_true_benign = np.mean(benign_probs[y_test == 1])
avg_for_true_malignant = np.mean(benign_probs[y_test == 0])

print(f"Avg P(benign) for truly BENIGN cases    : {avg_for_true_benign:.4f}")
print(f"Avg P(benign) for truly MALIGNANT cases : {avg_for_true_malignant:.4f}")

plt.figure(figsize=(8, 5))
plt.hist(benign_probs[y_test == 1], bins=10, alpha=0.7, label="true benign")
plt.hist(benign_probs[y_test == 0], bins=10, alpha=0.7, label="true malignant")
plt.xlabel("Predicted probability of benign")
plt.ylabel("Count")
plt.title("Calibration Analysis")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_analysis.png")
print("[Saved calibration_analysis.png]")
