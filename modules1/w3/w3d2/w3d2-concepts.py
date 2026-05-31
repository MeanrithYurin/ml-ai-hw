# Week 3 Day 2 Homework

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mape(y_true, y_pred):
    small = 1e-10
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + small)))


def binary_cross_entropy(y_true, y_pred):
    small = 1e-15
    y_pred = np.clip(y_pred, small, 1 - small)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


print("=== Loss Function Comparison ===")

y_true = np.array([1, 2, 3, 4, 5], dtype=float)
clean_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
noisy_pred = np.array([1.2, 3.5, 0.5, 6.0, 8.0])

bce_true = np.array([1, 0, 1, 1, 0])
bce_clean = np.array([0.9, 0.1, 0.8, 0.75, 0.2])
bce_noisy = np.array([0.6, 0.7, 0.4, 0.5, 0.8])

print(f"{'Loss':22} {'Clean':>12} {'Noisy':>12}")
print("-" * 48)
print(f"{'MSE':22} {mse(y_true, clean_pred):12.4f} {mse(y_true, noisy_pred):12.4f}")
print(f"{'RMSE':22} {rmse(y_true, clean_pred):12.4f} {rmse(y_true, noisy_pred):12.4f}")
print(f"{'MAE':22} {mae(y_true, clean_pred):12.4f} {mae(y_true, noisy_pred):12.4f}")
print(f"{'MAPE':22} {mape(y_true, clean_pred):11.2f}% {mape(y_true, noisy_pred):11.2f}%")
print(f"{'Binary Cross Entropy':22} {binary_cross_entropy(bce_true, bce_clean):12.4f} {binary_cross_entropy(bce_true, bce_noisy):12.4f}")


print("\n=== Outlier Sensitivity ===")

outlier_values = [1, 2, 4, 8, 16, 32]
mse_list = []
mae_list = []

for outlier in outlier_values:
    yt = np.zeros(20)
    yp = np.ones(20)
    yp[-1] = outlier

    m = mse(yt, yp)
    a = mae(yt, yp)
    mse_list.append(m)
    mae_list.append(a)

    print(f"Outlier={outlier:<2} MSE={m:.2f}, MAE={a:.2f}")

plt.figure(figsize=(7, 5))
plt.plot(outlier_values, mse_list, marker="o", label="MSE")
plt.plot(outlier_values, mae_list, marker="o", label="MAE")
plt.xlabel("Outlier Magnitude")
plt.ylabel("Loss Value")
plt.title("Outlier Sensitivity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outlier_sensitivity.png")
plt.close()


print("\n=== Bias-Variance Diagnosis ===")

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data,
    housing.target,
    test_size=0.3,
    random_state=42
)

for alpha in [0, 1, 1000]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    train_loss = mse(y_train, model.predict(X_train))
    test_loss = mse(y_test, model.predict(X_test))

    if train_loss > 0.65 and test_loss > 0.65:
        label = "Underfitting"
    elif test_loss - train_loss > 0.15:
        label = "Overfitting"
    else:
        label = "Good Fit"

    print(f"Ridge(alpha={alpha:<4}) Train MSE={train_loss:.4f}, Test MSE={test_loss:.4f} -> {label}")


print("\n=== Loss Curve Monitoring ===")

np.random.seed(0)

diabetes = load_diabetes()
X = diabetes.data[:, [2]]
y = diabetes.target

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

y_train_noisy = y_train + np.random.randn(len(y_train)) * 50

w = np.random.randn(1)
b = 0.0
learning_rate = 0.05
batch_size = 32
epochs = 200

train_losses = []
val_losses = []

for epoch in range(epochs):
    order = np.random.permutation(len(X_train))

    for start in range(0, len(X_train), batch_size):
        batch_id = order[start:start + batch_size]
        X_batch = X_train[batch_id]
        y_batch = y_train_noisy[batch_id]

        pred = X_batch @ w + b
        error = pred - y_batch

        dw = (2 / len(X_batch)) * (X_batch.T @ error)
        db = (2 / len(X_batch)) * np.sum(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    train_pred = X_train @ w + b
    val_pred = X_val @ w + b

    train_loss = mse(y_train_noisy, train_pred)
    val_loss = mse(y_val, val_pred)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: train loss={train_loss:.2f}, val loss={val_loss:.2f}")

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train loss noisy")
plt.plot(val_losses, label="Validation loss clean")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curves.png")
plt.close()


print("\nSaved outlier_sensitivity.png")
print("Saved loss_curves.png")
print(f"Final model: w={w[0]:.4f}, b={b:.4f}")
