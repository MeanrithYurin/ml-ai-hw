
# Week 2 Day 2 Homework
# Gradient Descent from Scratch
import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(f, w, epsilon=1e-5):
    w = np.array(w, dtype=float)
    grad = np.zeros_like(w)

    for i in range(len(w)):
        small_change = np.zeros_like(w)
        small_change[i] = epsilon

        plus = f(w + small_change)
        minus = f(w - small_change)

        grad[i] = (plus - minus) / (2 * epsilon)

    return grad


def f_check(w):
    return w[0]**2 + 2 * w[1]**2 + w[0] * w[1]


def grad_check(w):
    return np.array([
        2 * w[0] + w[1],
        4 * w[1] + w[0]
    ])


print("=== Numerical Gradient Check ===")

w = np.array([1.0, 2.0])
num_grad = numerical_gradient(f_check, w)
ana_grad = grad_check(w)

print("Function: f(w) = w1^2 + 2*w2^2 + w1*w2")
print("At w =", w)
print("Numerical gradient:", num_grad)
print("Analytical gradient:", ana_grad)
print("Max different:", np.max(np.abs(num_grad - ana_grad)))
print()


def gradient_descent_2d(f, grad_f, w_init, lr, n_steps):
    w = np.array(w_init, dtype=float)
    history = []

    for step in range(n_steps):
        loss = f(w)
        history.append(loss)

        grad = grad_f(w)
        w = w - lr * grad

    return w, history


def f_2d(w):
    return w[0]**2 + 5 * w[1]**2


def grad_2d(w):
    return np.array([2 * w[0], 10 * w[1]])


print("=== 2D Gradient Descent ===")

learning_rates = [0.01, 0.1, 0.19]
w_init = [3.0, 2.0]
all_history = {}

for lr in learning_rates:
    final_w, history = gradient_descent_2d(f_2d, grad_2d, w_init, lr, 100)
    all_history[lr] = history

    final_loss = f_2d(final_w)

    print("lr =", lr)
    print("final w =", final_w)
    print("final loss =", final_loss)
    print()


plt.figure(figsize=(8, 5))

for lr in learning_rates:
    plt.plot(all_history[lr], label="lr=" + str(lr))

plt.title("Gradient Descent Loss Compare")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("gd_comparison.png")
plt.close()

print("[Saved gd_comparison.png]")
print()

print("=== Linear Regression Training ===")

np.random.seed(0)

X = np.linspace(0, 5, 80)
noise = np.random.normal(0, 0.5, size=X.shape)
y = 2.5 * X - 1 + noise

w = 0.0
b = 0.0

learning_rate = 0.05
n_epochs = 200
n = len(X)

losses = []

print("Epoch     w        b        loss")

for epoch in range(n_epochs + 1):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    if epoch % 20 == 0:
        print(f"{epoch:5d}  {w:7.4f}  {b:7.4f}  {loss:8.4f}")

    dw = (2 / n) * np.sum((y_pred - y) * X)
    db = (2 / n) * np.sum(y_pred - y)

    w = w - learning_rate * dw
    b = b - learning_rate * db

print()
print("Final learned w =", w)
print("Final learned b =", b)
print("True w = 2.5")
print("True b = -1.0")

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="data")
plt.plot(X, w * X + b, label="fit line")
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("linear_fit.png")
plt.close()

print("[Saved linear_fit.png]")

