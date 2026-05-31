# Week 2 Day 1 Homework
# Vectors, Matrices and Linear Algebra for ML
import numpy as np

print("=== Manual Dot Product ===")

def manual_dot(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectors must have same size")
    return np.sum(a * b)

vector_tests = [
    (np.array([1., 2., 3.]), np.array([4., 5., 6.])),
    (np.array([0.5, 1.5, 2.5, 3.5]), np.array([1., 2., 3., 4.])),
    (np.array([2., 4.]), np.array([3., 5.])),
]

for a, b in vector_tests:
    my_result = manual_dot(a, b)
    np_result = np.dot(a, b)
    print(f"a = {a}, b = {b}")
    print("manual_dot result:", my_result)
    print("np.dot result:    ", np_result)
    print("Match:", np.isclose(my_result, np_result))
    print()

print("=== Manual Matrix Multiplication ===")

def manual_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Cannot multiply {A.shape} x {B.shape}. inner dimensions {A.shape[1]} and {B.shape[0]} not same"
        )

    C = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = manual_dot(A[i, :], B[:, j])

    return C


matrix_tests = [
    (
        np.array([[1., 2., 3.], [4., 5., 6.]]),
        np.array([[7., 8.], [9., 10.], [11., 12.]])
    ),
    (
        np.array([[2., 1.], [0., 3.], [4., 5.]]),
        np.array([[1., 2., 3.], [4., 5., 6.]])
    ),
]

for A, B in matrix_tests:
    my_result = manual_matmul(A, B)
    real_result = A @ B
    print(f"A {A.shape} @ B {B.shape}")
    print("manual_matmul result:")
    print(my_result)
    print("numpy result:")
    print(real_result)
    print("Match:", np.allclose(my_result, real_result))
    print()

print("Testing wrong shape:")
try:
    wrong_A = np.array([[1., 2., 3.], [4., 5., 6.]])
    wrong_B = np.array([[1., 2., 3.], [4., 5., 6.]])
    manual_matmul(wrong_A, wrong_B)
except ValueError as e:
    print("ValueError caught:", e)


print("\n=== Batch Linear Prediction ===")

X = np.array([
    [2, 6, 70],
    [4, 7, 80],
    [6, 8, 90],
    [1, 5, 60],
    [5, 6, 85],
])

w = np.array([5, 2, 0.4])
b = 10

predictions = X @ w + b

print("Student features: [study_hours, sleep_hours, attendance_percent]")
print("Weights:", w)
print("Bias:", b)

for i, sample in enumerate(X):
    print(f"Sample {i + 1}: {sample} -> Predicted score: {predictions[i]:.1f}")


print("\n=== Cosine Similarity: Word Vectors ===")

king = np.array([0.9, 0.3, 0.1])
queen = np.array([0.8, 0.4, 0.2])
apple = np.array([0.1, 0.1, 0.9])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


sim_king_queen = cosine_similarity(king, queen)
sim_king_apple = cosine_similarity(king, apple)
sim_queen_apple = cosine_similarity(queen, apple)

print(f"cos_sim(king, queen) = {sim_king_queen:.4f}")
print(f"cos_sim(king, apple) = {sim_king_apple:.4f}")
print(f"cos_sim(queen, apple) = {sim_queen_apple:.4f}")

