"""SYDE 572 Assignment 2 — Part 2: Gaussian basis regression.

Reproducible implementation of Questions 2(e)–2(f).
Gaussian widths are median positive pairwise distances in normalized training data.
Requires numpy, matplotlib, and housing.data in the same folder.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

data = np.loadtxt(BASE_DIR / "housing.data")
X, y = data[:, :13], data[:, 13]
X_train, y_train = X[:100], y[:100]
X_test, y_test = X[100:], y[100:]

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1.0
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_basis(X, centers, sigma):
    """Bias + K radial Gaussian features."""
    difference = X[:, None, :] - centers[None, :, :]
    distance_squared = np.sum(difference ** 2, axis=2)
    gaussian_values = np.exp(-distance_squared / (2 * sigma ** 2))
    return np.column_stack([np.ones(len(X)), gaussian_values])


def median_width(X_fit):
    """Median nonzero distance among distinct fitting samples."""
    difference = X_fit[:, None, :] - X_fit[None, :, :]
    distances = np.sqrt(np.sum(difference ** 2, axis=2))
    upper = distances[np.triu_indices(len(X_fit), k=1)]
    positive_distances = upper[upper > 0]
    return float(np.median(positive_distances))


def gaussian_regression(X_fit, y_fit, X_eval, y_eval, centers, sigma):
    """Unregularized Gaussian least squares."""
    Phi_fit = gaussian_basis(X_fit, centers, sigma)
    Phi_eval = gaussian_basis(X_eval, centers, sigma)
    weights, _, _, _ = np.linalg.lstsq(Phi_fit, y_fit, rcond=None)
    return (
        weights,
        rmse(y_fit, Phi_fit @ weights),
        rmse(y_eval, Phi_eval @ weights),
    )


# Question 2(e): K in {5, 15, ..., 95}, no regularization.
sigma = median_width(X_train_norm)
print("Question 2(e): Gaussian Regression")
print(f"Gaussian width (sigma): {sigma:.6f}")

rng = np.random.default_rng(42)
permutation = rng.permutation(len(X_train_norm))
basis_counts = list(range(5, 100, 10))
train_errors, test_errors = [], []
for K in basis_counts:
    centers = X_train_norm[permutation[:K]]
    weights, train_error, test_error = gaussian_regression(
        X_train_norm, y_train, X_test_norm, y_test, centers, sigma
    )
    train_errors.append(train_error)
    test_errors.append(test_error)
    print(f"K = {K:2d}: Training RMSE = {train_error:.6f}, "
          f"Testing RMSE = {test_error:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(basis_counts, train_errors, marker="o", linewidth=2,
         label="Training RMSE")
plt.plot(basis_counts, test_errors, marker="s", linewidth=2,
         label="Testing RMSE")
plt.xticks(basis_counts)
plt.xlabel("Number of Gaussian Basis Functions")
plt.ylabel("RMSE")
plt.title("Gaussian Regression: Training vs Testing Error")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "gaussian_rmse_vs_basis.png", dpi=190,
            bbox_inches="tight")
plt.close()


# Question 2(f): 90 basis functions, ridge regression, 10-fold CV.
def gaussian_regression_reg(X_fit, y_fit, centers, sigma, lam):
    """Ridge Gaussian least-squares; intercept excluded from the penalty."""
    Phi = gaussian_basis(X_fit, centers, sigma)
    if lam == 0:
        return np.linalg.lstsq(Phi, y_fit, rcond=None)[0]
    penalty = np.eye(Phi.shape[1])
    penalty[0, 0] = 0.0
    A = np.vstack([Phi, np.sqrt(lam) * penalty])
    b = np.concatenate([y_fit, np.zeros(Phi.shape[1])])
    return np.linalg.lstsq(A, b, rcond=None)[0]

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
K = 90
rng = np.random.default_rng(42)
fold_indices = rng.permutation(len(X_train))
folds = np.array_split(fold_indices, 10)
validation_errors = {lam: [] for lam in lambdas}

print("\nQuestion 2(f): Gaussian Regularization")
for fold_index in range(10):
    val_indices = folds[fold_index]
    fit_indices = np.concatenate([
        folds[j] for j in range(10) if j != fold_index
    ])
    X_fit, X_val = X_train[fit_indices], X_train[val_indices]
    y_fit, y_val = y_train[fit_indices], y_train[val_indices]

    # Prevent validation leakage: normalize using fitting observations.
    fold_mean = X_fit.mean(axis=0)
    fold_std = X_fit.std(axis=0)
    fold_std[fold_std == 0] = 1.0
    X_fit_norm = (X_fit - fold_mean) / fold_std
    X_val_norm = (X_val - fold_mean) / fold_std
    sigma_fold = median_width(X_fit_norm)

    # 90 training points per fold; all 90 are eligible centers.
    # Keep the same randomly ordered centers for every lambda in a fold.
    center_indices = rng.permutation(len(X_fit_norm))[:K]
    centers = X_fit_norm[center_indices]
    Phi_val = gaussian_basis(X_val_norm, centers, sigma_fold)

    for lam in lambdas:
        weights = gaussian_regression_reg(
            X_fit_norm, y_fit, centers, sigma_fold, lam
        )
        validation_errors[lam].append(rmse(y_val, Phi_val @ weights))

average_errors = []
for lam in lambdas:
    avg_error = float(np.mean(validation_errors[lam]))
    average_errors.append(avg_error)
    print(f"Lambda = {lam:7g}, Average Validation RMSE = {avg_error:.6f}")
best_lambda = lambdas[int(np.argmin(average_errors))]
print(f"\nBest lambda: {best_lambda}")

plt.figure(figsize=(10, 6))
plt.semilogx(lambdas[1:], average_errors[1:], marker="o", linewidth=2,
             label="Average Validation RMSE")
plt.axhline(average_errors[0], linestyle="--",
            label=f"Unregularized (lambda = 0): {average_errors[0]:.3f}")
plt.xlabel("Regularization Parameter (lambda)")
plt.ylabel("Average Validation RMSE")
plt.title("Gaussian Regression: 10-Fold Cross-Validation")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "gaussian_regularization_cv.png", dpi=190,
            bbox_inches="tight")
plt.close()
