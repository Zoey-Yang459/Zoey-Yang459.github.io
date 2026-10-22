"""SYDE 572 Assignment 2 — Part 1: polynomial basis regression.

Reproducible implementation of Questions 1(a)–1(d).
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

# Normalize using the first 100 training observations only.
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1.0
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

print("Dataset shape:", data.shape)
print("Training shape:", X_train_norm.shape)
print("Testing shape:", X_test_norm.shape)


def rmse(y_true, y_pred):
    """Root mean squared prediction error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def polynomial_features(X, degree):
    """Bias + univariate monomials, with no interaction/cross-terms."""
    columns = [np.ones(X.shape[0])]
    for power in range(1, degree + 1):
        for j in range(X.shape[1]):
            columns.append(X[:, j] ** power)
    return np.column_stack(columns)


def polynomial_regression(X_fit, y_fit, X_eval, y_eval, degree):
    """Unregularized least-squares polynomial regression."""
    Phi_fit = polynomial_features(X_fit, degree)
    Phi_eval = polynomial_features(X_eval, degree)
    weights, _, _, _ = np.linalg.lstsq(Phi_fit, y_fit, rcond=None)
    return (
        weights,
        rmse(y_fit, Phi_fit @ weights),
        rmse(y_eval, Phi_eval @ weights),
    )


# Question 1(a): degrees 1 through 7, all 13 normalized features.
degrees = list(range(1, 8))
poly_train_errors, poly_test_errors = [], []
print("\nQuestion 1(a): Polynomial Regression")
for degree in degrees:
    weights, train_error, test_error = polynomial_regression(
        X_train_norm, y_train, X_test_norm, y_test, degree
    )
    poly_train_errors.append(train_error)
    poly_test_errors.append(test_error)
    print(f"Degree {degree}: Train RMSE = {train_error:.6f}, "
          f"Test RMSE = {test_error:.6e}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, poly_train_errors, marker="o", linewidth=2, label="Training RMSE")
plt.plot(degrees, poly_test_errors, marker="s", linewidth=2, label="Testing RMSE")
plt.yscale("log")
plt.xticks(degrees)
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE (log scale)")
plt.title("Polynomial Regression: Training vs Testing Error")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "polynomial_rmse_vs_degree.png", dpi=190, bbox_inches="tight")
plt.close()

# Question 1(b): degree-1 weights, including bias.
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
weights, _, _ = polynomial_regression(
    X_train_norm, y_train, X_test_norm, y_test, degree=1
)
print("\nQuestion 1(b): Degree-1 Weights")
print(f"Bias (w0): {weights[0]:.6f}")
for i, name in enumerate(feature_names):
    print(f"{name:8s}: {weights[i + 1]:.6f}")
print(f"\nRM coefficient:  {weights[6]:.6f}")
print(f"DIS coefficient: {weights[8]:.6f}")


# Question 1(c): plot the normalized second feature (ZN) alone.
def polynomial_regression_1d_vis(degree, feature_index=1):
    X_fit = X_train_norm[:, feature_index:feature_index + 1]
    X_eval = X_test_norm[:, feature_index:feature_index + 1]
    weights, train_error, test_error = polynomial_regression(
        X_fit, y_train, X_eval, y_test, degree
    )
    x_min = min(X_fit.min(), X_eval.min())
    x_max = max(X_fit.max(), X_eval.max())
    grid = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    fitted_curve = polynomial_features(grid, degree) @ weights

    plt.figure(figsize=(10, 6))
    plt.scatter(X_fit[:, 0], y_train, alpha=0.65, s=38, label="Training Data")
    plt.scatter(X_eval[:, 0], y_test, alpha=0.35, s=27, label="Testing Data")
    plt.plot(grid[:, 0], fitted_curve, linewidth=2.7,
             label=f"Degree {degree} Polynomial")
    plt.xlabel("Normalized ZN (Second Feature)")
    plt.ylabel("Median House Value ($1000s)")
    plt.title(f"1D Polynomial Regression — Degree {degree}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"polynomial_1d_degree_{degree}.png",
                dpi=190, bbox_inches="tight")
    plt.close()
    return train_error, test_error

print("\nQuestion 1(c): 1D Polynomial Regression")
for degree in [1, 3, 7]:
    train_error, test_error = polynomial_regression_1d_vis(degree)
    print(f"Degree {degree}: Training RMSE = {train_error:.6f}, "
          f"Testing RMSE = {test_error:.6f}")


# Question 1(d): degree-8 ridge regression and 10-fold CV on ZN only.
def polynomial_regression_reg(X_fit, y_fit, degree, lam):
    """Ridge fit via augmented least squares; intercept is not penalized."""
    Phi = polynomial_features(X_fit, degree)
    if lam == 0:
        return np.linalg.lstsq(Phi, y_fit, rcond=None)[0]
    penalty = np.eye(Phi.shape[1])
    penalty[0, 0] = 0.0
    A = np.vstack([Phi, np.sqrt(lam) * penalty])
    b = np.concatenate([y_fit, np.zeros(Phi.shape[1])])
    return np.linalg.lstsq(A, b, rcond=None)[0]

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
X_1d_raw = X_train[:, 1]
rng = np.random.default_rng(42)
folds = np.array_split(rng.permutation(len(y_train)), 10)
average_errors = []
print("\nQuestion 1(d): 10-Fold Cross-Validation")
for lam in lambdas:
    fold_errors = []
    for val_indices in folds:
        fit_indices = np.setdiff1d(np.arange(len(y_train)), val_indices)
        x_fit, x_val = X_1d_raw[fit_indices], X_1d_raw[val_indices]
        y_fit, y_val = y_train[fit_indices], y_train[val_indices]

        # Fit normalization parameters on each fold's fitting data only.
        fold_mean, fold_std = x_fit.mean(), x_fit.std()
        if fold_std == 0:
            fold_std = 1.0
        x_fit = ((x_fit - fold_mean) / fold_std).reshape(-1, 1)
        x_val = ((x_val - fold_mean) / fold_std).reshape(-1, 1)

        ridge_weights = polynomial_regression_reg(x_fit, y_fit, 8, lam)
        val_prediction = polynomial_features(x_val, 8) @ ridge_weights
        fold_errors.append(rmse(y_val, val_prediction))
    average_errors.append(float(np.mean(fold_errors)))
    print(f"Lambda = {lam:7g}, "
          f"Average Validation RMSE = {average_errors[-1]:.6f}")

best_lambda = lambdas[int(np.argmin(average_errors))]
print(f"\nBest lambda: {best_lambda}")
plt.figure(figsize=(10, 6))
plt.semilogx(lambdas[1:], average_errors[1:], marker="o", linewidth=2,
             label="Average Validation RMSE")
plt.axhline(average_errors[0], linestyle="--",
            label=f"Unregularized (lambda = 0): {average_errors[0]:.3f}")
plt.xlabel("Regularization Parameter (lambda)")
plt.ylabel("Average Validation RMSE")
plt.title("Degree-8 Polynomial Regression: 10-Fold CV")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "polynomial_regularization_cv.png",
            dpi=190, bbox_inches="tight")
plt.close()
