
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pathlib import Path

# --------------------------------------
# Part 3
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("Dataset shape:", X.shape)



# Iris has 4 original features.
# PCA automatically centers the data.
# We do not standardize feature variances.

pca = PCA(n_components=4)

# Fit PCA and transform the data
X_pca = pca.fit_transform(X)

print("\nPCA transformed shape:", X_pca.shape)



print("\nPrincipal Component Vectors:")

for i, component in enumerate(pca.components_):
    print(f"\nPC{i + 1}:")
    for feature, weight in zip(feature_names, component):
        print(f"  {feature}: {weight:.6f}")



print("\nExplained Variance Ratios:")

for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i + 1}: {ratio * 100:.4f}%")

print("\nCumulative Explained Variance:")

cumulative = np.cumsum(pca.explained_variance_ratio_)

for i, ratio in enumerate(cumulative):
    print(f"First {i + 1} PCs: {ratio * 100:.4f}%")


plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.hist(
        X_pca[y == i, 0],
        bins=18,
        alpha=0.55,
        label=name
    )

plt.xlabel("First Principal Component (PC1)")
plt.ylabel("Number of Flowers")
plt.title("1D Iris PCA: First Principal Component")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_pca_1d.png",
    dpi=300
)

plt.show()


plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name,
        alpha=0.75
    )

plt.xlabel("First Principal Component (PC1)")
plt.ylabel("Second Principal Component (PC2)")
plt.title("2D Iris PCA: First Two Components")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_pca_2d.png",
    dpi=300
)

plt.show()



fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection="3d")

for i, name in enumerate(class_names):
    ax.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        X_pca[y == i, 2],
        label=name,
        alpha=0.75
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

ax.set_title("3D Iris PCA: First Three Components")
ax.legend()

ax.xaxis.labelpad = 8
ax.yaxis.labelpad = 8
ax.zaxis.labelpad = 8
ax.set_box_aspect(None, zoom=0.85)

plt.savefig(
    PLOTS_DIR / "iris_pca_3d.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.4
)


# --------------------------------------
# Part 4: Data Compression
# --------------------------------------

print("\nPart 4: Data Compression")

ratios = pca.explained_variance_ratio_
cumulative = np.cumsum(ratios)

# Question 1: Variance explained by first 2 PCs
variance_2d = cumulative[1]

print(
    f"Variance explained by 2 PCs: "
    f"{variance_2d * 100:.4f}%"
)

# Question 2: Minimum PCs to explain 50%
min_components = (
    np.searchsorted(cumulative, 0.50) + 1
)

print(
    f"Minimum PCs for 50% variance: "
    f"{min_components}"
)

# Verify using the Frobenius norm
X_centered = X - pca.mean_

Z_2d = X_pca[:, :2]

frobenius_ratio = (
    np.linalg.norm(Z_2d, ord="fro") ** 2
    / np.linalg.norm(X_centered, ord="fro") ** 2
)

print(
    f"Frobenius norm verification: "
    f"{frobenius_ratio * 100:.4f}%"
)
