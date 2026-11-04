
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

from scipy.sparse.csgraph import connected_components
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

iris = load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names

print("Dataset shape:", X.shape)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_variance = sum(
    pca.explained_variance_ratio_
)

print(
    f"PCA explained variance: "
    f"{pca_variance * 100:.4f}%"
)


n_neighbors = 3

isomap = Isomap(
    n_neighbors=n_neighbors,
    n_components=2
)

X_isomap = isomap.fit_transform(X)

print("ISOMAP shape:", X_isomap.shape)

# Check connectivity of the original
# three-nearest-neighbor graph
graph = kneighbors_graph(
    X,
    n_neighbors=3,
    mode="connectivity",
    include_self=False
)

n_graph_components, _ = connected_components(
    graph,
    directed=False
)

print(
    "3-NN graph connected components:",
    n_graph_components
)



# Silhouette score uses species labels only
# for evaluation, not for fitting PCA/ISOMAP.

pca_score = silhouette_score(X_pca, y)
isomap_score = silhouette_score(X_isomap, y)

print(f"PCA silhouette score: {pca_score:.6f}")
print(f"ISOMAP silhouette score: {isomap_score:.6f}")



plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.scatter(
        X_isomap[y == i, 0],
        X_isomap[y == i, 1],
        label=name,
        alpha=0.75
    )

plt.xlabel("ISOMAP Component 1")
plt.ylabel("ISOMAP Component 2")
plt.title("Iris ISOMAP: 3 Nearest Neighbors")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_isomap_3nn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



fig, axes = plt.subplots(
    1, 2, figsize=(14, 6)
)

for i, name in enumerate(class_names):

    axes[0].scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name,
        alpha=0.75
    )

    axes[1].scatter(
        X_isomap[y == i, 0],
        X_isomap[y == i, 1],
        label=name,
        alpha=0.75
    )

axes[0].set_title("PCA (2D)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].set_title("ISOMAP (2D, k=3)")
axes[1].set_xlabel("ISOMAP Component 1")
axes[1].set_ylabel("ISOMAP Component 2")

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_pca_vs_isomap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
