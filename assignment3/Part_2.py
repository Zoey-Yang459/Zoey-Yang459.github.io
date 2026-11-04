
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("Dataset shape:", X.shape)
print("Feature names:", feature_names)
print("Class names:", class_names)



plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.hist(
        X[y == i, 2],
        bins=16,
        alpha=0.55,
        label=name
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Number of Flowers")
plt.title("1D Iris Visualization: Petal Length")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_1d_petal_length.png",
    dpi=300
)
plt.show()



plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.scatter(
        X[y == i, 0],
        X[y == i, 1],
        label=name,
        alpha=0.75
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("2D Iris Visualization: Sepal Features")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_2d_sepal.png",
    dpi=300
)
plt.show()



plt.figure(figsize=(9, 6))

for i, name in enumerate(class_names):
    plt.scatter(
        X[y == i, 2],
        X[y == i, 3],
        label=name,
        alpha=0.75
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("2D Iris Visualization: Petal Features")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_2d_petal.png",
    dpi=300
)
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for i, name in enumerate(class_names):
    ax.scatter(
        X[y == i, 0],
        X[y == i, 2],
        X[y == i, 3],
        label=name,
        alpha=0.75
    )

ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
ax.set_zlabel("Petal Width (cm)")
ax.set_title("3D Iris Visualization")
ax.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "iris_3d.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
