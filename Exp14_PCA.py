import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Transformed data shape:", X_reduced.shape)
    print("Sample transformed data:\n", X_reduced[:5])

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title('PCA - 2 Principal Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
