import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def main():
    df = pd.DataFrame({
        'AnnualIncome': [15, 18, 19, 30, 32, 35, 40, 45, 48, 55, 60, 70],
        'SpendingScore': [39, 41, 42, 60, 62, 67, 72, 75, 77, 80, 85, 90]
    })

    inertia = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(df)
        inertia.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)
    df['Cluster'] = labels

    print("Cluster centers:\n", kmeans.cluster_centers_)
    print("\nClustered data:\n", df)

    plt.figure(figsize=(8, 6))
    plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
    plt.title('Customer Segmentation using K-Means')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
