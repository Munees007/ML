import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

def main():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    features = [f'F{i}' for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=features)
    df['target'] = y

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature importance scores:\n", importance_df)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'], color='darkgreen')
    plt.xticks(rotation=45)
    plt.title('Top Features Selected by ExtraTreesClassifier')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
