import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    data = {
        'Age': [22, 28, 35, 45, 50, 30, 41, 27, 60, 39],
        'MonthlyCharges': [50, 70, 90, 120, 150, 100, 130, 75, 180, 110],
        'ContractMonths': [12, 24, 6, 12, 24, 18, 12, 6, 24, 12],
        'CustomerServiceCalls': [1, 2, 3, 4, 5, 2, 3, 1, 6, 2],
        'Churn': [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nFeature Importance:\n", pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

if __name__ == "__main__":
    main()
