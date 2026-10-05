import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def main():
    data = {
        'Income': [30000, 45000, 55000, 65000, 70000, 42000, 51000, 80000],
        'CreditScore': [560, 610, 650, 700, 720, 590, 640, 760],
        'LoanAmount': [10000, 18000, 22000, 28000, 30000, 15000, 24000, 35000],
        'Approved': [0, 0, 1, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)

    X = df[['Income', 'CreditScore', 'LoanAmount']]
    y = df['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
