import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    data = {
        'Area': [1200, 1500, 1700, 1900, 2200, 2500, 2700],
        'Bedrooms': [2, 3, 3, 3, 4, 4, 4],
        'Age': [20, 15, 12, 8, 6, 4, 3],
        'Price': [250000, 300000, 320000, 350000, 420000, 470000, 520000],
    }
    df = pd.DataFrame(data)

    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Predicted Prices:\n", y_pred)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    main()
