import pandas as pd
import numpy as np

def main():
    data = {
        'Age': [25, 30, np.nan, 35, 40],
        'Salary': [30000, np.nan, 42000, 50000, 55000],
        'Experience': [1, 3, 5, np.nan, 7],
        'Department': ['HR', 'IT', 'Marketing', 'IT', None]
    }

    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)
    print("\nMissing value counts:\n", df.isnull().sum())

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Salary'] = df['Salary'].fillna(df['Salary'].median())
    df['Experience'] = df['Experience'].fillna(df['Experience'].mode()[0])
    df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
    print("\nAfter filling missing values:\n", df)

    df_dropped = df.dropna()
    print("\nAfter removing rows with missing values:\n", df_dropped)

if __name__ == "__main__":
    main()
