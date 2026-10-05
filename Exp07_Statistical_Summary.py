import pandas as pd
from io import StringIO

def load_dataset():
    csv_data = """
Age,Income,Score
22,25000,78
25,30000,85
27,28000,80
30,42000,90
35,52000,92
48,68000,88
52,74000,70
60,85000,76
"""
    return pd.read_csv(StringIO(csv_data))

def main():
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("Column names:", list(df.columns))
    print("Data types:\n", df.dtypes)
    print("\nSummary statistics:\n", df.describe())
    print("\nNull values:\n", df.isnull().sum())
    print("\nMemory usage:\n", df.memory_usage(deep=True))

if __name__ == "__main__":
    main()
