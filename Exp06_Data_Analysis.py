import pandas as pd
from io import StringIO

def load_dataset():
    csv_data = """
Product,Sales,Profit,Region
A,120,30,North
B,210,60,South
C,160,40,East
D,180,55,West
E,140,25,North
F,250,70,South
G,310,95,East
H,220,65,West
"""
    return pd.read_csv(StringIO(csv_data))

def main():
    df = load_dataset()
    print("Dataset head:\n", df.head())
    print("\nDataset tail:\n", df.tail())
    print("\nDataset description:\n", df.describe())
    print("\nDataset info:")
    df.info()
    print("\nUnique regions:", df['Region'].unique())
    print("\nValue counts:\n", df['Region'].value_counts())
    print("\nCorrelation matrix:\n", df[['Sales', 'Profit']].corr())

if __name__ == "__main__":
    main()
