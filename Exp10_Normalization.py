import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    data = np.array([[100, 10], [200, 20], [300, 30], [400, 40]], dtype=float)
    df = pd.DataFrame(data, columns=['FeatureA', 'FeatureB'])

    print("Original data:\n", df)

    minmax = MinMaxScaler()
    minmax_scaled = minmax.fit_transform(data)
    print("\nMin-Max Normalized:\n", pd.DataFrame(minmax_scaled, columns=df.columns))

    zscore = StandardScaler()
    zscore_scaled = zscore.fit_transform(data)
    print("\nZ-Score Normalized:\n", pd.DataFrame(zscore_scaled, columns=df.columns))

if __name__ == "__main__":
    main()
