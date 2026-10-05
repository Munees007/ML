import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = {
        'Math': [78, 82, 85, 90, 88, 94],
        'Science': [70, 76, 80, 92, 86, 96],
        'English': [82, 84, 88, 85, 91, 89],
        'History': [68, 70, 72, 78, 74, 88]
    }
    df = pd.DataFrame(data)

    cov_matrix = df.cov()
    corr_matrix = df.corr()

    print("Covariance Matrix:\n", cov_matrix)
    print("\nCorrelation Matrix:\n", corr_matrix)
    print("\nInterpretation: values closer to 1 or -1 indicate stronger linear relationships.")

    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
