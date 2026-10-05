import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs('plots', exist_ok=True)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    sales = [120, 150, 170, 160, 210, 240]
    df = pd.DataFrame({'Month': months, 'Sales': sales})

    plt.figure(figsize=(8, 5))
    plt.plot(df['Month'], df['Sales'], marker='o')
    plt.title('Line Chart')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('plots/line_chart.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(df['Month'], df['Sales'], color='steelblue')
    plt.title('Bar Chart')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('plots/bar_chart.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df['Sales'], bins=5, color='tomato', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/histogram.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.pie(df['Sales'], labels=df['Month'], autopct='%1.1f%%')
    plt.title('Pie Chart')
    plt.tight_layout()
    plt.savefig('plots/pie_chart.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df['Month'], df['Sales'])
    plt.title('Scatter Plot')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('plots/scatter_plot.png')
    plt.close()

    print('Saved visualizations to plots/')

if __name__ == "__main__":
    main()
