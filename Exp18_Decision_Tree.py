import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def main():
    data = {
        'Math': [85, 72, 90, 68, 78, 96, 60, 82],
        'Science': [80, 70, 88, 65, 74, 92, 58, 79],
        'Interest': ['Programming', 'Design', 'Engineering', 'Research', 'Business', 'Programming', 'Research', 'Design'],
        'GPA': [3.8, 3.2, 4.0, 2.8, 3.1, 4.2, 2.7, 3.4],
        'Elective': ['AI', 'UX', 'Robotics', 'Data Science', 'Marketing', 'AI', 'BioTech', 'UX']
    }
    df = pd.DataFrame(data)
    X = df[['Math', 'Science', 'GPA']]
    y = df['Elective']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Prediction results:", y_pred)
    print("Accuracy:", model.score(X_test, y_test))

    plt.figure(figsize=(14, 8))
    plot_tree(model, filled=True, feature_names=['Math', 'Science', 'GPA'])
    plt.show()

if __name__ == "__main__":
    main()
