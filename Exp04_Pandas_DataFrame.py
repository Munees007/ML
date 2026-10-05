import pandas as pd

def main():
    student_data = {
        'StudentID': [101, 102, 103, 104],
        'Name': ['Aisha', 'Daniel', 'Sofia', 'Lucas'],
        'Marks': [88, 76, 92, 81],
        'Department': ['CSE', 'ECE', 'CSE', 'ME']
    }

    df = pd.DataFrame(student_data)
    print("DataFrame:\n", df)
    print("\nRows and columns:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nSelected columns:\n", df[['Name', 'Marks']])

    df['Grade'] = ['A', 'B', 'A+', 'B+']
    print("\nAfter adding Grade column:\n", df)

    df = df.drop(columns=['Department'])
    print("\nAfter deleting Department column:\n", df)

    filtered = df[df['Marks'] >= 80]
    print("\nFiltered records:\n", filtered)

if __name__ == "__main__":
    main()
