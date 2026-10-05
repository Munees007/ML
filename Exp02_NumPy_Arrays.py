import numpy as np

def main():
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2], [3, 4]])
    arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print("1D array:", arr1)
    print("2D array:\n", arr2)
    print("3D array:\n", arr3)

    print("Indexing example:", arr2[1, 0])
    print("Slicing example:", arr1[1:4])
    print("Row slice example:\n", arr3[:, :, 1])

    print("Shape:", arr2.shape)
    print("Size:", arr2.size)
    print("Dimensions:", arr2.ndim)
    print("Datatype:", arr2.dtype)

    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    print("Matrix addition:\n", np.add(matrix_a, matrix_b))
    print("Matrix multiplication:\n", matrix_a @ matrix_b)

if __name__ == "__main__":
    main()
