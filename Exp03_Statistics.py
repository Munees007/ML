from statistics import mean, median, multimode, variance, stdev, harmonic_mean, geometric_mean

def main():
    data = [12, 16, 15, 14, 15, 18, 17, 19, 15, 20]

    print("Dataset:", data)
    print("Mean:", mean(data))
    print("Median:", median(data))
    print("Mode:", multimode(data))
    print("Variance:", variance(data))
    print("Standard Deviation:", stdev(data))
    print("Harmonic Mean:", harmonic_mean(data))
    print("Geometric Mean:", geometric_mean(data))

if __name__ == "__main__":
    main()
