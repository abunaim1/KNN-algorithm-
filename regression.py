import csv
import random
import math

# Load the dataset
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            dataset.append([float(feature) for feature in row])
    return dataset

file_path = "diabetes.csv"  # Replace with your dataset path
dataset = load_dataset(file_path)

# Shuffle and split the dataset
def split_dataset(dataset):
    random.shuffle(dataset)
    train_set, val_set, test_set = [], [], []
    
    for sample in dataset:
        R = random.random()
        if R <= 0.7:
            train_set.append(sample)
        elif 0.7 < R <= 0.85:
            val_set.append(sample)
        else:
            test_set.append(sample)
    
    return train_set, val_set, test_set

train_set, val_set, test_set = split_dataset(dataset)

# Euclidean distance calculation
def euclidean_distance(sample1, sample2):
    distance = 0
    for i in range(len(sample1) - 1):  # Exclude the output value
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)

# KNN Regression Algorithm
def knn_regression(train_set, val_sample, K):
    distances = []
    for train_sample in train_set:
        dist = euclidean_distance(val_sample, train_sample)
        distances.append((train_sample, dist))
    
    distances.sort(key=lambda x: x[1])  # Sort by distance in ascending order
    
    # Take the first K samples and calculate the average output
    K_nearest_outputs = [sample[0][-1] for sample in distances[:K]]  # Get outputs (regression targets)
    
    predicted_output = sum(K_nearest_outputs) / K  # Average of K nearest outputs
    return predicted_output

# Calculate Mean Squared Error (MSE)
def calculate_mse(train_set, val_set, K):
    error = 0
    for val_sample in val_set:
        predicted_output = knn_regression(train_set, val_sample, K)
        actual_output = val_sample[-1]
        error += (actual_output - predicted_output) ** 2
    
    mse = error / len(val_set)
    return mse

# Testing different K values
Ks = [1, 3, 5, 10, 15]
mse_results = []

for K in Ks:
    mse = calculate_mse(train_set, val_set, K)
    mse_results.append((K, mse))

# Display MSE results for different K values
for K, mse in mse_results:
    print(f"K: {K}, Mean Squared Error: {mse:.4f}")

# Find the best K (minimum MSE)
best_K = min(mse_results, key=lambda x: x[1])[0]
print(f"Best K: {best_K}")

# Use the best K for the test set
test_mse = calculate_mse(train_set, test_set, best_K)
print(f"Test Set Mean Squared Error with K={best_K}: {test_mse:.4f}")
