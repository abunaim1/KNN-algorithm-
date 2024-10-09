# Load the dataset from the provided link
import csv
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            dataset.append([float(feature) for feature in row[:-1]] + [row[-1]])  # Append features + class label
    return dataset

file_path = "iris.csv" 
dataset = load_dataset(file_path)
print(dataset)


# Randomly Split the dataset into Training (70%), Validation (15%) and Test (15%)
import random

def split_dataset(dataset):
    train_set = []
    val_set = []
    test_set = []
    
    for sample in dataset:
        R = random.random()
        if R <= 0.7:
            train_set.append(sample)
        elif 0.7 < R <= 0.85:
            val_set.append(sample)
        else:
            test_set.append(sample)
    
    return train_set, val_set, test_set

# random.shuffle(dataset) 
train_set, val_set, test_set = split_dataset(dataset)


# Euclidean Distance
import math
def euclidean_distance(sample1, sample2):
    distance = 0
    for i in range(len(sample1) - 1):  # Exclude the class label
        distance += (sample1[i] - sample2[i]) ** 2
    return math.sqrt(distance)



# KNN Classification Algorithm
from collections import Counter
def knn_classify(train_set, val_sample, K):
    distances = []
    for train_sample in train_set:
        dist = euclidean_distance(val_sample, train_sample)
        distances.append((train_sample, dist))
    
    distances.sort(key=lambda x: x[1])  # Sort by distance in ascending order
    
    # Select K nearest neighbors
    K_nearest_neighbors = [sample[0][-1] for sample in distances[:K]]  # Get class labels
    most_common_class = Counter(K_nearest_neighbors).most_common(1)[0][0]  # Get majority class
    return most_common_class



def calculate_accuracy(train_set, val_set, K):
    correct = 0
    for val_sample in val_set:
        predicted_class = knn_classify(train_set, val_sample, K)
        actual_class = val_sample[-1]
        if predicted_class == actual_class:
            correct += 1
    accuracy = (correct / len(val_set)) * 100
    return accuracy

Ks = [1, 3, 5, 10, 15]
validation_accuracies = []

for K in Ks:
    accuracy = calculate_accuracy(train_set, val_set, K)
    validation_accuracies.append((K, accuracy))

# Display validation accuracies
for K, accuracy in validation_accuracies:
    print(f"K: {K}, Validation Accuracy: {accuracy:.2f}%")


# Use the best K from validation accuracy for test set evaluation
best_K = max(validation_accuracies, key=lambda x: x[1])[0]
print("Best K:", best_K)
test_accuracy = calculate_accuracy(train_set, test_set, best_K)

print(f"Best K: {best_K}, Test Accuracy: {test_accuracy:.2f}%")
