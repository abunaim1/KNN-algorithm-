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