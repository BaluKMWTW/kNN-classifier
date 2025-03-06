import pandas as pd
import numpy as np

# Load the data from the Excel file
data = pd.read_excel(r"C:\Users\Jan\Downloads\Q1-Data(1).xlsx")  # Added 'r' before the string to denote a raw string

# Extract X and Y values from the DataFrame
x_values = data['X'].values
y_values = data['Y'].values

# Query point
query_point = np.array([4.9, 6.2])

# Calculate Euclidean distances
distances = np.sqrt((x_values - query_point[0])**2 + (y_values - query_point[1])**2)

# a) For the 1-nearest neighbor classifier:
closest_index = np.argmin(distances)
closest_class_a = "Class A" if closest_index < len(x_values) / 2 else "Class B"
print("a) Decision according to 1-nearest neighbor classifier:", closest_class_a)

# b) For the 3-nearest neighbor classifier:
closest_indices_3 = np.argsort(distances)[:3]
closest_classes_3 = ["Class A" if i < len(x_values) / 2 else "Class B" for i in closest_indices_3]
majority_class_3 = max(set(closest_classes_3), key=closest_classes_3.count)
print("b) Decision according to 3-nearest neighbor classifier:", majority_class_3)

# c) For the k=5-nearest neighbor classifier:
closest_indices_k5 = np.argsort(distances)[:5]
closest_classes_k5 = ["Class A" if i < len(x_values) / 2 else "Class B" for i in closest_indices_k5]
majority_class_k5 = max(set(closest_classes_k5), key=closest_classes_k5.count)
print("c) Decision according to k=5-nearest neighbor classifier:", majority_class_k5)
