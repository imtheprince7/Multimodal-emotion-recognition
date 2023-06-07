import numpy as np
import csv

# Load the .npy file
data = np.load('attention_matrix.npy')

# Define the output CSV file path
output_file = 'attention_matrix.csv'

# Write the data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
