import numpy as np
import csv

# Load the .npy file
data = np.load('BiModel\\fusion_embeddings.npy')

# Define the output CSV file path
output_file = 'BiModel\\fusion_embeddings.csv'

# Write the data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
