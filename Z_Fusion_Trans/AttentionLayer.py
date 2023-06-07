import numpy as np
import pandas as pd


data = pd.read_csv('fusion.csv')
column_data = data['data']

#Converting column data into a 2D NumPy array
list_data = np.array([eval(row) for row in column_data])

#Initializing attention matrix
num_rows = list_data.shape[0]
attention_matrix = np.zeros((num_rows, 128))

#Fill attention matrix
for i in range(num_rows):
    attention_matrix[i] = np.sum(list_data[i], axis=0)

# Save attention matrix in .npy file
np.save('attention_matrix.npy', attention_matrix)

print(attention_matrix.shape)
