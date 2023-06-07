import pandas as pd
import numpy as np
import tensorflow as tf
import csv

# Define the Attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u = tf.matmul(inputs, self.W) + self.b
        a = tf.nn.softmax(u, axis=1)
        output = inputs * a
        return output

# Read the data from CSV file
data = pd.read_csv('fusion.csv')


# Extract the column values as a numpy array
column_data = data.iloc[0:, 0:256].values # Assuming the columns you want to process start from the second column
print(column_data)
# Convert the column data to a tensor
column_tensor = tf.convert_to_tensor(column_data, dtype=tf.float32)

# Define the attention model
attention_model = tf.keras.Sequential([
    AttentionLayer(),
    tf.keras.layers.Dense(128, activation='relu'),
])

# Pass the column tensor through the attention model
attended_column_tensor = attention_model(column_tensor)

# Convert the attended column tensor back to a numpy array
attended_column_data = attended_column_tensor.numpy()

# Define the output CSV file path
output_file = 'output.csv'

# Write the data to the CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(attended_column_data)


# best accuracy Epoch 20, batch_size = 16,64  ==> Acc: 0.2155    TestAccu = 0.2048




import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data from CSV file
data = pd.read_csv('Z_Fusion_Trans/fusion.csv')

# Separate features and labels
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
LE=LabelEncoder()
labels=LE.fit_transform(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)

# Convert data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# # Normalize feature data
# X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
# X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)




# Print classification_report

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)
# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
