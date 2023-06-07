import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,MaxAbsScaler,StandardScaler

# Load data from CSV file
data = pd.read_csv('BiModel\\fusion_embeddings.csv')

# Separate features and labels
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
LE=LabelEncoder()
labels=LE.fit_transform(labels)

features=MaxAbsScaler().fit_transform(features)
features=StandardScaler().fit_transform(features)
features=StandardScaler().fit_transform(features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)

# Convert data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
model.save('BiModel/FullyConnected.h5')


print('Test accuracy:', test_acc)
