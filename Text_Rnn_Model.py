import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load the CSV file
data = pd.read_csv('dataset_label\\completeData.csv')

# Preprocessing
X = data['text'].values
y = data['emotion'].values

# Tokenize the input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure consistent length
max_seq_length = 100  # Specify the desired sequence length
X = tf.keras.utils.pad_sequences(X, maxlen=max_seq_length)

# Encode the output labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y = to_categorical(y, num_classes=num_classes)
vocab_size = len(tokenizer.word_index) + 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_seq_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('rnn_model.h5')

# Make predictions on new data
# new_data = ['your new input text']
# new_data = tokenizer.texts_to_sequences(new_data)
# new_data = pad_sequences(new_data, maxlen=max_seq_length)
# predictions = model.predict(new_data)
# predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))