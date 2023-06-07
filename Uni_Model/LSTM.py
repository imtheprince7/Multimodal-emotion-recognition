import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
data = pd.read_csv('VideoFR/features1.csv')

# Preprocessing
X = data['text'].values
y = data['label'].values
LE=LabelEncoder()
y=LE.fit_transform(y)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Tokenize and pad the input sequences
max_words = 1000  # Maximum number of words to keep in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max(len(sequence) for sequence in X_train_seq)
X_train_pad = tf.keras.utils.pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = tf.keras.utils.pad_sequences(X_test_seq, maxlen=max_sequence_length)

# # Step 5: Convert emotion labels to categorical
# num_classes = len(set(y_train))
# y_train_cat = to_categorical(y_train, num_classes)
# y_test_cat = to_categorical(y_test, num_classes)

# # Step 6: Build the LSTM model
# model = Sequential()
# model.add(Embedding(max_words, 100, input_length=max_sequence_length))
# model.add(LSTM(128))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Step 7: Train the model
# batch_size = 32
# epochs = 10
# model.fit(X_train_pad, y_train_cat, batch_size=batch_size, epochs=epochs, validation_data=(X_test_pad, y_test_cat))
# model.save('Uni_Model/LSTM_MOdel')
# Step 8: Test the model using a sample text input
def Test(text):
    model=load_model('Uni_Model/LSTM_MOdel')
    sample_text = text
    sample_seq = tokenizer.texts_to_sequences([sample_text])
    sample_pad = tf.keras.utils.pad_sequences(sample_seq, maxlen=max_sequence_length)
    predicted_probabilities = model.predict(sample_pad)
    predicted_label = predicted_probabilities.argmax(axis=1)
    return LE.inverse_transform(predicted_label)

# # Step 9: Print the predicted label
# print("Predicted emotion label:", predicted_answer)


# Step 10: Evaluate the model's accuracy on the test set
# _, accuracy = model.evaluate(X_test_pad, y_test_cat)
# print("Test accuracy:", accuracy)
