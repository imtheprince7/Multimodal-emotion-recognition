import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load the CSV file
data = pd.read_csv('VideoFR/features1.csv')

# Preprocessing
X = data['text'].values
y = data['label'].values

# # Tokenize the input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# # Pad sequences to ensure consistent length
max_seq_length = 100  # Specify the desired sequence length
X = tf.keras.utils.pad_sequences(X, maxlen=max_seq_length)

# # Encode the output labels
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
model.add(Bidirectional(LSTM(units=128)))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save('Uni_Model/bilstm_model.h5')
test='Hi how are you'
X = tf.keras.utils.pad_sequences(test, maxlen=max_seq_length)
print(model.predict(X))



# def Test(text):
#     model=load_model('Uni_Model/bilstm_model.h5')
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(text)
#     text = tokenizer.texts_to_sequences(text)
#     X=tf.keras.utils.pad_sequences(text, maxlen=100)
#     return LabelEncoder().inverse_transform(model.predict(X))