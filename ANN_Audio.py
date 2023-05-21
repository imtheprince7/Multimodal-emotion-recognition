import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv('data.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1] 

# LabelEncoder encode_output_Feature
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# StandardScaler__standardize_input_features
scaler = RobustScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating_  ANN_model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training_Model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy:', test_acc)