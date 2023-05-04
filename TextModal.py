import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('data.csv')

# Spliting into input & output 
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizing input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encoding output labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

# model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(9, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define K-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate the model with K-fold cross-validation
scores = []
for train, test in kfold.split(X, np.argmax(y, axis=1)):
    model.fit(X[train], y[train], epochs=50, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(X[test], y[test], verbose=0)
    print(accuracy)
    scores.append(accuracy) 
print("Accuracy: {:.2f}% (+/- {:.2f}%)".format(np.mean(scores)*100, np.std(scores)*100))
