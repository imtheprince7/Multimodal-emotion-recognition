import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Attention,concatenate

data = pd.read_csv('AG.csv')

data = data.iloc[:, 1:]  

X = data.iloc[:, :-1].values  
X=MinMaxScaler().fit_transform(X)
y = data.iloc[:, -1].values 
y=LabelEncoder().fit_transform(y)

# Step 2: Preprocess the data
# converting categorical variables into numerical representations, && scaling numeric features, or encoding labels.


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#  Building the CNN1-D model
input_shape = (X_train.shape[1],) 
num_classes = len(np.unique(y_train))  # Number of target classes

input_layer = Input(shape=input_shape)
dense_query = Dense(64)(input_layer)  # Generate query tensor
dense_value = Dense(64)(input_layer)  # Generate value tensor
attention_layer = Attention()([dense_query, dense_value])
output_layer = Dense(num_classes, activation='softmax')(attention_layer)

model = Model(inputs=input_layer, outputs=output_layer)



# Step 4: Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Perform any necessary post-processing on the predictions
model.save('attentionModel.h5')


# Evaluate the model's performance using appropriate metrics
evaluation = model.evaluate(X_test, y_test)
print(f"Accuracy: {evaluation[1]:.4f}")

