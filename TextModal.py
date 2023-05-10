import pandas as pd
<<<<<<< HEAD
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split


data = pd.read_csv('dataSet\\dataset_label\\completeData.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#Tokenizing input_data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True)

# Converting labels => numerical values
labels = list(train_data['emotion'].unique())
train_data['emotion'] = train_data['emotion'].apply(lambda x: labels.index(x))
test_data['emotion'] = test_data['emotion'].apply(lambda x: labels.index(x))

# Converting data => PyTorch tensors
train_labels = torch.tensor(list(train_data['emotion']))
test_labels = torch.tensor(list(test_data['emotion']))

train_encodings.pop('token_type_ids', None)
test_encodings.pop('token_type_ids', None)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), 
                                               torch.tensor(train_encodings['attention_mask']), 
                                               train_labels)
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']), 
                                              torch.tensor(test_encodings['attention_mask']), 
                                              test_labels)

# Defining Model, Set optimizer and the learning rate
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

batch_size = 8
num_epochs = 3

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Train the model
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print('Test Accuracy:', accuracy)
=======
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
>>>>>>> fa1822ec888780ef2140ea7e16718fd2d9b70d4b
