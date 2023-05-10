import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from CSV file
data = pd.read_csv('dataSet/dataset_label/completeData.csv')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize input data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True)

# Convert labels to numerical values
labels = list(train_data['emotion'].unique())
train_data['emotion'] = train_data['emotion'].apply(lambda x: labels.index(x))
test_data['emotion'] = test_data['emotion'].apply(lambda x: labels.index(x))

# Convert data to PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(list(train_data['emotion']))

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(list(test_data['emotion']))

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Define DataLoader instances for efficient loading of data into GPU
batch_size = 8
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Train the model
num_epochs = 3

model = model.to(device)
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    for batch in train_dataloader:
        # Move batch to device (GPU or CPU)
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        # Clear gradients
        optimizer.zero_grad()
        # Compute model outputs and loss
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        # Backpropagate and update model weights
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_dataloader:
        # Move batch to device (GPU or CPU)
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch
        # Compute model outputs
        outputs = model(inputs, attention_mask=masks)
        # Compute predicted labels and accuracy
        _, predicted = torch.max(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Accuracy is:", accuracy)
