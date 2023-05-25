import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Defined a custom_dataset_Class to load data from CSV file
class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder()
        self.input_texts = self.data['text'].tolist()
        self.labels = self.label_encoder.fit_transform(self.data['emotion'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = self.input_texts[index]
        label = self.labels[index]

# Tokenizing  the input text
        input_tokens = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = input_tokens['input_ids'].squeeze()
        attention_mask = input_tokens['attention_mask'].squeeze()

        return input_ids, attention_mask, label

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

# Set the device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


csv_file = 'dataset_1000.csv'
model_file = 'BertModel.pt'

# Create an instance of the custom dataset
dataset = CustomDataset(csv_file, tokenizer)

# Define data loader for batching and shuffling the dataset
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch in data_loader:
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('Program Rrunning.........')

    epoch_loss = running_loss / len(data_loader)
#    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')

# Save trained model
torch.save(model.state_dict(), model_file)
