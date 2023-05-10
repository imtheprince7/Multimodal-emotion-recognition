import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

# Load the dataset
df = pd.read_csv('dataSet\\dataset_label\\completeData.csv')
X=df['text']
Y=df['emotion']
Y=LabelEncoder().fit_transform(Y)
# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(X
    , Y, test_size=0.2, random_state=42)

# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the texts and encode the labels as integers
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# Define the RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Define the training and validation datasets
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels)
val_dataset = TextClassificationDataset(val_encodings, val_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=50,
    learning_rate=1e-5,
    save_total_limit=1,
    save_strategy='best',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                'attention_mask': torch.stack([item['attention_mask'] for item in data]),
                                'labels': torch.stack([item['labels'] for item in data])}
)

# Train the model
trainer.train()

# Evaluate the model on the validation set
eval_results = trainer.evaluate()

# Print the accuracy
print("Validation accuracy:", eval_results['eval_accuracy'])
