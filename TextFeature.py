from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Define the text documents
# documents = ['This is the first document', 'This is the second document', 'And this is the third document']
documents=pd.read_csv('dataset_label\\1st.csv')['text']
# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Tokenize the documents and add special tokens
tokens = tokenizer.batch_encode_plus(documents, padding=True, truncation=True, return_tensors='pt')

# Extract the features from the BERT model
with torch.no_grad():
    outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])

# Extract the last hidden state of the BERT model as the document features
document_features = outputs.last_hidden_state[:, 0, :].numpy()

# Print the document features
dataset=pd.DataFrame(document_features)
dataset.to_csv('text_features.csv')
print('text_feature_extraction_done')
