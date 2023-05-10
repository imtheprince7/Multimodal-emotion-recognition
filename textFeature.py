from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

documents=pd.read_csv('dataSet\dataset_label\completeData.csv')['text']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokens = tokenizer.batch_encode_plus(documents, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])

document_features = outputs.last_hidden_state[:, 0, :].numpy()


dataset=pd.DataFrame(document_features)
dataset.to_csv('text_features.csv')
print('text_feature_extraction_done')
