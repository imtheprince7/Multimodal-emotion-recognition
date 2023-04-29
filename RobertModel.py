import torch
from transformers import RobertaModel, RobertaTokenizer

# Load pre-trained RoBERTa model and tokenizer
model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define input text
input_text = "E:\DataSet\Session-1\Transcription\Ses01F_impro01.txt"

# Tokenize input text
tokenized_text = tokenizer.encode(input_text, add_special_tokens=True)

# Convert tokenized input to PyTorch tensors
input_ids = torch.tensor([tokenized_text])

# Obtain RoBERTa embeddings
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state

# Extract features
features = embeddings.squeeze(0).mean(0).numpy()
