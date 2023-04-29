import torch
from transformers import BertTokenizer, BertModel


# Load pre-trained BERT model and tokenizer
def bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Define input text
    # text = "E:\DataSet\Session-1\Transcription\Ses01F_impro01.txt"

    # Tokenize input text and add special tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Convert tokens to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.ones_like(input_ids)

    # Extract features using BERT model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get the last hidden state of the tokenized input sequence
    last_hidden_state = outputs.last_hidden_state

    # Average the last hidden state across all tokens to get a sentence-level representation
    sentence_embedding = torch.mean(last_hidden_state, dim=1)

    # Print the sentence-level representation as a NumPy array
    return sentence_embedding.numpy()
