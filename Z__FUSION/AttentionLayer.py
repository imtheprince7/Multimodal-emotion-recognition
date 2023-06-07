import torch
import torch.nn as nn
import numpy as np

fusion_data = np.load('Fusion.npy')

#NumPy array to a PyTorch tensor
fusion_data_tensor = torch.from_numpy(fusion_data).float()

# Defining self-attention module
class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        query = self.query(x.float())
        key = self.key(x.float())
        value = self.value(x.float())

        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        return attended_values

# Instantiate self-attention module
input_size = fusion_data_tensor.shape[-1]
self_attention = SelfAttention(input_size)

# Applying self-attention to fusion data
attended_data = self_attention(fusion_data_tensor)

# Convert attended data back to a NumPy array
attended_data_np = attended_data.detach().numpy()

# Save the attended data as a new .npy file
np.save('attended_data.npy', attended_data_np)
