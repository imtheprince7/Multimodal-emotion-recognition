import numpy as np

# Assume you already have features extracted for Text and Audio separately
text_features = [...] # shape (num_text_samples, text_feature_dim)
audio_features = [...] # shape (num_audio_samples, audio_feature_dim)

# Define the sequence lengths for each modality
text_seq_len = [...] # shape (num_text_samples,)
audio_seq_len = [...] # shape (num_audio_samples,)

# Define the maximum sequence length for each modality
max_text_seq_len = max(text_seq_len)
max_audio_seq_len = max(audio_seq_len)

# Pad the sequences to their maximum length with zeros
text_features_padded = np.zeros((len(text_features), max_text_seq_len, text_feature_dim))
audio_features_padded = np.zeros((len(audio_features), max_audio_seq_len, audio_feature_dim))

for i in range(len(text_features)):
    text_features_padded[i, :text_seq_len[i], :] = text_features[i]

for i in range(len(audio_features)):
    audio_features_padded[i, :audio_seq_len[i], :] = audio_features[i]

# Concatenate the padded sequences along the time dimension
features_concatenated = np.concatenate([text_features_padded, audio_features_padded], axis=1)

# Perform sequence aggregation (e.g. mean pooling)
features_aggregated = np.mean(features_concatenated, axis=1)
