import numpy as np

# Assuming we have extracted features from text and audio
text_features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
audio_features = np.array([[1, 2], [3, 4], [5, 6]])

# Concatenate the features along the axis of time
concatenated_features = np.concatenate((text_features, audio_features), axis=1)

# Calculate the mean along the axis of time
aggregated_features = np.mean(concatenated_features, axis=0)

# Print the aggregated features
print(aggregated_features)
