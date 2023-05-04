import pandas as pd

# Load the audio and text feature CSV files
audio_features = pd.read_csv("audio_features.csv")
text_features = pd.read_csv("text_features.csv")

# Create sequence columns for both dataframes
audio_features["sequence"] = range(len(audio_features))
text_features["sequence"] = range(len(text_features))

# Merge the two dataframes using the sequence column
merged_features = pd.merge(audio_features, text_features, on="sequence")

# Save the merged dataframe to a CSV file
merged_features.to_csv("sequenceAgg.csv", index=False)
