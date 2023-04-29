import pandas as pd

# Read in the aggregated data
df = pd.read_csv('E:\Multimodal-emotion-recognition\csv_file')

# Convert audio feature columns to float32 to save space
audio_cols = [col for col in df.columns if col.startswith('audio_')]
df[audio_cols] = df[audio_cols].astype('float32')

# Convert text feature columns to category type to save space
text_cols = [col for col in df.columns if col.startswith('text_')]
df[text_cols] = df[text_cols].astype('category')

# Use the minimum possible int type for label column
label_col = 'label'
label_min = df[label_col].min()
label_max = df[label_col].max()

if label_min >= 0 and label_max <= 255:
    df[label_col] = df[label_col].astype('uint8')
elif label_min >= -128 and label_max <= 127:
    df[label_col] = df[label_col].astype('int8')
elif label_min >= 0 and label_max <= 65535:
    df[label_col] = df[label_col].astype('uint16')
else:
    df[label_col] = df[label_col].astype('int32')

# Write the optimized data to a new CSV file
df.to_csv('optimized_data.csv', index=False)
