import pandas as pd
# load the audio and text feature CSV files into pandas dataframes
audio_df = pd.read_csv('audio_features.csv')
text_df = pd.read_csv('text_features.csv')

# extract the common sequence identifier from the filenames:
audio_df['sequence'] = audio_df['audio_features.csv'].apply(lambda x: x.split('_')[0])
text_df['sequence'] = text_df['text_features.csv'].apply(lambda x: x.split('_')[0])

# add a new column to each dataframe to identify its source
audio_df['source'] = 'audio'
text_df['source'] = 'text'

# concatenate the two dataframes vertically
concatenated_df = pd.concat([audio_df, text_df], ignore_index=True)

# group the concatenated dataframe by the sequence and aggregate the sources
aggregated_df = concatenated_df.groupby('sequence')['source'].apply(lambda x: ','.join(x)).reset_index()

# save the aggregated dataframe to a new CSV file
aggregated_df.to_csv('aggregated_features.csv', index=False)
