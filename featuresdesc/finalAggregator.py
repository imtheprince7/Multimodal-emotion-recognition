import pandas as pd

# Read the attention module output from CSV file
attention_output = pd.read_csv("attentionModel_Output.csv")

# Aggregate the attention weights by the filename
aggregated_output = attention_output.groupby("attention_output").mean()

# Save the aggregated output to a CSV file
aggregated_output.to_csv("aggregated_Final.csv")
