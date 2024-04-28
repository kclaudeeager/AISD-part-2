import os
import pandas as pd

# Get the directory of this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Construct the full path to the CSV file
csv_path = os.path.join(dir_path, 'experiments', 'iris_extended_encoded.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Create a sample DataFrame (let's say 10% of the original DataFrame)
sample_df = df.sample(frac=0.1)

# Construct the full path to the output CSV file
output_path = os.path.join(dir_path, 'experiments', 'sample_iris_extended_encoded.csv')

# Write the sample DataFrame to a new CSV file
sample_df.to_csv(output_path, index=False)

print(f"Sample DataFrame saved to {output_path}")