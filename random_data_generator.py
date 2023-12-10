import pandas as pd
import numpy as np

# Load the existing dataset to understand the range and type of data in each column
df = pd.read_csv('/mnt/data/27_natural_disaster_prediction_data (1).csv')

# Function to generate random values based on the existing column data
def generate_random_data(column):
    if pd.api.types.is_numeric_dtype(column):
        return np.random.uniform(column.min(), column.max(), size=n_rows)
    elif pd.api.types.is_datetime64_dtype(column):
        return pd.to_datetime(np.random.choice(pd.date_range(column.min(), column.max()), size=n_rows))
    elif pd.api.types.is_object_dtype(column):
        return np.random.choice(column.unique(), size=n_rows)
    else:
        return np.random.choice(column.unique(), size=n_rows)

# Generate random data for each column
n_rows = 3000
random_data = {col: generate_random_data(df[col]) for col in df.columns}

# Create a new DataFrame with the generated random data
random_df = pd.DataFrame(random_data)

# Save the generated data to a new CSV file
output_file_path = '/mnt/data/generated_random_data.csv'
random_df.to_csv(output_file_path, index=False)