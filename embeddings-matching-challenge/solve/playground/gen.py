import pandas as pd

# Load the CSV file
input_csv = './files/train_data.csv'  # Replace with the path to your input file
output_csv = './files/train_data_processed.csv'  # Replace with the path to your output file

# List of index values you want to filter
# 0,87
# 1,105
filter_indices = [87, 105]  # Replace with your desired indices

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Filter rows based on the given indices
filtered_df = df.iloc[filter_indices]

print(filtered_df)
# Save the filtered DataFrame to a new CSV
filtered_df.to_csv(output_csv, index=False)

print(f"Filtered rows saved to {output_csv}.")

