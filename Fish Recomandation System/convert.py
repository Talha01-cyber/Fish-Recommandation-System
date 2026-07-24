import pandas as pd

# Load the Excel file
excel_file = 'realfishdataset.xlsx'  # Make sure this file is in the same folder
df = pd.read_excel(excel_file)

# Optionally: preview or clean data here
# df = df.dropna()  # If needed, remove missing rows

# Save the DataFrame as CSV
csv_file = 'processed_fish_data.csv'
df.to_csv(csv_file, index=False)

print(f"Conversion complete. CSV saved as '{csv_file}'")
