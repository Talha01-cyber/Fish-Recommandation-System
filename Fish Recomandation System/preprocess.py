import pandas as pd

# Step 1: Load CSV
df = pd.read_csv('processed_fish_data.csv')

# Step 2: Basic preprocessing
# Remove duplicates
df = df.drop_duplicates()

# Remove rows with missing/null values
df = df.dropna()

# Ensure correct data types
df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
df['turbidity'] = pd.to_numeric(df['turbidity'], errors='coerce')

# Step 3: Remove rows with out-of-range values (optional thresholds)
df = df[
    (df['ph'] >= 0) & (df['ph'] <= 14) &
    (df['temperature'] >= 0) & (df['temperature'] <= 100) &
    (df['turbidity'] >= 0)
]

# Step 4: Save cleaned data
df.to_csv('preprocessed_fish_data.csv', index=False)

print("✅ Preprocessing complete. Cleaned data saved to 'preprocessed_fish_data.csv'.")
