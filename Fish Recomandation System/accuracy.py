import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('processed_fish_data.csv')
X = df.drop('Fish_Name', axis=1)
y = df['Fish_Name']

# Load model
with open('random_forest_fish_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict and evaluate
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy on full dataset: {accuracy:.2f}")