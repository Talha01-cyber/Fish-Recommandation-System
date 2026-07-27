import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
print("Loading dataset processed_fish_data.csv...")
df = pd.read_csv('processed_fish_data.csv')

# Split features and labels
X = df[['ph', 'temperature', 'turbidity']]
y = df['fish']

# Encode the target labels
print("Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train Random Forest Classifier
print("Training RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
accuracy = rf_model.score(X_test, y_test)
print(f"Success - Accuracy on test set: {accuracy:.4f}")

# Save the trained model and label encoder
print("Saving random_forest_fish_model.pkl and label_encoder.pkl using pickle...")
with open('random_forest_fish_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Success - Model training complete and models saved.")
