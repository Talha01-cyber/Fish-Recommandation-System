# 1. Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#  2. Load the dataset
df = pd.read_csv("realfishdataset.csv")

# 3. Feature and Target selection
X = df[["ph", "temperature", "turbidity"]]
y = df["fish"]

#  4. Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#  5. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

#  6. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 📌 7. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 📌 8. Predict on test data
y_pred = rf_model.predict(X_test)

# 📌 9. Evaluate model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 📌 10. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📌 11. Save the model and label encoder
joblib.dump(rf_model, "fish_model.pkl")
joblib.dump(label_encoder, "fish_label_encoder.pkl")