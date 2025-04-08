import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

file_path = r"C:\Users\bachh\patient-no-show-prediction\data\KaggleV2-May-2016.csv.csv"

if os.path.exists(file_path):
    print("✅ File exists!")
    df = pd.read_csv(file_path)
else:
    print("❌ File NOT found! Check the path.")
    exit()

# Drop unnecessary columns
df = df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'])

# Target conversion
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

# Feature engineering
df['Is_Healthy'] = ((df['Hypertension'] == 0) & (df['Diabetes'] == 0) & 
                    (df['Alcoholism'] == 0) & (df['Handicap'] == 0)).astype(int)
df['Is_Young'] = (df['Age'] < 40).astype(int)
df['Healthy_Young'] = ((df['Is_Healthy'] == 1) & (df['Is_Young'] == 1)).astype(int)

# One-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood'], drop_first=True)

# Features and target
X = df.drop('No-show', axis=1)
y = df['No-show']

# Assign weights: more weight to healthy young who showed up (No-show = 0)
weights = ((df['Healthy_Young'] == 1) & (y == 0)).astype(int) * 1 + 1  # weight = 2 if healthy young & showed up, else 1

# Balance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Also resample weights
resampled_weights = pd.Series(weights).iloc[sm.sample_indices_].values

# Train-test split
X_train, X_test, y_train, y_test, w_train, _ = train_test_split(X_res, y_res, resampled_weights, test_size=0.2, random_state=42)

# Model training (no need for GridSearch to keep it simpler here)
model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Weighted Model Accuracy: {acc:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("💾 Weighted model saved as model.pkl")
