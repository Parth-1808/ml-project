import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\bachh\patient-no-show-prediction\data\KaggleV2-May-2016.csv.csv")

# Drop irrelevant columns
df.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], inplace=True)

# Encode target
df['No-show'] = df['No-show'].map({'No':0 , 'Yes': 1 })

# Rename column to match your Streamlit prediction logic
df.rename(columns={'Hypertension': 'Hipertension'}, inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood'], drop_first=False)

# Features and target
X = df.drop(columns=['No-show'])
y = df['No-show']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

# Optional: Print model input features
print("Model trained and saved as model.pkl")
print("Model expects features:")
print(model.feature_names_in_)
