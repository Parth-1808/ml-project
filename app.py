import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# Load dataset
df = pd.read_csv(r"C:\Users\bachh\patient-no-show-prediction\data\KaggleV2-May-2016.csv.csv")
df['No-show'] = df['No-show'].map({'No': 1, 'Yes': 0})

# Load model
model = joblib.load("model.pkl")

st.title("📅 Patient No-Show Prediction App")
st.markdown("Predict whether a patient will **miss** their scheduled medical appointment based on real-world behaviors.")

menu = st.sidebar.radio("Choose a section", ["📊 Data Visualizations", "🔮 Predict No-Show"])

# --- EDA Section ---
if menu == "📊 Data Visualizations":
    st.subheader("Exploratory Data Analysis")

    st.write("### Age Distribution of Patients")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=30, kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

    st.write("### Gender vs Show/No-Show")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='No-show', data=df, palette='Set2', ax=ax)
    ax.legend(title="Show (1=Show, 0=No-show)")
    st.pyplot(fig)

    st.write("### SMS Received vs Show/No-Show")
    fig, ax = plt.subplots()
    sns.countplot(x='SMS_received', hue='No-show', data=df, palette='coolwarm', ax=ax)
    ax.legend(title="Show (1=Show, 0=No-show)")
    st.pyplot(fig)

    st.write("### Top 20 Neighborhoods with Highest No-Show Rates")
    no_show_by_neigh = df.groupby('Neighbourhood')['No-show'].mean().sort_values(ascending=True)
    st.bar_chart(no_show_by_neigh.head(20))

# --- Prediction Section ---
elif menu == "🔮 Predict No-Show":
    st.subheader("Enter Patient Details")

    gender = st.selectbox("Gender", ["F", "M"])
    age = st.slider("Age", 0, 115, 30)
    neighbourhood = st.selectbox("Neighbourhood", sorted(df['Neighbourhood'].unique()))
    scholarship = st.selectbox("On Scholarship Program (Free/Reduced Cost)?", [0, 1])
    sms_received = st.selectbox("Received SMS Reminder?", [0, 1])
    hypertension = st.selectbox("Has Hypertension?", [0, 1])
    diabetes = st.selectbox("Has Diabetes?", [0, 1])
    alcoholism = st.selectbox("Alcohol Use Disorder?", [0, 1])
    handicap = st.selectbox("Handicap Level (0 = None)", [0, 1, 2, 3, 4])
    previous_no_shows = st.slider("Previous Missed Appointments", 0, 10, 0)
    days_until_appointment = st.slider("Days Until Appointment", 0, 60, 5)
    appointment_day = st.selectbox("Appointment Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

    if st.button("Predict No-Show Risk"):
        input_df = pd.DataFrame({
            'Age': [age],
            'Scholarship': [scholarship],
            'Hipertension': [hypertension],
            'Diabetes': [diabetes],
            'Alcoholism': [alcoholism],
            'Handcap': [handicap],
            'SMS_received': [sms_received]
        })

        input_df[f'Gender_{gender}'] = 1
        for g in ['Gender_M', 'Gender_F']:
            if g not in input_df.columns:
                input_df[g] = 0

        for col in df['Neighbourhood'].unique():
            input_df[f'Neighbourhood_{col}'] = 1 if col == neighbourhood else 0

        model_input_cols = model.feature_names_in_
        for col in model_input_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_input_cols]

        # Base model prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][0] * 100  # Class 0 = Miss

        # --- Enhanced Real-World Adjustments ---
        adjustment = 0
        reasons = []

        if age >= 60:
            adjustment -= 10
            reasons.append("Older age suggests higher reliability.")
        if sms_received == 1:
            adjustment -= 15
            reasons.append("Reminder SMS increases likelihood of showing up.")
        if hypertension == 1 or diabetes == 1:
            adjustment -= 10
            reasons.append("Chronic illness indicates seriousness.")
        if alcoholism == 1:
            adjustment += 15
            reasons.append("Alcohol use may affect responsibility.")
        if handicap > 0:
            adjustment += 10
            reasons.append("Handicap might impact access to clinic.")
        if scholarship == 1:
            adjustment -= 5
            reasons.append("Free care increases likelihood of attendance.")
        if days_until_appointment > 15:
            adjustment += 10
            reasons.append("Long wait increases risk of forgetting.")
        if appointment_day in ["Friday", "Saturday"]:
            adjustment += 5
            reasons.append("End of week appointments are more likely missed.")
        if previous_no_shows >= 2:
            adjustment += 15
            reasons.append("Past no-shows are a strong predictor.")
        if (age < 30 and hypertension == 0 and diabetes == 0 and alcoholism == 0
                and handicap == 0 and sms_received == 0 and previous_no_shows == 0):
            adjustment += 10
            reasons.append("Young, healthy, and no reminders — might skip casually.")

        adjusted_prob = probability + adjustment
        adjusted_prob = max(0, min(100, adjusted_prob))

        # --- Output Section ---
        st.subheader("Prediction Outcome")
        if adjusted_prob >= 50:
            st.error(f"❌ Patient is likely to **miss** the appointment.\n📊 **Adjusted Miss Probability: {adjusted_prob:.2f}%**")
        else:
            st.success(f"✅ Patient is likely to **attend** the appointment.\n📊 **Adjusted Miss Probability: {adjusted_prob:.2f}%**")

        st.markdown("---")
        st.markdown("#### Interpretation:")
        for reason in reasons:
            st.write(f"• {reason}")
