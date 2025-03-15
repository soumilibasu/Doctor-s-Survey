import os
os.system("pip install --no-cache-dir --upgrade scikit-learn")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_excel("Doctor_data.xlsx", sheet_name="Dataset")
    df["Login Time"] = pd.to_datetime(df["Login Time"])
    df["Logout Time"] = pd.to_datetime(df["Logout Time"])
    
    # Create active hours binary array
    npi_active_hours = {}
    for _, row in df.iterrows():
        npi = row["NPI"]
        login_hour = row["Login Time"].hour
        logout_hour = row["Logout Time"].hour
        if logout_hour < login_hour:
            logout_hour += 24  # Handle cases where logout is past midnight
        
        if npi not in npi_active_hours:
            npi_active_hours[npi] = [0] * 24
        for hour in range(login_hour, logout_hour + 1):
            npi_active_hours[npi][hour % 24] = 1
    
    npi_active_df = pd.DataFrame.from_dict(npi_active_hours, orient="index", columns=[f"Hour_{i}" for i in range(24)])
    npi_active_df["NPI"] = npi_active_df.index
    df = df.drop_duplicates(subset=["NPI"]).drop(columns=["Login Time", "Logout Time"])
    df = df.merge(npi_active_df, on="NPI")
    
    return df

def train_model(df):
    le_region = LabelEncoder()
    le_specialty = LabelEncoder()
    le_state = LabelEncoder()
    
    df["Region"] = le_region.fit_transform(df["Region"])
    df["Speciality"] = le_specialty.fit_transform(df["Speciality"])
    df["State"] = le_state.fit_transform(df["State"])
    
    X = df[["State", "Region", "Speciality", "Count of Survey Attempts"]]
    y = df[[f"Hour_{i}" for i in range(24)]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    
    return model, X_test, df

def predict_best_doctors(hour, model, X_test, df):
    y_prob = model.predict_proba(X_test)
    hour_index = hour
    prob_active = np.array([prob[1] if len(prob) > 1 else 0 for prob in y_prob[hour_index]])
    npi_test = df.iloc[X_test.index]["NPI"].values
    results_df = pd.DataFrame({"NPI": npi_test, "Probability": prob_active})
    best_doctors = results_df[results_df["Probability"] > 0.5].sort_values(by="Probability", ascending=False)
    return best_doctors

df = load_data()
model, X_test, df = train_model(df)

st.title("Doctor Survey Targeting Web App")
st.write("Enter an hour to get the list of doctors most likely to attend the survey.")

hour = st.number_input("Enter Hour (0-23):", min_value=0, max_value=23, step=1)
if st.button("Predict Best Doctors"):
    best_doctors = predict_best_doctors(hour, model, X_test, df)
    st.write(best_doctors)
    csv = best_doctors.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "best_doctors.csv", "text/csv")
