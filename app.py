import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
label_encoders = joblib.load('label_encoders.pkl')
lr_base = joblib.load('linear_regression_base.pkl')
rf = joblib.load('random_forest.pkl')
lr_final = joblib.load('linear_regression_final.pkl')

# Load datasets
df_raw = pd.read_csv('vehicles.csv', low_memory=False)
df_preprocessed = pd.read_csv('vehicles_preprocessed1.csv')

# Create mappings
make_mapping = dict(zip(df_raw['make'], df_preprocessed['make']))
model_mapping = dict(zip(df_raw['model'], df_preprocessed['model']))
fuel_mapping = dict(zip(df_raw['fuelType'], df_preprocessed['fuelType']))
vclass_mapping = dict(zip(df_raw['VClass'], df_preprocessed['VClass']))

# Reverse mappings
make_reverse = {v: k for k, v in make_mapping.items()}
model_reverse = {v: k for k, v in model_mapping.items()}
fuel_reverse = {v: k for k, v in fuel_mapping.items()}
vclass_reverse = {v: k for k, v in vclass_mapping.items()}

# Default values
default_make = make_mapping['Chevrolet']
default_model = model_mapping.get('Malibu', model_mapping.get('Cruze', 2470))
default_year = 2016
default_cylinders = 4.0
default_displ = 2.0
default_fuel = fuel_mapping['Regular']
default_vclass = vclass_mapping['Compact Cars']

# Prediction function
def predict_mpg(make, model, year, cylinders, displ, fuel_type, v_class):
    input_data = pd.DataFrame({
        'make': [make],
        'model': [model],
        'year': [year],
        'cylinders': [cylinders],
        'displ': [displ],
        'fuelType': [fuel_type],
        'VClass': [v_class]
    })

    # st.write("🔹 Raw input:", input_data)

    # Convert to array
    X = input_data.values
    # st.write("🔹 Feature array (X):", X)

    # Step 1: LR prediction
    y_pred_lr = lr_base.predict(X)
    # st.write("📈 y_pred_lr (1st):", y_pred_lr)

    # Step 2: RF prediction using LR output
    X_rf_input = np.column_stack((X, y_pred_lr))
    y_pred_rf = rf.predict(X_rf_input)
    # st.write("🌲 y_pred_rf (2nd):", y_pred_rf)

    # Step 3: Final LR prediction using RF output
    X_final = np.column_stack((X, y_pred_lr, y_pred_rf))
    # st.write("✅ X_final shape:", X_final.shape)  # Should be (1, 9)
    final_pred = lr_final.predict(X_final)

    return final_pred[0]

# Streamlit UI
st.title("Mileage Prediction System")
st.subheader("Enter Vehicle Details")
st.write("(Leave unchanged if unknown)")

# Make
default_make_label = make_reverse[default_make]
make = st.selectbox("Make", sorted(make_mapping.keys()), index=sorted(make_mapping.keys()).index(default_make_label))

# Model
default_model_label = model_reverse[default_model]
model = st.selectbox("Model", sorted(model_mapping.keys()), index=sorted(model_mapping.keys()).index(default_model_label))

# Year
year = st.slider("Year", 1984, 2025, default_year)

# Cylinders
cylinders = st.slider("Cylinders", 2.0, 16.0, default_cylinders, step=1.0)

# Displacement
displ = st.number_input("Displacement (L)", min_value=0.0, max_value=8.4, value=default_displ)

# Fuel Type
default_fuel_label = fuel_reverse[default_fuel]
fuel_type = st.selectbox("Fuel Type", sorted(fuel_mapping.keys()), index=sorted(fuel_mapping.keys()).index(default_fuel_label))

# Vehicle Class
default_vclass_label = vclass_reverse[default_vclass]
v_class = st.selectbox("Vehicle Class", sorted(vclass_mapping.keys()), index=sorted(vclass_mapping.keys()).index(default_vclass_label))

# Predict button
if st.button("Predict MPG"):
    encoded_input = [
        make_mapping[make],
        model_mapping[model],
        year,
        cylinders,
        displ,
        fuel_mapping[fuel_type],
        vclass_mapping[v_class]
    ]
    prediction = predict_mpg(*encoded_input)
    st.success(f"Predicted MPG: {prediction:.2f}")