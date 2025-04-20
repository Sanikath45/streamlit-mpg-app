import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Function to load artifacts with error handling
def load_artifact(file_path, file_type="file"):
    try:
        if file_type == "pkl":
            return joblib.load(file_path)
        elif file_type == "csv":
            return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Error: {file_path} not found. Please ensure the file is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Load artifacts with error handling
label_encoders = load_artifact('label_encoders.pkl', 'pkl')
lr_base = load_artifact('linear_regression_base.pkl', 'pkl')
rf = load_artifact('random_forest.pkl', 'pkl')
lr_final = load_artifact('linear_regression_final.pkl', 'pkl')
df_raw = load_artifact('vehicles.csv', 'csv')
df_preprocessed = load_artifact('vehicles_preprocessed1.csv', 'csv')

# Check if all artifacts loaded successfully
if any(x is None for x in [label_encoders, lr_base, rf, lr_final, df_raw, df_preprocessed]):
    st.stop()

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
default_make = make_mapping.get('Chevrolet', list(make_mapping.values())[0])
default_model = model_mapping.get('Malibu', model_mapping.get('Cruze', 2470))
default_year = 2016
default_cylinders = 4.0
default_displ = 2.0
default_fuel = fuel_mapping.get('Regular', list(fuel_mapping.values())[0])
default_vclass = vclass_mapping.get('Compact Cars', list(vclass_mapping.values())[0])

# Prediction function
def predict_mpg(make, model, year, cylinders, displ, fuel_type, v_class):
    try:
        input_data = pd.DataFrame({
            'Brand': [make],
            'model': [model],
            'year': [year],
            'cylinders': [cylinders],
            'displ': [displ],
            'fuelType': [fuel_type],
            'VClass': [v_class]
        })

        # Convert to array
        X = input_data.values

        # Step 1: LR prediction
        y_pred_lr = lr_base.predict(X)

        # Step 2: RF prediction using LR output
        X_rf_input = np.column_stack((X, y_pred_lr))
        y_pred_rf = rf.predict(X_rf_input)

        # Step 3: Final LR prediction using RF output
        X_final = np.column_stack((X, y_pred_lr, y_pred_rf))
        final_pred = lr_final.predict(X_final)

        return final_pred[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Streamlit UI
st.title("Mileage Prediction System")
st.subheader("Enter Vehicle Details for Fuel Efficiency Prediction")
st.write("(Leave unchanged if unknown.)")

# Make
default_make_label = make_reverse.get(default_make, list(make_reverse.values())[0])
make = st.selectbox("Make", sorted(make_mapping.keys()), index=sorted(make_mapping.keys()).index(default_make_label))

# Model
default_model_label = model_reverse.get(default_model, list(model_reverse.values())[0])
model = st.selectbox("Model", sorted(model_mapping.keys()), index=sorted(model_mapping.keys()).index(default_model_label))

# Year
year = st.slider("Year", 1984, 2025, default_year)

# Cylinders
cylinders = st.slider("Cylinders (Number of Cylinders)", 2.0, 16.0, default_cylinders, step=1.0)

# Displacement
displ = st.number_input("Displacement (Liters)", min_value=0.0, max_value=8.4, value=default_displ)

# Fuel Type
default_fuel_label = fuel_reverse.get(default_fuel, list(fuel_reverse.values())[0])
fuel_type = st.selectbox("Fuel Type", sorted(fuel_mapping.keys()), index=sorted(fuel_mapping.keys()).index(default_fuel_label))

# Vehicle Class
default_vclass_label = vclass_reverse.get(default_vclass, list(vclass_reverse.values())[0])
v_class = st.selectbox("Vehicle Class", sorted(vclass_mapping.keys()), index=sorted(vclass_mapping.keys()).index(default_vclass_label))

# Predict button
if st.button("Predict Fuel Efficiency"):
    encoded_input = [
        make_mapping[make],
        model_mapping[model],
        year,
        cylinders,
        displ,
        fuel_mapping[fuel_type],
        vclass_mapping[v_class]
    ]
    prediction_mpg = predict_mpg(*encoded_input)
    if prediction_mpg is not None:
        prediction_km_l = prediction_mpg * 0.425143707  # Convert MPG to km/L
        st.success(f"Predicted Fuel Efficiency: {prediction_mpg:.2f} MPG (US) or {prediction_km_l:.2f} km/L")
