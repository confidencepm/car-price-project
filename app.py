import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
import copy

warnings.filterwarnings('ignore')

# 1. Loading the saved model
model_filename = 'best_car_price_model.joblib'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()

# 2. Defining the data and preprocessing constants
SCALER_MEANS = {
    'vehicle_age': 6.0614914668968725,
    'mileage': 19.751153463045355,
    'engine': 1471.934324988379,
    'max_power': 98.95508533103127,
    'km_driven_log': 10.681394415842837,
}

SCALER_STDS = {
    'vehicle_age': 3.0145644039783646,
    'mileage': 4.142013863579522,
    'engine': 497.6243872815609,
    'max_power': 39.01055268527468,
    'km_driven_log': 0.7589327116209047,
}

# Defining the dropdown options
ALL_BRANDS = ['BMW', 'Datsun', 'Hyundai', 'Jaguar', 'Land Rover', 
              'Maruti', 'Mercedes-Benz', 'Renault', 'Toyota']

ALL_SELLER_TYPES = ['Individual', 'Dealer', 'Trustmark Dealer']
ALL_FUEL_TYPES = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
ALL_TRANSMISSIONS = ['Manual', 'Automatic']


# 3. Defining the prediction function
def preprocess_input(data_raw):
    """
    Taking a dictionary of raw user input and converting it into the
    1D array (of 20 features) that the model expects.
    THIS FUNCTION IS NOW ALIGNED WITH THE NOTEBOOK.
    """
    
    # Creating a deep copy to force Streamlit to see this as a new object
    data = copy.deepcopy(data_raw)
    
    # 1. Log Transforming 'km_driven'
    data['km_driven_log'] = np.log1p(data['km_driven'])
    
    # 2. Scaling the numerical features using the *correct* stats
    # We are scaling all of them first
    for feature, mean in SCALER_MEANS.items():
        std = SCALER_STDS[feature]
        data[feature] = (data[feature] - mean) / std

    # 3. Creating the final feature vector in the *EXACT* order from the notebook
    processed_data = []
    
    # This list is based on my notebook's output    
    processed_data.append(data['vehicle_age'])         # 1.
    processed_data.append(data['mileage'])             # 2.
    processed_data.append(data['engine'])              # 3.
    processed_data.append(data['max_power'])           # 4.
    processed_data.append(data['seats'])               # 5. (Not scaled, which is correct)
    processed_data.append(data['km_driven_log'])       # 6.
    
    # Adding one-hot encoded features
    processed_data.append(1 if data['brand'] == 'BMW' else 0)              # 7.
    processed_data.append(1 if data['brand'] == 'Datsun' else 0)           # 8.
    processed_data.append(1 if data['brand'] == 'Hyundai' else 0)          # 9.
    processed_data.append(1 if data['brand'] == 'Jaguar' else 0)           # 10.
    processed_data.append(1 if data['brand'] == 'Land Rover' else 0)       # 11.
    processed_data.append(1 if data['brand'] == 'Maruti' else 0)           # 12.
    processed_data.append(1 if data['brand'] == 'Mercedes-Benz' else 0)    # 13.
    processed_data.append(1 if data['brand'] == 'Renault' else 0)          # 14.
    processed_data.append(1 if data['brand'] == 'Toyota' else 0)           # 15.
    
    processed_data.append(1 if data['seller_type'] == 'Individual' else 0) # 16.
    
    processed_data.append(1 if data['fuel_type'] == 'Diesel' else 0)       # 17.
    processed_data.append(1 if data['fuel_type'] == 'LPG' else 0)          # 18.
    processed_data.append(1 if data['fuel_type'] == 'Petrol' else 0)       # 19.
    
    processed_data.append(1 if data['transmission_type'] == 'Manual' else 0) # 20.
    
    # Returning as a 2D array, as the model expects a "batch" of 1
    return [processed_data]

# 4. Building the Streamlit UI
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Car Price Prediction Application")
st.write("Enter the car's details to get a predicted selling price.")

with st.form("prediction_form"):
    st.header("Car Details")
    
    # Splitting into columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", options=ALL_BRANDS)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=40, value=5)
        km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=1000000, value=50000)
        seats = st.selectbox("Seats", options=[2, 4, 5, 6, 7, 8, 9, 10], index=2)

    with col2:
        fuel_type = st.selectbox("Fuel Type", options=ALL_FUEL_TYPES)
        transmission_type = st.selectbox("Transmission", options=ALL_TRANSMISSIONS)
        seller_type = st.selectbox("Seller Type", options=ALL_SELLER_TYPES)
        mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=45.0, value=19.5)

    with col3:
        engine = st.number_input("Engine (CC)", min_value=600, max_value=6000, value=1484)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=92.0)

    st.write("")
    submit_button = st.form_submit_button(label="Predict Price")

# 5. Handling the prediction
if submit_button:
    # Collecting all inputs into a dictionary
    user_data = {
        'brand': brand,
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'seats': seats,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type,
        'seller_type': seller_type,
        'mileage': mileage,
        'engine': engine,       
        'max_power': max_power  
    }
    
    st.header("Debug: Raw User Input")
    st.json(user_data)

    try:
        # Preprocessing the data
        processed_user_data = preprocess_input(user_data)
        
        st.header("Debug: Processed Data (Input to Model)")
        # Displaying the 20 numbers being sent to the model
        st.json(processed_user_data[0]) 

        # Making the prediction
        log_price = model.predict(processed_user_data)
        predicted_price = np.expm1(log_price[0])
        
        # Formatting the price for display
        st.success(f"**Predicted Selling Price: ₹ {predicted_price:,.0f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
