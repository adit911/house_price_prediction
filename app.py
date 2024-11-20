import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.drop(columns=['site_location', 'society'], inplace=True)
    data['availability'] = data['availability'].apply(lambda x: 'Ready to move in' if 'Ready' in x else 'Date')

    def extract_bhk(size):
        if isinstance(size, str):
            if 'BHK' in size:
                return int(size.split()[0])
            elif 'Bedroom' in size:
                return int(size.split()[0])
        return None

    data['BHK'] = data['size'].apply(extract_bhk)
    data.dropna(subset=['BHK'], inplace=True)

    def convert_sqft_to_num(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            try:
                tokens = x.split('-')
                if len(tokens) == 2:
                    return (float(tokens[0]) + float(tokens[1])) / 2
            except (ValueError, TypeError):
                pass
            return None

    data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
    data.dropna(subset=['total_sqft'], inplace=True)
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data.dropna(subset=['price'], inplace=True)
    data['price_per_square_ft'] = data['price'] / data['total_sqft']
    data['price_per_room'] = data['price'] / data['BHK']
    data.drop(columns=['size'], inplace=True)
    
    categorical_columns = ['area_type', 'availability']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    return data

# Load the data
file_path = "Pune_House_Data.csv"  # Replace with your file path
data = load_data(file_path)

# Split features and target
X = data.drop(columns=['price'])
y = data['price']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_scaled, y)

# Streamlit App
st.title("House Price Prediction App")

# Sidebar Inputs
st.sidebar.header("Provide House Details")
total_sqft = st.sidebar.number_input("Total Square Feet", min_value=100.0, value=1500.0)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, value=3)
balcony = st.sidebar.number_input("Number of Balconies", min_value=0, value=2)
BHK = st.sidebar.number_input("Number of BHKs", min_value=1, value=3)
price_per_sqft = st.sidebar.number_input("Price per Square Foot", min_value=1.0, value=450.0)
price_per_room = st.sidebar.number_input("Price per Room", min_value=1.0, value=3000.0)
area_type = st.sidebar.selectbox("Area Type", ["Carpet Area", "Plot Area", "Super built-up Area"])
availability = st.sidebar.selectbox("Availability", ["Ready to move in", "Date"])

# One-hot encode area_type and availability
area_type_cols = {'Carpet Area': 1, 'Plot Area': 0, 'Super built-up Area': 0}
availability_col = 1 if availability == "Ready to move in" else 0

# Predict button
if st.button("Predict Price"):
    # Create input data
    input_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony],
        'BHK': [BHK],
        'price_per_square_ft': [price_per_sqft],
        'price_per_room': [price_per_room],
        'area_type_Carpet  Area': [area_type_cols.get(area_type, 0)],
        'area_type_Plot  Area': [1 if area_type == 'Plot Area' else 0],
        'area_type_Super built-up  Area': [1 if area_type == 'Super built-up Area' else 0],
        'availability_Ready to move in': [availability_col]
    })

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict price
    predicted_price = rf_model.predict(input_scaled)
    st.success(f"Predicted Price: ₹{predicted_price[0]:,.2f}")


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and scaler
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl')

# Streamlit UI for input
st.title("House Price Prediction")

total_sqft = st.number_input("Enter Total Square Footage", min_value=0)
bath = st.number_input("Enter Number of Bathrooms", min_value=0)
balcony = st.number_input("Enter Number of Balconies", min_value=0)
BHK = st.number_input("Enter Number of BHK", min_value=0)
price_per_square_ft = st.number_input("Enter Price per Square Foot", min_value=0)
price_per_room = st.number_input("Enter Price per Room", min_value=0)

# One-hot encoded columns for area_type and availability (for simplicity, using default values)
area_type = 'Carpet  Area'  # Modify according to the selected area_type
availability = 'Ready to move in'  # Modify according to availability status

# Prepare the input data in the same format as training data
input_data = {
    'total_sqft': [total_sqft],
    'bath': [bath],
    'balcony': [balcony],
    'BHK': [BHK],
    'price_per_square_ft': [price_per_square_ft],
    'price_per_room': [price_per_room],
    'area_type_Carpet  Area': [1 if area_type == 'Carpet  Area' else 0],
    'area_type_Plot  Area': [1 if area_type == 'Plot  Area' else 0],
    'area_type_Super built-up  Area': [1 if area_type == 'Super built-up  Area' else 0],
    'availability_Ready to move in': [1 if availability == 'Ready to move in' else 0],
}

input_df = pd.DataFrame(input_data)

# Apply the same scaling transformation as used during training
input_scaled = scaler.transform(input_df)

# Predict using the trained Random Forest model
prediction = rf_model.predict(input_scaled)

# Display the predicted price
st.write(f"Predicted House Price: ₹{prediction[0]:,.2f}")
