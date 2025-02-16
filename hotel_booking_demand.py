# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Judul Utama
st.title('Survival Rate Predictor')
st.text('This web can be used to predict your survival rate')



# Menambahkan sidebar
st.sidebar.header("Please input your features")

def create_user_input():
    
    # Numerical Features with constraints
    previous_cancellations = st.sidebar.slider('previous_cancellations', min_value=0, max_value=26, value=0)
    booking_changes = st.sidebar.slider('booking_changes', min_value=0, max_value=21, value=0)
    required_car_parking_spaces = st.sidebar.slider('required_car_parking_spaces', min_value=0, max_value=8, value=0)
    total_of_special_requests = st.sidebar.slider('total_of_special_requests', min_value=0, max_value=5, value=0)
    days_in_waiting_list = st.sidebar.slider('days_in_waiting_list', min_value=0, max_value=391, value=0)

    # Categorical Features with constraints
    market_segment = st.sidebar.radio('market_segment', ['Corporate', 'Offline TA/TO', 'Direct', 'Groups', 
                                                         'Online TA', 'Complementary', 'Aviation', 'Undefined'])
    deposit_type = st.sidebar.radio('deposit_type', ['No Deposit', 'Non Refund', 'Refundable'])
    customer_type = st.sidebar.radio('customer_type', ['Transient-Party', 'Transient', 'Contract', 'Group'])
    reserved_room_type = st.sidebar.radio('reserved_room_type', ['A', 'G', 'E', 'D', 'B', 'C', 'F', 'H', 'P', 'L'])

    # Creating a dictionary with user input
    user_data = {
        'previous_cancellations': previous_cancellations,
        'booking_changes': booking_changes,
        'required_car_parking_spaces': required_car_parking_spaces,
        'total_of_special_requests': total_of_special_requests,
        'days_in_waiting_list': days_in_waiting_list,
        'market_segment': market_segment,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'reserved_room_type': reserved_room_type
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get customer data
data_customer = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'Hotel_booking.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict to data
kelas = model_loaded.predict(data_customer)
probability = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities

# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Prediction Result')
    if kelas == 1:
        st.write('Class 1: This customer will Survive')
    else:
        st.write('Class 2: This customer will Survive')
    
    # Displaying the probability of the customer buying
    st.write(f"Probability of Survive: {probability[1]:.2f}")  # Probability of class 1 (BUY)
