import streamlit as st
import pandas as pd
import pickle

# Load the MinMax scaler from the pickle file for normalization
scaler_filename = 'Data_Normalisasi_MinMax.pkl'
with open(scaler_filename, 'rb') as scaler_file:
    minmax_scaler = pickle.load(scaler_file)

# Load the Random Forest model from the pickle file
model_filename = 'model_terbaik_rf.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
def main():
    st.title("Klasifikasi Kelenjar Payudara")

    # Input columns
    I0 = st.number_input("Masukkan Nilai I0:")
    PA500 = st.number_input("Masukkan Nilai PA500:")
    HFS = st.number_input("Masukkan Nilai HFS:")
    DA = st.number_input("Masukkan Nilai DA:")
    Area = st.number_input("Masukkan Nilai Area:")
    ADA = st.number_input("Masukkan Nilai A/DA:")
    Maip = st.number_input("Masukkan Nilai Max IP:")
    DR = st.number_input("Masukkan Nilai DR:")
    P = st.number_input("Masukkan Nilai P:")

    # Create a dataframe with user input
    user_data = pd.DataFrame({'I0': [I0], 'PA500': [PA500], 'HFS': [HFS], 'DA': [DA], 'Area': [Area], 'A/DA': [ADA], 'Max IP':[Maip], 'DR': [DR], 'P': [P]})

    # Add a button to trigger prediction
    if st.button("Prediksi"):
        # Normalize the user input data using MinMax scaler
        normalized_data = minmax_scaler.transform(user_data)

        # Classify using the Random Forest model
        prediction = model.predict(normalized_data)

        # Display the result
        st.subheader("Prediction:")
        st.write(prediction[0])

if __name__ == '__main__':
    main()