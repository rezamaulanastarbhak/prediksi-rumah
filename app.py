import streamlit as st
import pandas as pd
import joblib
from babel.numbers import format_currency

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")
st.title("Aplikasi Prediksi Harga Rumah")

#@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/random_forest_regressor_model.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        feature_scaler = joblib.load("models/feature_scaler.pkl")
        price_scaler = joblib.load("models/price_scaler.pkl")
        return model, feature_columns, feature_scaler, price_scaler
    except:
        return None, None, None, None

# load asset
model, feature_columns, feature_scaler, price_scaler = load_assets()

if model is None:
    st.error("Gagal loading modal")
else:
    st.markdown("""
    Masukan detail property di bawah ini untuk mendapatkan estimasi harga.
    Aplikasi ini menggunakan model *machine learning* untuk memberikan prediksi harga
    """)

    with st.form("Prediction_Form"):
      st.header("Masukan Detail Property")

      col1, col2 = st.columns(2)

      with col1:
        area = st.number_input("Luas Bangunan (m)", min_value=30.0,
                               max_value=1000.0, value=120.0, step=10.0)
        bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3, step=1)
        garage = st.number_input("Kapasitas Garasi", min_value=0, max_value=10, value=1, step=1)

      with col2:
        building_area = st.number_input("Luas Bangunan (m2)", min_value=30.0, max_value=900.0,
                                        value=90.0, step=10.0)
        bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2, step=1)
        city = st.selectbox("Kota", ("Jakarta Selatan", "Jakarta Timur", "Jakarta Barat", "Jakarta Pusat",
                                     'Depok', 'Bogor', 'Bekasi', 'Tangerang', 'Tangerang Selatan'))
        submit_button = st.form_submit_button("Prediksi Harga")

    if submit_button:
        try:
          input_data = {
            'area': [area],
            'bedrooms': [bedrooms],
            'garage': [garage],
            'building_area': [building_area],
            'bathrooms': [bathrooms],
            'city': [city]
          }
          input_df = pd.DataFrame(input_data)
          # scaling feature 
          input_df[['area','building_area']] = feature_scaler.transform(input_df[['area','building_area']])
          # one hot encoding
          input_df = pd.get_dummies(input_df, columns=['city', 'bedrooms', 'bathrooms', 'garage'],
                                  prefix=['City','Bedrooms', 'Bathrooms', 'Garage'])

          # urutkan columns 
          input_processed = input_df.reindex(columns=feature_columns, fill_value=0)

          # prediksi (panggil model)
          scaled_prediction = model.predict(input_processed)
          original_price = price_scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0][0] * 1000000

          # format rupiah
          original_price = format_currency(original_price, "IDR", locale='id_ID',)
          
          st.success(st.metric(label = 'Prediksi Harga',
                    value = original_price) )
          
        except:
          st.error("Gagal load data")
        

