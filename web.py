import pickle
import streamlit as st
import numpy as np
import pandas as pd

model = pickle.load(open('prediksi_gaji.sav', 'rb'))

st.title('Prediksi Gaji Pegawai Berdasarkan Usia dan Jumlah Pengalaman Kerja')

age = st.number_input('Usia')
years_of_experience = st.number_input('Jumlah Pengalaman Kerja')

if st.button('Prediksi'):
    X = np.array([[age, years_of_experience]])
    gaji_X = model.predict(X)
    st.write(f"Prediksi gajinya adalah {int(gaji_X[0])}")

data = pd.read_csv('SalaryData.csv')
st.dataframe(data)
st.dataframe(data.describe())

selected_column = st.selectbox('Pilih kolom untuk ditampilkan:', data.columns)
bar_chart_data = data[selected_column].value_counts()
st.bar_chart(bar_chart_data)

