import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Load the machine learning model
model = pickle.load(open('prediksi_gaji.sav', 'rb'))

# Load the dataset
df1 = pd.read_csv('SalaryData.csv')

# App title and description
st.title('Prediksi Gaji Pegawai')

# Import Image
st.image('./img.jpg')

st.write("Aplikasi dataset yang kami kembangkan melalui web ini adalah memprediksi gaji pegawai berdasarkan usia dan jumlah pengalaman kerja.")

# Display dataset and its statistics
st.header("Dataset")
st.dataframe(df1)

st.header("Statistik Dataset")
st.dataframe(df1)
st.write(df1.describe())

# Visualizations
selected_column = st.selectbox('Pilih kolom untuk ditampilkan:', df1.columns)
bar_chart_data = df1[selected_column].value_counts()
st.bar_chart(bar_chart_data)

# Prediction button
age = st.number_input('Usia')
years_of_experience = st.number_input('Jumlah Pengalaman Kerja')

# Display the prediction in a formatted way
if st.button('Prediksi Gaji'):
    X = np.array([[age, years_of_experience]])
    gaji_X = model.predict(X)
    st.write(f"Prediksi gajinya adalah {int(gaji_X[0])}")

# Add a reset button
if st.button('Reset'):
    st.experimental_rerun()

# Footer and additional information
st.sidebar.header("Tentang Aplikasi")
st.sidebar.markdown("Aplikasi ini dikembangkan oleh:\n - Ahmad Ilham M (223307060)\n - Al Jihan Verina A (223307062)\n - Rico Ronaldo (223307081).")
st.sidebar.write("Versi 1.0")

# Provide information about the model
st.sidebar.header("Informasi Model")
st.sidebar.write("Model yang digunakan adalah model linear regression untuk memprediksi gaji pegawai berdasarkan data-data yang ada.")

# Provide a link to the dataset source or any references used
st.sidebar.header("Referensi")
st.sidebar.write("Dataset:[https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer ]")

# Add a feedback section
st.sidebar.header("Feedback")
st.sidebar.write("Berikan masukan atau laporkan masalah melalui.")

# Add a section for user instructions or help
st.sidebar.header("Panduan Pengguna")
st.sidebar.write("Cara penggunaan aplikasi:")
st.sidebar.write("1. Isi nilai-nilai input.")
st.sidebar.write("2. Klik tombol 'Prediksi gaji'.")
st.sidebar.write("3. Periksa hasil prediksi.")