import math
import pandas as pd
import streamlit as st

st.title("Prediksi Risiko Diabetes (KNN)")

data = [
    ("P01", 25, 21.5,  90, 110, 0, "Rendah"),
    ("P02", 30, 23.1,  95, 115, 0, "Rendah"),
    ("P03", 35, 25.0, 105, 120, 0, "Rendah"),
    ("P04", 40, 27.5, 120, 125, 1, "Tinggi"),
    ("P05", 45, 30.2, 140, 135, 1, "Tinggi"),
    ("P06", 50, 31.5, 155, 140, 1, "Tinggi"),
    ("P07", 55, 33.0, 165, 145, 1, "Tinggi"),
    ("P08", 28, 22.5,  92, 112, 0, "Rendah"),
    ("P09", 38, 26.8, 115, 122, 0, "Rendah"),
    ("P10", 48, 29.5, 145, 130, 1, "Tinggi"),
]

st.subheader("Data Pasien")
st.dataframe(pd.DataFrame(data, columns=["Pasien", "Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga", "Kelas"]))

st.subheader("Input Pasien Baru")
usia = st.number_input("Usia", 1, 100, 42)
bmi = st.number_input("BMI", 10.0, 60.0, 28.5)
glukosa = st.number_input("Glukosa", 50, 300, 130)
tekanan = st.number_input("Tekanan Darah", 50, 250, 128)
riwayat = st.selectbox("Riwayat Keluarga", [1, 0], format_func=lambda x: "Ada" if x == 1 else "Tidak ada")
k = st.slider("Nilai K", 1, 9, 3, step=2)

if st.button("Prediksi"):
    baru = (usia, bmi, glukosa, tekanan, riwayat)
    jarak = []
    for p in data:
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p[1:6], baru)))
        jarak.append((d, p[0], p[6]))
    jarak.sort()

    hasil = pd.DataFrame(jarak, columns=["Jarak", "Pasien", "Kelas"])
    st.dataframe(hasil.round(4))

    tetangga = [j[2] for j in jarak[:k]]
    prediksi = max(set(tetangga), key=tetangga.count)
    st.success(f"Hasil prediksi: Risiko {prediksi}")
