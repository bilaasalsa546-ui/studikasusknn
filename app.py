
import math
import pandas as pd
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Sistem Prediksi Risiko Diabetes")
st.subheader("Menggunakan Algoritma K-Nearest Neighbor (KNN)")

st.write(
    "Aplikasi ini digunakan untuk memprediksi risiko diabetes "
    "berdasarkan data pasien menggunakan metode KNN."
)

st.divider()

# ==========================================
# 2. DATASET PASIEN
# ==========================================

data = [
    ("P01", 25, 21.5, 90, 110, 0, "Rendah"),
    ("P02", 30, 23.1, 95, 115, 0, "Rendah"),
    ("P03", 35, 25.0, 105, 120, 0, "Rendah"),
    ("P04", 40, 27.5, 120, 125, 1, "Tinggi"),
    ("P05", 45, 30.2, 140, 135, 1, "Tinggi"),
    ("P06", 50, 31.5, 155, 140, 1, "Tinggi"),
    ("P07", 55, 33.0, 165, 145, 1, "Tinggi"),
    ("P08", 28, 22.5, 92, 112, 0, "Rendah"),
    ("P09", 38, 26.8, 115, 122, 0, "Rendah"),
    ("P10", 48, 29.5, 145, 130, 1, "Tinggi"),
]

kolom = [
    "Pasien",
    "Usia",
    "BMI",
    "Glukosa",
    "Tekanan Darah",
    "Riwayat Keluarga",
    "Kelas"
]

df = pd.DataFrame(data, columns=kolom)

# ==========================================
# 3. MENAMPILKAN DATASET
# ==========================================

st.header("1. Dataset Pasien")

st.write("Data training yang digunakan dalam proses klasifikasi:")

st.dataframe(df, use_container_width=True)

col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Data", len(df))
col2.metric("Risiko Rendah", len(df[df["Kelas"] == "Rendah"]))
col3.metric("Risiko Tinggi", len(df[df["Kelas"] == "Tinggi"]))

st.divider()

# ==========================================
# 4. INPUT DATA PASIEN BARU
# ==========================================

st.header("2. Input Data Pasien Baru")

st.write("Masukkan data pasien yang akan diprediksi.")

with st.form("form_prediksi"):

    col1, col2 = st.columns(2)

    with col1:
        usia = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=100,
            value=42
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=28.5,
            step=0.1
        )

        glukosa = st.number_input(
            "Glukosa",
            min_value=50,
            max_value=300,
            value=130
        )

    with col2:
        tekanan = st.number_input(
            "Tekanan Darah",
            min_value=50,
            max_value=250,
            value=128
        )

        riwayat = st.selectbox(
            "Riwayat Keluarga",
            [0, 1],
            format_func=lambda x:
            "Tidak Ada" if x == 0 else "Ada"
        )

        k = st.selectbox(
            "Nilai K (Jumlah Tetangga Terdekat)",
            [1, 3, 5, 7, 9],
            index=1
        )

    tombol = st.form_submit_button(
        "🔍 Lakukan Prediksi",
        use_container_width=True
    )

# ==========================================
# 5. PROSES ALGORITMA KNN
# ==========================================

if tombol:

    baru = (usia, bmi, glukosa, tekanan, riwayat)

    jarak = []

    # Menghitung Euclidean Distance
    for p in data:

        d = math.sqrt(
            (p[1] - baru[0]) ** 2 +
            (p[2] - baru[1]) ** 2 +
            (p[3] - baru[2]) ** 2 +
            (p[4] - baru[3]) ** 2 +
            (p[5] - baru[4]) ** 2
        )

        jarak.append({
            "Pasien": p[0],
            "Jarak": d,
            "Kelas": p[6]
        })

    # Mengurutkan jarak dari terkecil
    jarak.sort(key=lambda x: x["Jarak"])

    hasil = pd.DataFrame(jarak)

    # ======================================
    # 6. MENAMPILKAN HASIL PERHITUNGAN JARAK
    # ======================================

    st.divider()

    st.header("3. Hasil Perhitungan Euclidean Distance")

    st.write(
        "Berikut hasil perhitungan jarak antara pasien baru "
        "dengan seluruh data training."
    )

    hasil_tampil = hasil.copy()
    hasil_tampil["Jarak"] = hasil_tampil["Jarak"].round(4)

    st.dataframe(
        hasil_tampil,
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # 7. MENENTUKAN K TETANGGA TERDEKAT
    # ======================================

    tetangga = hasil.head(k).copy()

    st.header("4. Menentukan Tetangga Terdekat")

    st.write(f"Nilai K yang digunakan adalah **{k}**.")

    st.dataframe(
        tetangga.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ======================================
    # 8. PROSES VOTING MAYORITAS
    # ======================================

    voting = tetangga["Kelas"].value_counts()

    rendah = voting.get("Rendah", 0)
    tinggi = voting.get("Tinggi", 0)

    if rendah > tinggi:
        prediksi = "Rendah"
    elif tinggi > rendah:
        prediksi = "Tinggi"
    else:
        # Jika seri, gunakan kelas tetangga
        # dengan jarak paling dekat.
        prediksi = tetangga.iloc[0]["Kelas"]

    # ======================================
    # 9. MENAMPILKAN HASIL PREDIKSI
    # ======================================

    st.divider()

    st.header("5. Hasil Prediksi Risiko Diabetes")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Jumlah Voting Rendah",
            rendah
        )

    with col2:
        st.metric(
            "Jumlah Voting Tinggi",
            tinggi
        )

    total = rendah + tinggi

    persen_rendah = rendah / total * 100
    persen_tinggi = tinggi / total * 100

    st.write("### Persentase Voting")

    st.progress(rendah / total)
    st.write(f"Risiko Rendah: {persen_rendah:.2f}%")

    st.progress(tinggi / total)
    st.write(f"Risiko Tinggi: {persen_tinggi:.2f}%")

    st.write("### Kesimpulan")

    if prediksi == "Rendah":
        st.success(
            "Hasil prediksi: RISIKO RENDAH"
        )
    else:
        st.error(
            "Hasil prediksi: RISIKO TINGGI"
        )

    st.info(
        f"Berdasarkan algoritma KNN dengan nilai K = {k}, "
        f"hasil klasifikasi pasien baru adalah "
        f"Risiko {prediksi}."
    )

# ==========================================
# 10. PENJELASAN METODE
# ==========================================

st.divider()

st.header("6. Penjelasan Algoritma KNN")

st.write("""
Algoritma K-Nearest Neighbor (KNN) merupakan metode klasifikasi
yang menentukan kelas suatu data baru berdasarkan sejumlah
tetangga terdekat dari data training.

Tahapan algoritma KNN:

1. Menentukan data training.
2. Memasukkan data pasien baru.
3. Menghitung jarak menggunakan Euclidean Distance.
4. Mengurutkan jarak dari yang terkecil.
5. Mengambil sejumlah K tetangga terdekat.
6. Melakukan voting berdasarkan kelas mayoritas.
7. Menentukan hasil klasifikasi.

Rumus Euclidean Distance:
""")

st.latex(
    r"d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}"
)

st.caption(
    "Catatan: Aplikasi ini merupakan simulasi pembelajaran "
    "Data Mining, bukan alat diagnosis medis."
)
