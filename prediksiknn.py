import math

# DATASET: (pasien, usia, bmi, glukosa, tekanan darah, riwayat keluarga, kelas)
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

# Data pasien baru: (usia, bmi, glukosa, tekanan darah, riwayat keluarga)
pasien_baru = (42, 28.5, 130, 128, 1)
K = 3

# Rumus jarak Euclidean
def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# Hitung jarak pasien baru ke setiap pasien
jarak = []
for p in data:
    nama, fitur, kelas = p[0], p[1:6], p[6]
    jarak.append((euclidean(fitur, pasien_baru), nama, kelas))

# Urutkan dari jarak terkecil
jarak.sort()

print("Peringkat  Pasien  Kelas    Jarak")
for i, (d, nama, kelas) in enumerate(jarak, start=1):
    print(f"{i:<10} {nama:<7} {kelas:<8} {d:.4f}")

# Ambil K tetangga terdekat lalu voting
tetangga = jarak[:K]
voting = {"Rendah": 0, "Tinggi": 0}
for d, nama, kelas in tetangga:
    voting[kelas] += 1

hasil = "Tinggi" if voting["Tinggi"] > voting["Rendah"] else "Rendah"

print(f"\n{K} tetangga terdekat:", ", ".join(n for d, n, k in tetangga))
print("Voting:", voting)
print("Hasil prediksi: Risiko", hasil)