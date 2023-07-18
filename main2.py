# %%
from data.pemrosesanData import ambilData, labelEncode, MinMax, cetak_kolom_kategorikal
from algoritma.KNN import K, splitDataset, jarakTerdekat, hasilJarakTerdekat, acakK, Klasifikasi, semuaPrediksi
from pengujian.metrik_evaluasi import Akurasi, Presisi, Recall, F1, CM, Metrik, visualisaiMetrik, maksMetrik, visualisasiCM
import numpy as np
import pandas as pd

# %%
dataset = ambilData() # Ambil dataset
print("\t\tHasil Import Dataset")
print(dataset) # menampilkan dataset

# %%
print("\t\tInformasi Kolom dataset")
dataset.info() #informasi kolom dataset

# %%
print("\t\tInformasi Kolom kategorikal")
cetak_kolom_kategorikal(dataset) # menampilkan informasi kolom kategorikal (object)

# %%
dataset = labelEncode(dataset) # encoding nilai kategorikal menjadi numerikal
print ("\t\tDataset Yang Telah Diencoding")
print (dataset) # menampilkan dataset yang telah diencoding

# %%
dataset = MinMax(dataset) # Normalisasi dataset
print ("\t\tDataset Yang Telah dinormalisasi")
print (dataset) # menampilkan dataset yang telah dinormalisasi

# %%
acak_K = acakK()
maks_K = np.max(acak_K)
print("nilai k yang dihasilkan :", acak_K)

# %%
kolom_target = 'HeartDisease' #membuang kolom HeartDisease , digunakan sebagai target
X_train, X_test, y_train, y_test = splitDataset(dataset, kolom_target, test_size=0.2, random_state=42)

# %%
print("Data Training\n") 
print(X_train) # menampilkan data training

# %%
print("Data Testing\n") 
print(X_test) # menampilkan data testing

# %%
#mencari ketetanggan terdekat
terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)

# %%
#mencari ketetanggan terdekat
terdekat, indeks = jarakTerdekat(X_test, X_train, y_train, maks_K)
# Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
y_pred = semuaPrediksi(acak_K, X_train, y_train, X_test)
# mencari jarak terdekat
hasil_jarak_terdekat = hasilJarakTerdekat(X_test, y_train, terdekat, indeks)    
#konversi data frame
df_hasil_jarak_terdekat = pd.DataFrame(hasil_jarak_terdekat, columns=["Testing", "Training", "Jarak", "Kelas"])
print(df_hasil_jarak_terdekat)

# %%
kolom_prediksi = ["K" + str(val) for val in acak_K] # Membuat dictionary kolom
prediksi = pd.DataFrame(np.transpose(y_pred), columns=kolom_prediksi) # Membuat dataframe
klasifikasi = Klasifikasi(X_test, prediksi, y_test) # Hasil klasifikasi
print(klasifikasi)

# %%
akurasi, presisi, recall, f1, cm = Metrik(y_test, prediksi)

# %%
visualisasiCM(cm, acak_K)

# %%


# %%
metrik = pd.DataFrame({'k': acak_K,'akurasi': akurasi, 'presisi': presisi, 'recall': recall, 'f1': f1})
print(metrik)

# %%
maksMetrik(metrik)

# %%
visualisaiMetrik(metrik)


