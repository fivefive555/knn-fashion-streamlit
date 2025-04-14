import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Judul dan deskripsi halaman
st.title("Sistem Rekomendasi Fashion")
st.markdown("Temukan fashion item yang mirip berdasarkan preferensi Anda.")

# Load dataset dan model
df = pd.read_csv("fashion_dataset.csv")
model = joblib.load("knn_model.pkl")

# Persiapan fitur (pastikan ini sesuai dengan model kamu)
features = ['Brand', 'Category', 'Color', 'Size', 'Price']
df_encoded = pd.get_dummies(df[features])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Sidebar untuk input
st.sidebar.header("Masukkan Preferensi Anda")
brand = st.sidebar.selectbox("Brand Favorit", df['Brand'].unique())
kategori = st.sidebar.selectbox("Kategori Produk", df['Category'].unique())
warna = st.sidebar.selectbox("Warna Kesukaan", df['Color'].unique())
ukuran = st.sidebar.selectbox("Ukuran", df['Size'].unique())
harga = st.sidebar.slider("Budget Maksimal (Rp)", 50000, 500000, 250000, step=5000)

# Encode input
input_df = pd.DataFrame([[brand, kategori, warna, ukuran, harga]], columns=features)
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# Rekomendasi
model_knn = NearestNeighbors(n_neighbors=3)
model_knn.fit(df_scaled)
distances, indices = model_knn.kneighbors(input_scaled)

# Hasil
st.subheader("Rekomendasi Fashion yang Mirip:")
rekomendasi = df.iloc[indices[0]]
st.table(rekomendasi)

st.caption("Tip: Ubah preferensi di sidebar untuk mendapatkan rekomendasi yang berbeda.")
