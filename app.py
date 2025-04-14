
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Data
data = {
    "Brand": ["Nike", "Adidas", "Zara", "Uniqlo", "Nike"],
    "Category": ["T-shirt", "Jeans", "Dress", "Hoodie", "T-shirt"],
    "Color": ["Red", "Blue", "Black", "Gray", "White"],
    "Size": ["M", "L", "S", "XL", "L"],
    "Price": [100000, 250000, 300000, 400000, 120000]
}
df = pd.DataFrame(data)

# Preprocessing
df_encoded = pd.get_dummies(df.drop(columns=["Price"]))
scaler = StandardScaler()
df_encoded["Price"] = scaler.fit_transform(df[["Price"]])

# Model
model = NearestNeighbors(n_neighbors=3)
model.fit(df_encoded)

# Streamlit UI
st.title("Sistem Rekomendasi Fashion")

brand = st.selectbox("Pilih Brand", df["Brand"].unique())
category = st.selectbox("Pilih Kategori", df["Category"].unique())
color = st.selectbox("Pilih Warna", df["Color"].unique())
size = st.selectbox("Pilih Ukuran", df["Size"].unique())
price = st.slider("Pilih Harga", 50000, 500000, 150000, step=5000)

# Input user
input_dict = {
    f"Brand_{brand}": 1,
    f"Category_{category}": 1,
    f"Color_{color}": 1,
    f"Size_{size}": 1
}
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=df_encoded.columns, fill_value=0)
input_df["Price"] = scaler.transform([[price]])[0]

# Rekomendasi
distances, indices = model.kneighbors(input_df)
st.write("Rekomendasi Fashion yang Mirip:")
st.dataframe(df.iloc[indices[0]])
