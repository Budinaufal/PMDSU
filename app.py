import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "plotly", "pandas", "numpy", "openpyxl"], check=True)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# === Judul Aplikasi ===
st.set_page_config(page_title="Digital Twin Project Performance", layout="wide")
st.title("🛢️ Digital Twin Project Performance Dashboard")
st.markdown("### Optimasi Efisiensi dan Keberlanjutan Operasi Pemboran Migas")

# === Load Data ===
df = pd.read_excel("Hasil_Pengolahan_DigitalTwin.xlsx")

# Sidebar - Filter
st.sidebar.header("Filter Data")
depth_min, depth_max = st.sidebar.slider(
    "Rentang Kedalaman (m)",
    float(df["Depth"].min()), float(df["Depth"].max()),
    (float(df["Depth"].min()), float(df["Depth"].max()))
)
df_filtered = df[(df["Depth"] >= depth_min) & (df["Depth"] <= depth_max)]

# === Grafik Utama ===
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df_filtered, x="Depth", y=["ROP_AVG","Energy_Intensity"],
                   labels={'value':'Normalized Value'},
                   title="Kinerja Pemboran dan Konsumsi Energi")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(df_filtered, x="Depth", y="Sustainability_Index",
                   color_discrete_sequence=['green'],
                   title="Indeks Keberlanjutan Operasi Pemboran")
    st.plotly_chart(fig2, use_container_width=True)

# === Indikator ===
avg_rop = df_filtered["ROP_AVG"].mean()
avg_energy = df_filtered["Energy_Intensity"].mean()
avg_sustain = df_filtered["Sustainability_Index"].mean()

st.markdown("### 📊 Indikator Rata-rata")
col3, col4, col5 = st.columns(3)
col3.metric("ROP (Normalized)", f"{avg_rop:.3f}")
col4.metric("Energy Intensity", f"{avg_energy:.3f}")
col5.metric("Sustainability Index", f"{avg_sustain:.3f}")

# === Scatter Prediksi vs Aktual (dari model AI) ===
st.markdown("### 🔍 Prediksi vs Aktual ROP (Model AI)")
# kamu bisa load prediksi dari model Random Forest / ANN jika sudah disimpan
# contoh:
# st.image("comparison_plot.png")
st.info("Tambahkan grafik perbandingan model AI di sini untuk menunjukkan akurasi prediksi Digital Twin.")
