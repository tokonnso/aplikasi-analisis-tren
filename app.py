import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import datetime

# Import library untuk Machine Learning (Regresi Linier)
from sklearn.linear_model import LinearRegression

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis & Prediksi Teknikal (Lanjutan)",
    page_icon="🚀",
    layout="wide"
)

# --- Judul Aplikasi ---
st.title("🚀 Aplikasi Analisis Teknikal Lanjutan")
st.caption("Analisis (SMA, RSI, MACD, BBANDS, STOCH) & Prediksi Tren (Regresi Linier)")

# --- Sidebar (Input dari User) ---
st.sidebar.header("Input Parameter")

# Input Ticker
ticker = st.sidebar.text_input("Simbol Ticker", "BBCA.JK")
st.sidebar.caption("Contoh: BBCA.JK (Jakarta), BTC-USD (Crypto), AAPL (US)")

# Input Tanggal
today = datetime.date.today()
start_date = st.sidebar.date_input('Tanggal Mulai', today - datetime.timedelta(days=365*2))
end_date = st.sidebar.date_input('Tanggal Selesai', today)

# Input untuk berapa hari prediksi
pred_days = st.sidebar.slider("Hari Prediksi ke Depan", 1, 90, 30)

# Tombol untuk Menjalankan
run_button = st.sidebar.button("Jalankan Analisis Lengkap")

# --- Area Utama Aplikasi ---
if run_button:
    # 1. Ambil Data
    st.header(f"Analisis & Prediksi untuk: {ticker}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("Gagal mengambil data. Periksa kembali simbol Ticker.")
            st.stop() 
    except Exception as e:
        st.error(f"Terjadi error saat mengambil data: {e}")
        st.stop()

    df = data.copy()

    # --- PERBAIKAN UNTUK MultiIndex dan KeyError ---
    # 1. Ratakan (flatten) MultiIndex jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 2. BARU: Ubah semua nama kolom jadi lowercase agar pandas-ta bisa baca
    df.columns = df.columns.str.lower()
    # Sekarang kita punya kolom 'open', 'high', 'low', 'close', 'volume'
    # -----------------------------------------------------------------


    # --- BAGIAN MACHINE LEARNING (PREDIKSI TREN) ---
    st.subheader(f"Prediksi Tren untuk {pred_days} Hari ke Depan")

    X = np.arange(len(df)).reshape(-1, 1) 
    # BARU: Gunakan 'close' (huruf kecil)
    y = df['close'] 
    model = LinearRegression()
    model.fit(X, y)

    last_day_index = len(df) - 1
    future_indices = np.arange(last_day_index + 1, last_day_index + 1 + pred_days).reshape(-1, 1)
    future_predictions = model.predict(future_indices)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)
    pred_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediksi Tren'])
    
    # BARU: Gunakan 'close' (huruf kecil)
    df_plot = pd.concat([df[['close']], pred_df])
    
    st.line_chart(df_plot)
    st.caption("Catatan: Ini adalah prediksi Regresi Linier, yang hanya menunjukkan tren garis lurus.")
    
    st.write("---") # Pemisah

    # --- BAGIAN ANALISIS TEKNIKAL ---
    st.header("Analisis Indikator Teknikal")

    # 2. Hitung Indikator
    # (Sekarang akan berhasil karena ada 'high', 'low', 'close')
    df.ta.sma(length=10, col_names="SMA_10", append=True)
    df.ta.sma(length=20, col_names="SMA_20", append=True)
    df.ta.rsi(length=14, col_names="RSI_14", append=True)
    df.ta.macd(fast=12, slow=26, signal=9, col_names=("MACD", "MACD_hist", "MACD_signal"), append=True)
    df.ta.bbands(length=20, col_names=("BB_Lower", "BB_Mid", "BB_Upper", "BB_Bandwidth", "BB_Percent"), append=True)
    df.ta.stoch(k=14, d=3, col_names=("STOCH_K", "STOCH_D"), append=True)
    
    # 3. Buat Sinyal (Logika Kombinasi)
    df['Sinyal_SMA_RSI'] = np.where(
        (df['SMA_10'] > df['SMA_20']) & (df['RSI_14'] < 45), 'BELI (Buy the Dip)',
        np.where(
            (df['SMA_10'] < df['SMA_20']) & (df['RSI_14'] > 55), 'JUAL (Sell the Rally)',
            'TAHAN' 
        )
    )

    # 4. Tampilkan Sinyal Terkini
    st.subheader("Sinyal Terkini (Strategi SMA/RSI)")
    sinyal_terkini = df['Sinyal_SMA_RSI'].iloc[-1]
    
    if sinyal_terkini.startswith('BELI'):
        st.success(f"**{sinyal_terkini}**")
    elif sinyal_terkini.startswith('JUAL'):
        st.error(f"**{sinyal_terkini}**")
    else:
        st.info(f"**{sinyal_terkini}**")

    # 5. Tampilkan Grafik Indikator
    
    st.subheader("Grafik Harga, SMA, dan Bollinger Bands")
    # BARU: Gunakan 'close' (huruf kecil)
    st.line_chart(df[['close', 'SMA_10', 'SMA_20', 'BB_Lower', 'BB_Upper']])

    st.subheader("Grafik Indikator Momentum")
    col1, col2 = st.columns(2)

    with col1:
        st.write("RSI (Relative Strength Index)")
        rsi_data = df[['RSI_14']].copy()
        rsi_data['Overbought (70)'] = 70
        rsi_data['Oversold (30)'] = 30
        st.line_chart(rsi_data)

    with col2:
        st.write("Stochastic (Overbought > 80, Oversold < 20)")
        # Baris ini (sekarang sekitar baris 150) seharusnya sudah aman
        stoch_data = df[['STOCH_K', 'STOCH_D']].copy()
        stoch_data['Overbought (80)'] = 80
        stoch_data['Oversold (20)'] = 20
        st.line_chart(stoch_data)

    st.subheader("Grafik MACD")
    st.line_chart(df[['MACD', 'MACD_signal']])
    st.bar_chart(df[['MACD_hist']])
    
    # 6. Tampilkan Data Mentah
    st.subheader(f"Data Lengkap (10 Hari Terakhir)")
    st.dataframe(df.tail(10))

else:
    st.info("Silakan masukkan Ticker di sidebar dan klik 'Jalankan Analisis Lengkap'.")
