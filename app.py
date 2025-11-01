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
    page_title="Analisis & Prediksi Teknikal",
    page_icon="🔮",
    layout="wide"
)

# --- Judul Aplikasi ---
st.title("🔮 Aplikasi Analisis Teknikal & Prediksi Tren")
st.caption("Analisis Indikator (SMA, RSI, MACD) dan Prediksi Tren (Regresi Linier)")

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
run_button = st.sidebar.button("Jalankan Analisis & Prediksi")

# --- Area Utama Aplikasi ---
if run_button:
    # 1. Ambil Data
    st.header(f"Analisis & Prediksi untuk: {ticker}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("Gagal mengambil data. Periksa kembali simbol Ticker.")
            st.stop() # Hentikan eksekusi jika data gagal
    except Exception as e:
        st.error(f"Terjadi error saat mengambil data: {e}")
        st.stop()

    df = data.copy()

    # --- BAGIAN MACHINE LEARNING (PREDIKSI TREN) ---
    st.subheader(f"Prediksi Tren untuk {pred_days} Hari ke Depan")

    # 1. Feature Engineering (Membuat fitur X dan y)
    # Kita gunakan 'hari ke-' (angka 0, 1, 2, ...) sebagai fitur (X)
    X = np.arange(len(df)).reshape(-1, 1) 
    y = df['Close'] # Kita gunakan harga 'Close' sebagai target (y)

    # 2. Training Model
    model = LinearRegression()
    model.fit(X, y)

    # 3. Buat data 'X' untuk masa depan (future)
    last_day_index = len(df) - 1
    future_indices = np.arange(last_day_index + 1, last_day_index + 1 + pred_days).reshape(-1, 1)

    # 4. Lakukan Prediksi
    future_predictions = model.predict(future_indices)

    # 5. Siapkan DataFrame untuk plotting
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)
    
    pred_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediksi Tren'])
    
    # Gabungkan DataFrame historis (df) dengan prediksi (pred_df)
    # Ini akan membuat beberapa kolom NaN, tapi tidak masalah untuk plotting
    df_plot = pd.concat([df[['Close']], pred_df])
    
    # Tampilkan Grafik Prediksi
    st.line_chart(df_plot)
    
    st.caption("Catatan: Ini adalah prediksi Regresi Linier, yang hanya menunjukkan tren garis lurus (linear) dan tidak memperhitungkan volatilitas atau peristiwa pasar.")
    # --- SELESAI BAGIAN ML ---
    

    st.write("---") # Pemisah

    # --- BAGIAN ANALISIS TEKNIKAL ---
    st.header("Analisis Indikator Teknikal")

    # 2. Hitung Indikator (Gunakan Strategi pandas-ta)
    MyStrategy = ta.Strategy(
        name="Strategi Analisis",
        ta=[
            {"kind": "sma", "length": 10, "col_names": "SMA_10"},
            {"kind": "sma", "length": 20, "col_names": "SMA_20"},
            {"kind": "rsi", "length": 14, "col_names": "RSI_14"},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9, "col_names": ("MACD", "MACD_hist", "MACD_signal")}
        ]
    )
    df.ta.strategy(MyStrategy)
    
    # 3. Buat Sinyal (Logika Kombinasi kita sebelumnya)
    df['Sinyal_Kombinasi'] = np.where(
        (df['SMA_10'] > df['SMA_20']) & (df['RSI_14'] < 45), 'BELI (Buy the Dip)',
        np.where(
            (df['SMA_10'] < df['SMA_20']) & (df['RSI_14'] > 55), 'JUAL (Sell the Rally)',
            'TAHAN' 
        )
    )

    # 4. Tampilkan Sinyal Terkini
    st.subheader("Sinyal Terkini")
    sinyal_terkini = df['Sinyal_Kombinasi'].iloc[-1]
    
    if sinyal_terkini.startswith('BELI'):
        st.success(f"**{sinyal_terkini}**")
    elif sinyal_terkini.startswith('JUAL'):
        st.error(f"**{sinyal_terkini}**")
    else:
        st.info(f"**{sinyal_terkini}**")
    
    st.caption("Sinyal berdasarkan strategi konfirmasi SMA (10, 20) dan RSI (14).")

    # 5. Tampilkan Grafik Indikator
    
    # Chart 1: Harga dan SMA
    st.subheader("Grafik Harga dan Moving Averages")
    st.line_chart(df[['Close', 'SMA_10', 'SMA_20']])

    # Chart 2: RSI
    st.subheader("Grafik RSI (Relative Strength Index)")
    rsi_data = df[['RSI_14']].copy()
    rsi_data['Overbought (70)'] = 70
    rsi_data['Oversold (30)'] = 30
    st.line_chart(rsi_data)

    # Chart 3: MACD
    st.subheader("Grafik MACD (Moving Average Convergence Divergence)")
    st.line_chart(df[['MACD', 'MACD_signal']])
    st.bar_chart(df[['MACD_hist']])
    
    # 6. Tampilkan Data Mentah
    st.subheader(f"Data Lengkap untuk {ticker} (10 Hari Terakhir)")
    st.dataframe(df.tail(10))

else:
    st.info("Silakan masukkan Ticker di sidebar dan klik 'Jalankan Analisis & Prediksi'.")