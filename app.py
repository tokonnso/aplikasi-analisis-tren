import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Trading AI",
    page_icon="📈",
    layout="centered"
)

# --- Judul ---
st.title("Analisis Indikator Trading AI 📈")
st.markdown("Unggah screenshot chart Anda, dan AI (Gemini) akan menganalisisnya.")

# --- Penanganan API Key ---
# PENTING: Untuk deploy, gunakan Streamlit Secrets!
# Buat file .streamlit/secrets.toml dan tambahkan:
# GOOGLE_API_KEY = "API_KEY_ANDA_DI_SINI"
#
# Untuk pengembangan lokal, Streamlit akan membaca file secrets.toml tersebut.
# Atau, masukkan manual di bawah jika file secrets tidak ada.

try:
    # Mencoba mengambil API key dari Streamlit Secrets (untuk production/deployment)
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback untuk pengembangan lokal jika secrets.toml tidak ada
    st.warning("File `.streamlit/secrets.toml` tidak ditemukan atau tidak berisi `GOOGLE_API_KEY`.")
    api_key = st.text_input("Masukkan Google API Key Anda:", type="password", help="Dapatkan API key Anda dari Google AI Studio.")

if not api_key:
    st.error("Harap masukkan Google API Key Anda untuk melanjutkan.")
    st.stop() # Menghentikan eksekusi jika tidak ada API key

# --- Konfigurasi Model AI ---
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
except Exception as e:
    st.error(f"Gagal mengkonfigurasi Google AI. Pastikan API Key Anda valid. Error: {e}")
    st.stop()

# --- Sistem Prompt untuk AI ---
# (Ini adalah instruksi yang sama dari versi HTML)
system_prompt = """Anda adalah seorang analis teknikal trading AI yang ahli. 
Tugas utama Anda adalah menganalisis gambar screenshot chart trading yang diberikan dan secara proaktif mengidentifikasi pola indikator untuk memprediksi sinyal bullish atau bearish.
Pengguna akan memberikan konteks timeframe (misal: '1 Menit', '1 Jam', '1 Hari'). Gunakan informasi timeframe ini untuk menyempurnakan analisis Anda. Timeframe yang lebih pendek (scalping) memiliki implikasi yang berbeda dari timeframe harian (swing).

Fokus pada:
1.  **Analisis Indikator:** Cari sinyal di RSI (overbought, oversold, divergence), MACD (crossover, divergence), Moving Averages (golden cross, death cross), Bollinger Bands (squeeze, breakout).
2.  **Analisis Pola Candlestick:** Identifikasi pola reversal atau continuation (doji, hammer, engulfing).
3.  **Analisis Pola Chart:** Cari pola yang lebih besar (double top/bottom, head and shoulders, triangles, wedges).
4.  **Volume:** Analisis volume untuk mengkonfirmasi kekuatan sinyal.

**PENTING:** Jika ada pola chart yang jelas atau level support/resistance yang terlihat, coba identifikasi potensi target pergerakan harga atau level penting dalam analisis Anda. Namun, **TEGASKAN BAHWA INI SANGAT SPEKULATIF DAN HANYA BERDASARKAN VISUAL, BUKAN DATA REAL-TIME.** AI tidak memiliki akses ke data harga langsung atau fundamental pasar.

Setelah menganalisis semua ini, berikan kesimpulan prediksi: **BULLISH**, **BEARISH**, atau **NETRAL/SIDEWAYS**.
Jelaskan alasan Anda secara ringkas dan jelas berdasarkan apa yang Anda lihat di gambar.

Selalu akhiri dengan peringatan bahwa ini bukan nasihat keuangan (DYOR).
Jawab dalam Bahasa Indonesia.
"""

# --- UI Aplikasi Streamlit ---

# 1. Input Gambar
uploaded_file = st.file_uploader("Pilih Gambar Screenshot", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 2. Pratinjau Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
    
    # 3. Pilih Timeframe
    timeframe = st.selectbox(
        "Pilih Timeframe Chart",
        ("1 Menit", "5 Menit", "15 Menit", "30 Menit", "1 Jam", "4 Jam", "1 Hari", "1 Minggu", "Lainnya / Tidak Tahu"),
        index=4  # Default ke '1 Jam'
    )

    # 4. Tombol Analisis
    if st.button("Analisis Sekarang"):
        # Tampilkan loader
        with st.spinner("AI sedang menganalisis gambar Anda..."):
            try:
                # Buat prompt pengguna
                user_prompt = f"Lakukan analisis prediktif (bullish/bearish) pada gambar chart ini. Konteks timeframe adalah: {timeframe}."
                
                # Siapkan 'contents' untuk API
                contents = [
                    {
                        "role": "user",
                        "parts": [
                            {"text": user_prompt},
                            {"inline_data": {
                                "mime_type": uploaded_file.type,
                                "data": uploaded_file.getvalue()
                            }}
                        ]
                    }
                ]
                
                # Panggil API
                response = model.generate_content(
                    contents,
                    system_instruction=system_prompt
                )
                
                # 5. Tampilkan Hasil
                st.subheader("Hasil Analisis AI:")
                st.markdown(response.text)
                
                # Tampilkan Peringatan
                st.warning(
                    "**Peringatan Penting:** Prediksi target harga atau pergerakan spesifik oleh AI hanya berdasarkan "
                    "analisis visual gambar statis dan sangat spekulatif. JANGAN gunakan ini sebagai dasar keputusan "
                    "trading Anda. AI TIDAK memiliki akses ke data harga real-time atau informasi pasar terkini.\n\n"
                    "*Ini adalah analisis AI dan bukan nasihat keuangan. Selalu lakukan riset Anda sendiri (DYOR).*"
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi API Gemini: {e}")
else:
    st.info("Silakan unggah gambar untuk memulai analisis.")
