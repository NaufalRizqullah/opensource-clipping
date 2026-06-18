"""
generate_youtube_token.py

Tujuan file ini:
- Membuka login Google lewat browser.
- Meminta izin akses YouTube sesuai scope yang dibutuhkan.
- Membuat file .credentials/youtube_token.json.
- File token ini nanti dipakai oleh run_upload.py / uploader.py untuk upload video ke YouTube.

Jalankan file ini hanya saat pertama kali membuat token,
atau saat token lama rusak / ingin login ulang akun YouTube.
"""

import os

# Library ini dipakai untuk menjalankan OAuth flow aplikasi desktop/local.
# Saat dijalankan, browser akan terbuka untuk login Google.
from google_auth_oauthlib.flow import InstalledAppFlow


# Scope = izin akses yang diminta ke akun Google/YouTube.
# youtube.upload  -> izin upload video ke YouTube.
# youtube.readonly -> izin baca data channel/video, misalnya cek jadwal upload yang sudah ada.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


# File OAuth Client ID dari Google Cloud Console.
# File ini didapat dari:
# Google Cloud → APIs & Services → Credentials → OAuth client ID → Desktop app → Download JSON
#
# Rename hasil download menjadi client_secret.json
# lalu simpan di folder .credentials/
CLIENT_SECRET_FILE = ".credentials/client_secret.json"


# File output token hasil login Google.
# File inilah yang nanti dibaca oleh uploader.py.
TOKEN_FILE = ".credentials/youtube_token.json"


def main():
    """
    Fungsi utama untuk generate token YouTube.

    Alurnya:
    1. Cek apakah client_secret.json sudah ada.
    2. Buat folder .credentials jika belum ada.
    3. Buka OAuth login lewat browser.
    4. Setelah user approve, simpan token ke youtube_token.json.
    """

    # Cek apakah file client_secret.json tersedia.
    # Kalau belum ada, script dihentikan karena OAuth flow tidak bisa dimulai.
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise FileNotFoundError(
            f"{CLIENT_SECRET_FILE} tidak ditemukan. "
            "Download OAuth Client ID JSON dari Google Cloud, rename jadi client_secret.json, "
            "lalu simpan ke folder .credentials/"
        )

    # Pastikan folder .credentials tersedia.
    # Kalau belum ada, folder akan dibuat otomatis.
    os.makedirs(".credentials", exist_ok=True)

    # Membuat OAuth flow dari file client_secret.json.
    # Di sini kita kasih daftar scope/izin YouTube yang dibutuhkan.
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=YOUTUBE_SCOPES,
    )

    # Menjalankan login OAuth lewat browser lokal.
    #
    # port=0:
    #   Python akan memilih port kosong otomatis.
    #
    # access_type="offline":
    #   Meminta refresh_token, supaya token bisa diperbarui otomatis
    #   tanpa perlu login ulang setiap access_token expired.
    #
    # prompt="consent":
    #   Memaksa Google menampilkan layar persetujuan lagi.
    #   Ini berguna supaya refresh_token benar-benar keluar.
    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )

    # Simpan hasil credential/token ke file youtube_token.json.
    # File ini berisi access_token, refresh_token, client_id, client_secret, dan scope.
    #
    # PENTING:
    # Jangan upload file youtube_token.json ke GitHub.
    # Anggap file ini seperti password akses YouTube kamu.
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print(f"✅ Token berhasil dibuat: {TOKEN_FILE}")


# Bagian ini membuat main() hanya berjalan kalau file ini dijalankan langsung:
#
# python generate_youtube_token.py
#
# Kalau file ini di-import oleh file Python lain, main() tidak otomatis berjalan.
if __name__ == "__main__":
    main()
