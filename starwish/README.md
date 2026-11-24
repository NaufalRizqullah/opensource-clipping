# Render klip di CPU Starwish

Agen [opensource-clipping](opensource-clipping/agent.yaml) merender **1–3 rentang waktu pilihan pengguna** dari satu video. Setiap klip mengambil bagian tengah gambar dalam format vertikal 720 × 1280 dan mempertahankan suara sumber jika tersedia.

## Apa yang dikirim dan diterima pelanggan?

Pilih satu sumber yang Anda berhak proses. Waktu mulai dan akhir setiap klip dihitung dari awal video. Video harus mencakup semua rentang yang Anda minta.

| Sumber | Cara mengirim |
| --- | --- |
| MP4 | Lampirkan satu file MP4. Parameter boleh kosong. Ini jalur utama untuk pengujian. |
| YouTube publik | Kirim `{"youtube_url": "https://www.youtube.com/watch?v=kR8bhkVAgBY"}` melalui kolom `params`. Akses bergantung pada YouTube. Jika YouTube meminta login dari IP pusat data, gunakan lampiran MP4. |

Contoh `params` berikut mengambil 45 detik di menit keenam dari video YouTube.

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=kR8bhkVAgBY",
  "ranges": [{"start": "06:00", "end": "06:45"}]
}
```

Isi `ranges` dengan satu sampai tiga pasangan `start` dan `end`. Keduanya menerima angka dalam detik (misalnya `360` dan `405`) atau teks dalam format `MM:SS` dan `HH:MM:SS`, dengan pecahan detik opsional seperti `"06:00.5"`. Nilai awal minimal nol dan nilai akhir harus lebih besar. Jika `ranges` dihilangkan, agen memakai rentang contoh 0,2–59,6 detik.

Untuk YouTube, agen memprioritaskan video H.264 hingga 720p dan audio AAC jika tersedia. Klip pertama mendapat nama `clip.mp4` dan `thumbnail.jpg` (frame pertama). Klip berikutnya memakai `clip_2.mp4`, `thumbnail_2.jpg`, lalu `clip_3.mp4`, `thumbnail_3.jpg`, sesuai urutan permintaan. `metadata.json` memuat daftar `clips` berisi nama file, waktu absolut di sumber, durasi hasil, dimensi, dan keberadaan audio per klip. Field tingkat atas tetap berisi data klip pertama agar permintaan lama dapat dibaca dengan cara yang sama. [Manifest agen](opensource-clipping/agent.yaml)

## Bagaimana Starwish menjalankannya?

1. Pelanggan mengirim permintaan dan sumber video. Starwish menyiapkan dua mesin virtual untuk job tersebut. [Kontrak job](https://starwish.ing/docs/job-contract.md)
2. Di mesin pengendali, [harness Python](opensource-clipping/harness.py) membaca parameter permintaan. Starwish memberikan token internal untuk job itu agar harness dapat memanggil `run_command` di mesin kerja. Pelanggan dan pemilik agen tidak perlu memasukkan API key model.
3. Mesin kerja memiliki 4 vCPU dan disk 20.000 MB menurut spesifikasi Starwish. [Program mesin kerja](opensource-clipping/workspace_job.py) membaca MP4 dari `/job/input/` atau meminta bagian video YouTube yang mencakup rentang pilihan. [Renderer](opensource-clipping/render.py) menjalankan FFmpeg pada CPU, memeriksa durasi dan format dengan ffprobe, lalu memastikan klip dapat didekode sebelum menyerahkan hasil. [Spesifikasi dua mesin](https://starwish.ing/docs/two-machines.md)
4. Starwish memeriksa hasil di `/job/output/` untuk menentukan penyelesaian job. Setelah agen terbit, pelanggan dapat memesannya dengan harga dalam *stars* dan batas waktu yang tercantum pada listing. [Kontrak hasil](https://starwish.ing/docs/job-contract.md)

## Dependensi, akses, dan isi paket

Pengendali memakai pustaka standar Python. Resep di manifest meminta Starwish menyediakan FFmpeg dan ffprobe dari paket Debian `ffmpeg`, serta `yt-dlp[default]==2026.10.26` untuk opsi YouTube. Starwish menjalankan perintah instalasi itu saat membangun mesin kerja. Opsi YouTube memerlukan akses jaringan. Jalur lampiran memproses media yang sudah ada di mesin kerja.

Folder yang diajukan untuk unggah hanya `starwish/opensource-clipping/`. [Daftar file paket](opensource-clipping/.starwishignore) mencakup manifest, instruksi, program Python termasuk pembaca rentang waktu, video uji sintetis, dan [salinan lisensi MIT proyek](opensource-clipping/LICENSE). Paket ini tidak memerlukan Kaggle, GPU, cookies, ataupun kredensial provider AI.

## Urutan uji setelah Anda menyetujui instalasi dan unggah

1. Pasang CLI `starwish` di Linux atau WSL pada Windows. Untuk alur listing, dokumentasi meminta login akun Starwish melalui browser dengan izin `hire,publish,credentials`. Ini otorisasi akun untuk CLI. [Panduan CLI resmi](https://starwish.ing/agent.md)
2. Dari root repositori, periksa manifest dan keberadaan file uji secara lokal.

   ```sh
   starwish doctor ./starwish/opensource-clipping --json
   ```

3. Jalankan satu job pengembangan di CPU Starwish dengan [video sintetis 60 detik](opensource-clipping/fixtures/color-bars.mp4). Perintah ini mengirim paket agen dan input ke Starwish. Putar hasilnya, periksa bahwa pola warna bergerak, lalu cocokkan gambar awal dan durasinya dengan metadata. Ulangi dengan MP4 bersuara untuk memeriksa audio. Opsi URL YouTube dapat dicoba terpisah agar kegagalan akses sumber mudah dibedakan dari kegagalan render.

   ```sh
   starwish dev ./starwish/opensource-clipping "Render the default portrait clip" --file ./starwish/opensource-clipping/fixtures/color-bars.mp4 --json
   ```

   Untuk contoh YouTube di atas, simpan JSON parameter sebagai `params.json` di luar folder paket lalu jalankan satu job dengan `starwish dev ./starwish/opensource-clipping "Render 06:00–06:45" --params @params.json --json`.

   Jika render gagal, jalankan `starwish logs <job_id> --json` untuk melihat tahap yang gagal, kode keluar, dan bagian akhir pesan error FFmpeg.

4. Setelah 2–3 run, gunakan `starwish price` untuk melihat hasil pengukuran. Pemilik menentukan harga dan batas waktu. Keduanya masih `TODO` di manifest. Untuk memeriksa versi tersebut, jalankan `starwish test ./starwish/opensource-clipping --json` setelah kedua nilai terisi. Starwish menilai hasil terhadap kriteria `must` di manifest. [Prosedur pengujian](https://starwish.ing/docs/testing.md)
5. Publikasi memerlukan persetujuan tersendiri. Perintah `starwish publish` menjalankan kembali suite dan memasang versi di marketplace jika semua uji lulus. [Proses publikasi](https://starwish.ing/docs/build-an-agent.md)
