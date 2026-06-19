# Koleksi Musik Latar (BGM)

Folder ini digunakan untuk menyimpan koleksi musik latar (BGM) secara lokal untuk fitur **Auto-BGM & Ducking**.

Karena mengunduh file MP3 berukuran besar secara otomatis dari internet (seperti dari Pixabay) sering gagal karena *rate limit* atau perubahan struktur web, aplikasi ini sekarang membaca langsung dari folder ini.

## Cara Menggunakan

1. Cari file MP3 *royalty-free* (bebas hak cipta).
2. Download dan masukkan file MP3 tersebut ke dalam folder *mood* yang sesuai di dalam direktori ini:
   - `chill/` (Untuk musik santai/lofi)
   - `epic/` (Untuk musik cinematic/action)
   - `sad/` (Untuk musik sedih/emosional)
   - `upbeat/` (Untuk musik semangat/ceria)
   - `suspense/` (Untuk musik misteri/ketegangan)
3. Anda bisa memasukkan lebih dari satu file MP3 ke dalam satu folder. Aplikasi akan **memilih secara acak (random)** dari file-file tersebut setiap kali merender video.
4. Jika sebuah folder *mood* kosong, script akan melakukan fallback ke `chill`. Jika `chill` juga kosong, script akan **melewati proses penambahan BGM** (sama seperti jika Anda memakai flag `--no-bgm`).

---

## Rekomendasi Sumber Download BGM Gratis

Jika Anda bingung mencari lagu, berikut beberapa sumber gratis dan aman untuk *Youtube/Tiktok*:

1. **Pixabay Music** (Sangat disarankan)
   - Chill: [https://pixabay.com/music/search/mood/chill/](https://pixabay.com/music/search/mood/chill/)
   - Epic: [https://pixabay.com/music/search/mood/epic/](https://pixabay.com/music/search/mood/epic/)
   - Upbeat: [https://pixabay.com/music/search/mood/upbeat/](https://pixabay.com/music/search/mood/upbeat/)

2. **Youtube Audio Library** (Aman dari Copyright Strike)
   - Buka YouTube Studio Anda, masuk ke menu **Audio Library**.
   - Filter berdasarkan Mood (Calm, Dramatic, Happy, dll).
   - Pastikan memilih lagu yang "No attribution required" agar lebih aman.

3. **Incompetech** (Karya Kevin MacLeod)
   - [https://incompetech.com/music/royalty-free/music.html](https://incompetech.com/music/royalty-free/music.html)
   - *Catatan: Biasanya memerlukan atribusi (menyebutkan nama di deskripsi).*

## Contoh Struktur Folder yang Benar

```text
assets/
└── bgm/
    ├── chill/
    │   ├── lofi-study-112191.mp3
    │   └── calm-acoustic-2415.mp3
    ├── epic/
    │   └── trailer-music-325357.mp3
    ├── sad/
    ├── suspense/
    └── upbeat/
        └── happy-pop-307007.mp3
```
