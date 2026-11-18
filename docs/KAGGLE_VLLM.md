# vLLM + Gemma 4 Q4 di Kaggle

Gunakan [notebook Kaggle vLLM](../notebooks/kaggle-vllm-gemma4.ipynb) untuk menjalankan analisis transkrip dengan server milik sendiri. Log dari sesi Kaggle T4 menunjukkan model berhasil dimuat, lalu inference pertama gagal karena batas shared memory Triton. **Workaround T4 di bawah masih menunggu validasi inference di Kaggle**. Tidak ada instalasi, test, atau pemuatan model yang dilakukan di PC lokal selama pengerjaan.

## Preset awal

| Komponen | Nilai |
| --- | --- |
| Runtime | Linux x86_64, Internet aktif; kernel Python 3.13 boleh, bootstrap menyiapkan Python 3.12 terpisah |
| GPU | Satu GPU kelas 16 GB, compute capability SM 7.5 atau lebih baru |
| Model | `google/gemma-4-E2B-it-qat-w4a16-ct` |
| Revision model/tokenizer | `971342c08f607aa7779983f6b5289778b5d271a7` |
| Server | vLLM 0.32.0, wheel CUDA 12.9; Transformers 5.16.1 |
| Quantization | QAT W4A16 dari checkpoint: bobot INT4, activations FP16 |
| Context / output | 16.384 token total / maksimal 4.096 token output |
| Serving | Satu GPU, satu sequence, eager, text-only, thinking dimatikan |
| Kompatibilitas T4 | Guard tile attention 16 untuk head besar FP16 pada CUDA dengan shared memory <96 KiB; diterapkan saat Start pada SM 7.5 |
| Memori GPU | `--gpu-memory-utilization 0.85` |
| Endpoint | `http://127.0.0.1:8001`; Studio tetap di port 8000 |

T4 memenuhi batas arsitektur SM 7.5, tetapi Gemma 4 pada vLLM 0.32.0 memerlukan workaround attention di bawah. P100 (SM 6.0) tidak didukung preset ini meskipun memiliki VRAM 16 GB. Dua T4 masing-masing 16 GB tidak otomatis menyediakan satu pool 32 GB; helper memakai indeks 0 secara default. RAM sistem juga berbeda dari VRAM. Ukuran bobot belum mencakup KV cache, workspace, dan biaya runtime, sehingga **kapasitas 16 GB belum menjamin model muat atau menghasilkan klip berkualitas**.

Pilihan model mengikuti [resep resmi Gemma 4 untuk vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) dan [checkpoint Google QAT](https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct). Quantization dibaca otomatis dari checkpoint; tidak perlu bitsandbytes atau konversi model. Ketergantungan server dipisahkan karena aplikasi menggunakan keluarga Torch/NumPy/MediaPipe berbeda. Detail wheel dan checksum ada di [requirements server](../requirements-vllm-kaggle.txt); driver sesi Kaggle tetap harus cocok dengan CUDA 12.9.

## Urutan menjalankan

1. Unggah/buka notebook di Kaggle. Pilih GPU dan aktifkan Internet. Pastikan `REPO_REF` menunjuk branch/commit yang sudah memuat perubahan ini. Checkout lama tidak dihapus, ditarik ulang, atau di-reset otomatis.
2. Jalankan preflight GPU sebelum setup. Jika mendapat P100 atau CPU saja, ganti accelerator yang sesuai; helper ini tidak memasang backend CPU.
   Setelah preflight, jalankan **Prepare Python 3.12**. Notebook memakai interpreter 3.12 yang tersedia atau mengunduhnya ke `.cache/kaggle-python` memakai uv 0.13.9 dengan SHA-256 terverifikasi. Tidak ada paket yang dipasang ke kernel, symlink Python sistem yang diganti, atau environment lama yang dihapus. Semua setup berikutnya memakai `BOOTSTRAP_PYTHON`, bukan `sys.executable` kernel. Python dikelola menggunakan [mekanisme instalasi uv](https://docs.astral.sh/uv/guides/install-python/).
3. `HF_TOKEN` dapat dipasang di Kaggle Secrets untuk download terautentikasi, dengan akses notebook diaktifkan. Checkpoint publik tidak mewajibkan token; model alternatif yang gated memerlukan akses akun yang sesuai. `NGROK_AUTHTOKEN` hanya untuk Studio, `PEXELS_API_KEY` opsional untuk B-roll. Provider vLLM tidak memerlukan Google/NVIDIA API key.
4. Jalankan setup dua lingkungan: `.cache/kaggle-studio/venv` untuk aplikasi, `.cache/kaggle-vllm/venv` untuk server. Jangan memasang `requirements-vllm-kaggle.txt` ke kernel notebook atau lingkungan aplikasi. Instalasi dan `pip check` yang gagal menghentikan setup.
5. Jalankan Start vLLM. Pada T4, helper memeriksa/menerapkan workaround attention dalam environment server sebelum memuat model, termasuk pada environment yang sudah terpasang. Unduhan pertama dapat lama; helper menunggu model yang tepat muncul pada `/v1/models`. Respons ini belum membuktikan inference berhasil; lanjutkan ke eval.
6. Jalankan eval fixture pendek, satu klip. Hasil disimpan ke `/kaggle/working/vllm-eval-short.json`. Evaluasi ini tidak memuat Whisper atau merender video.
7. Opsional: jalankan regression tests, lalu smoke test nyata melalui switch di notebook. Setelah itu jalankan Studio dan uji video pendek untuk menilai transkripsi, hasil kurasi, serta rendering.
8. Jalankan Stop vLLM setelah selesai agar VRAM dibebaskan. Simpan hasil/log dari Kaggle Files/Output sebelum sesi berakhir.

Jangan mengganti `VLLM_API_KEY` ketika server masih berjalan. Notebook membuat key acak di memori bila belum ada, meneruskannya melalui environment, dan tidak mencetak nilainya. Restart kernel kehilangan key tersebut; jalankan Stop lalu konfigurasi dan Start kembali. Helper memeriksa identitas proses dan hanya menghentikan server yang dikelolanya sendiri.

## Penggunaan CLI di lingkungan Kaggle

Perintah berikut dijalankan dari root checkout di terminal Kaggle, setelah setup dan konfigurasi secrets. Notebook menyediakan bentuk subprocess dengan environment terisolasi yang sama. `preflight`, `status`, dan `stop` hanya memerlukan interpreter kernel; `setup` harus memakai Python 3.11/3.12. Di notebook gunakan `BOOTSTRAP_PYTHON` hasil Prepare Python untuk setup/start seperti pada sel yang disediakan.

```sh
python scripts/kaggle_vllm.py preflight
# Untuk setup/start gunakan path BOOTSTRAP_PYTHON dari notebook, misalnya:
/path/to/python3.12 scripts/kaggle_vllm.py setup
/path/to/python3.12 scripts/kaggle_vllm.py start
python scripts/kaggle_vllm.py status
python scripts/kaggle_vllm.py stop
```

Contoh analisis pipeline dengan GPU yang dipakai vLLM:

```sh
.cache/kaggle-studio/venv/bin/python main.py \
  --url "VIDEO_URL" --ai-provider vllm --clips 1 \
  --whisper-model small --whisper-device cpu --whisper-compute-type int8
```

Gunakan URL video milik sendiri atau sumber yang boleh diproses. Variabel `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`, `VLLM_MAX_TOKENS`, `VLLM_CONTEXT_LIMIT`, dan `VLLM_TIMEOUT` harus tersedia pada proses aplikasi jika diubah dari preset. `VLLM_CONTEXT_LIMIT` hanya membatasi request klien; nilai itu tidak memperbesar context server. Mengubah context server memerlukan Stop lalu Start dengan `--context`.

Pada Studio, pilih **vLLM**, model preset, satu klip, **Whisper Device: CPU**, dan model `small` sebagai titik awal. Untuk kombinasi vLLM + CPU, UI memilih compute type `int8`. vLLM mempertahankan alokasi VRAM selama server hidup, sehingga Whisper CUDA/diarization GPU bersamaan dapat menyebabkan OOM. Jika ingin memakai Whisper CUDA, lakukan di sesi/proses terpisah dengan pengelolaan memori yang terukur; preset ini tidak otomatis melepas atau memindahkan model antar tahap.

Analisis vLLM tidak fallback ke cloud. Jika voiceover CLI diaktifkan, pembuatan naskah mengikuti vLLM; TTS edge-tts dan B-roll Pexels tetap membutuhkan Internet. Fitur voiceover pada worker web tetap mengikuti dukungan pipeline yang sudah ada. Cache analisis tetap memakai nama `gemini_response.json` demi kompatibilitas; nama itu tidak menunjukkan provider aktual. Metadata provider tersimpan terpisah di `ai_run_meta.json` untuk job baru.

## Error yang perlu diperiksa

| Gejala | Tindakan |
| --- | --- |
| `Use Python 3.11 or 3.12` pada preflight, traceback `/usr/lib/python3.13` | Checkout/notebook lama menolak versi kernel sebelum membaca GPU. Ambil perbaikan branch yang dipakai, buka notebook terbaru, lalu ulangi preflight dan Prepare Python. Kernel tetap 3.13; setup memakai Python 3.12 terpisah. Jangan sekadar menghapus batas Python setup karena dependency aplikasi masih dibatasi. |
| Download Python/uv gagal | Pastikan Internet Kaggle aktif dan GitHub dapat diakses, lalu ulangi Prepare Python. Setup dependency belum dimulai. Jangan mengganti Python sistem atau memasang paket vLLM ke kernel. |
| P100 / compute capability di bawah 7.5 | Pilih T4 atau GPU yang lebih baru; jangan memaksakan wheel ini pada P100. |
| CUDA/driver atau kernel tidak kompatibel | Periksa `nvidia-smi` dan `logs/kaggle-vllm.log`. Setup yang sukses belum membuktikan kompatibilitas inference. Gunakan runtime dengan driver yang cocok; jangan mengubah paket Torch aplikasi untuk memperbaiki server. |
| `OutOfResources: shared memory, Required: 98304, Hardware limit: 65536` | Gunakan workaround T4 di bawah, lalu Stop → Start → Eval. Ini batas shared memory per blok kernel; mengurangi context atau menambah GPU tidak mengubah batas tersebut. |
| `vLLM tile selector differs` / versi workaround tidak cocok | Helper berhenti sebelum memuat model dan tidak menimpa sumber yang tidak dikenali. Periksa versi environment server dan patch lain; jangan menghapus guard agar patch dipaksakan pada versi berbeda. |
| 401/403 dari Hugging Face | Pastikan akun menerima ketentuan model dan `HF_TOKEN` memiliki akses; ulangi konfigurasi secrets. |
| 401 dari server vLLM | Aplikasi dan server harus memakai key yang sama. Setelah perubahan key/kernel restart, Stop lalu Start dengan konfigurasi baru. |
| OOM saat startup | Hentikan proses GPU lain. Coba context 8192, tetap satu sequence/eager. Input + output harus muat; mengurangi output juga mungkin diperlukan. RAM sistem atau disk yang habis memerlukan pengurangan beban/sesi yang lebih besar. |
| Context overflow | Provider menghitung token prompt berformat chat dan menambahkan jatah output. Kurangi panjang transkrip/jumlah klip/output, atau tingkatkan context server hanya jika memori memadai. Chunking belum otomatis. |
| Output terpotong / JSON atau timestamp tidak valid | Kurangi jumlah klip, tinjau report eval, atau sesuaikan budget output dan context bersama-sama. Perbaikan output dibatasi; hasil tidak diam-diam dipotong/clamp menjadi klip lain. |
| Port 8001 sibuk / setting server berubah | Jalankan `stop` untuk server yang dikelola helper. Proses lain tidak diadopsi atau dimatikan otomatis; pilih port lain dan perbarui `VLLM_BASE_URL`. |
| Studio siap tetapi job gagal | Periksa log tahap job; readiness Studio tidak menguji inference vLLM. Jalankan fixture eval terlebih dahulu. |

### T4: shared memory Triton 96 KiB melebihi batas 64 KiB

Pada log sesi Kaggle yang dilaporkan, `/v1/models` dan `/tokenize` berhasil, kemudian kernel `triton_unified_attention.py` gagal dengan `Required: 98304, Hardware limit: 65536`. `EngineDeadError`, HTTP 500 dan koneksi yang terputus adalah akibat engine berhenti. Bobot sudah dimuat (sekitar 7,12 GiB); exception ini mengacu pada shared memory kernel, bukan total VRAM 16 GB. Dua T4 juga tidak menggabungkan shared memory per blok.

[PR vLLM #39018](https://github.com/vllm-project/vllm/pull/39018) mengusulkan tile lebih kecil untuk attention head besar pada GPU dengan shared memory terbatas. Helper `scripts/kaggle_vllm_compat.py` mengadaptasi guard tersebut untuk preset ini: tile 16 pada CUDA dengan padded head dimension ≥512, elemen minimal 2 byte, dan shared memory <96 KiB. Ukuran head kecil, syarat tile FP8, dan perangkat dengan shared memory cukup mempertahankan perilaku sebelumnya. Ini workaround terbatas, bukan klaim bahwa PR sudah masuk rilis vLLM 0.32.0; performa dan hasilnya tetap perlu dievaluasi.

Patcher hanya dijalankan oleh Start saat GPU terpilih SM 7.5, menggunakan Python environment **server**. Ia memeriksa versi `0.32.0`/`0.32.0+cu129`, lokasi paket, serta fungsi selector yang persis dikenal; sumber berbeda ditolak. Sumber asli disimpan di sebelah file paket sebagai `triton_unified_attention.py.osc-t4-original`, kemudian perubahan ditulis atomik. Rerun mengenali patch yang sudah lengkap. Tidak ada perubahan pada model, environment Studio, Python kernel, atau driver.

Untuk sesi yang sudah terpasang:

1. Simpan log kegagalan. Pastikan perbaikan sudah di-commit/push ke branch yang dipakai Kaggle, kemudian perbarui checkout Kaggle dengan `git pull --ff-only` pada branch itu. Sel checkout notebook tidak menarik pembaruan otomatis.
2. Jalankan **Stop vLLM**, lalu **Start vLLM**. Tidak perlu instal ulang dependency atau mengunduh ulang model. Pastikan output Start memuat `T4 attention workaround applied` atau `already applied`.
3. Jalankan **Eval transkrip pendek** dan periksa `success` dalam JSON. Jika gagal lagi, notebook terbaru menampilkan bagian akhir log server; simpan report dan log terbaru. Restart kernel hanya diperlukan jika runtime memang bermasalah, bukan untuk memasang patch ini.

Notebook yang sudah terbuka tidak berubah ketika checkout diperbarui. Sel Start lama tetap memanggil helper terbaru; buka notebook terbaru untuk memperoleh penampilan log otomatis pada sel Eval. Jangan mengganti model atau memaksa tensor parallel dua GPU untuk mengatasi error shared memory ini.

### Server tidak terjangkau setelah eval dimulai

Periksa `run` pada `vllm-eval-short.json`. Jika `prompt_tokens` sudah tercatat dan `attempts` lebih dari nol, server telah merespons probe/tokenisasi dan klien sudah mencoba inference. Pesan koneksi terakhir saja tidak membuktikan base URL salah atau server belum pernah berjalan. Server dapat berhenti setelah request pertama; penyebabnya perlu dilihat pada log server. `run.failures` menyimpan tahap/status HTTP setiap kegagalan agar error inference pertama tidak tertutup oleh kegagalan koneksi saat retry; `run.error_stage` menunjukkan tahap terakhir yang gagal. Report dari versi lama belum memiliki kedua field ini.

Sebelum me-restart server, jalankan sel diagnostik ini **di Kaggle**:

```python
from pathlib import Path
import subprocess
import sys

project = Path('/kaggle/working/opensource-clipping')
subprocess.run([sys.executable, str(project / 'scripts/kaggle_vllm.py'), 'status'], check=False)
log = project / 'logs/kaggle-vllm.log'
print('\n'.join(log.read_text(errors='replace').splitlines()[-120:])
      if log.is_file() else f'Log tidak ditemukan: {log}')
```

Gunakan traceback tersebut untuk membedakan error CUDA/kernel, alokasi memori, atau proses yang dihentikan. Jumlah token yang masih muat dalam context tidak menjamin penggunaan VRAM/RAM muat; sebaliknya, pesan koneksi saja belum membuktikan OOM. Simpan log sebelum mengubah model, dtype atau budget memori.

Server hanya bind ke loopback. Tunnel notebook membuka Studio, bukan endpoint vLLM. Base URL dan API key vLLM adalah konfigurasi operator; job browser tidak boleh menentukan endpoint arbitrer. Lihat [panduan Studio](KAGGLE_STUDIO.md) untuk ngrok, readiness, dan log aplikasi.

`logs/kaggle-vllm.log` berisi output pihak ketiga dari startup/inference server; request/output logging dimatikan oleh helper. File ini berbeda dari structured log aplikasi yang menyaring metadata. Tinjau isinya sebelum dibagikan. Helper tidak menghapus model cache, log, atau hasil eval ketika Stop.

## Validasi yang masih diperlukan

- Inference dengan workaround T4 pada GPU/driver Kaggle aktual; log sebelum patch baru membuktikan model termuat dan endpoint tokenisasi merespons.
- Schema JSON lengkap, timestamp, jumlah klip, kualitas bahasa Indonesia/Inggris, dan pemilihan momen pada transkrip pendek, sedang, dan panjang.
- Latensi dan penggunaan VRAM/RAM selama analisis, lalu pipeline video dengan Whisper CPU dan rendering.
- Regression suite cloud dan web; test yang ditambahkan belum dijalankan pada PC lokal sesuai permintaan pengguna.

Preset ini adalah titik awal pembelajaran vLLM. Tidak ada angka benchmark atau klaim kualitas/kapasitas yang dianggap terbukti sebelum hasil Kaggle tersedia.
