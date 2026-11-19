# Plan: Local LLM Provider — vLLM dahulu (Linux GPU / CPU)

> **Status:** Implementasi awal di workspace, 2026-11-17. Kode, notebook, dokumentasi dan test telah ditulis; validasi runtime Kaggle belum dilakukan.
> **Arahan pengguna:** vLLM terlebih dahulu; target Kaggle satu GPU 16 GB dan Gemma 4 Q4. Jangan menjalankan test, instalasi, server atau model di PC lokal.
> **Scope awal:** Integrasi vLLM sebagai service terpisah. Gemini/NVIDIA NIM tetap tersedia; Ollama ditunda.
> **Dasar review:** Source lokal, README, CHANGELOG sampai v1.15.7, Wiki, test, dan dokumentasi resmi. Codebase-memory gagal dengan `Transport closed`, sehingga verifikasi memakai source/rg.

## Status implementasi 2026-11-17

- Gemini/NIM dipisah ke class provider dengan facade dan titik patch lama; schema cloud dibekukan dalam fixture independen. Dispatcher tetap di `engine.py`; belum ada kebutuhan registry/fallback chain generik.
- vLLM tersedia lewat CLI, worker, React dan Studio statis. Endpoint/key hanya dari operator; job hanya memilih model/alias dan budget output. Analisis dan script voiceover CLI memakai vLLM tanpa fallback cloud.
- Tokenisasi server memeriksa input + output; schema/timestamp divalidasi, retry/repair dibatasi, konteks/output terpotong gagal jelas, cache lama dipertahankan dan metadata run disimpan terpisah.
- Preset eksperimen: `google/gemma-4-E2B-it-qat-w4a16-ct`, revision `971342c08f607aa7779983f6b5289778b5d271a7`, vLLM 0.32.0, GPU tunggal, FP16, context 16384, output 4096. T4 merupakan target kompatibilitas sumber; P100 ditolak. Belum ada klaim model/pipeline muat di 16 GB.
- Notebook/helper Kaggle memisahkan environment server dan aplikasi. Whisper CPU/int8 dipakai untuk percobaan satu GPU. Lihat [panduan Kaggle](../KAGGLE_VLLM.md), [provider](../providers.md), dan [notebook](../../notebooks/kaggle-vllm-gemma4.ipynb).
- Test kontrak/mocked dan live opt-in telah ditulis, **belum dijalankan**, mengikuti larangan test di PC. Eval awal hanya fixture pendek; benchmark kualitas ID/EN, transkrip panjang dan peak RAM/VRAM masih menunggu Kaggle.
- Ditunda: Ollama, generic provider/fallback chain, chunking/staged, endpoint editor/discovery web, preset server CPU, serta perubahan versi dan field cloud web yang tidak terkait.

Bagian berikut mempertahankan dasar review dan urutan acceptance awal. Checkbox fase belum berarti test telah lulus; keputusan deployment Kaggle di atas menggantikan scope server eksternal awal.

## 1. Hasil review dan scope

Plan bisa diterapkan, tetapi urutan dan beberapa asumsi perlu diperbaiki:

- Lakukan inventaris hardware dan spike vLLM **sebelum refactor besar**. Dukungan API tidak membuktikan model muat atau cukup cepat pada server target.
- Pisahkan runtime Linux GPU dan CPU; client aplikasi tetap sama. Jangan memasang server vLLM ke environment clipping.
- Pertahankan facade, schema/payload cloud, retry lama, cache `gemini_response.json`, dan default Gemini.
- Provider lokal tidak fallback ke cloud secara diam-diam.
- Preflight token, validasi hasil, dan retry terbatas wajib sejak provider pertama. Jangan menundanya sampai fase setelah eval.
- Tunda Ollama, provider generik, fallback chain configurable, dan chunking/staged sampai ada kebutuhan terukur.
- Dependency OpenAI merupakan perbaikan tersendiri: memasangnya dapat mengaktifkan NIM yang sebelumnya gagal import, sehingga bukan sekadar refactor tanpa perubahan perilaku.

Milestone validasi pertama: analisis vLLM dari transkrip tersimpan, lalu satu pipeline end-to-end tanpa voiceover. Implementasi voiceover lokal telah ditulis, tetapi tetap memerlukan test Kaggle tersendiri. Tidak ada klaim pipeline sepenuhnya offline karena TTS/B-roll tetap memakai layanan jaringan.

## 2. Temuan kode yang diverifikasi

Nomor baris adalah snapshot review dan dapat bergeser.

| Area | Source | Implikasi |
|---|---|---|
| NIM | `clipping/engine.py:929` | Sudah memakai SDK OpenAI; `nvext.guided_json` khusus NIM |
| Dispatcher | `clipping/engine.py:1135` | NIM tanpa key/gagal beralih ke Gemini |
| Gemini | `clipping/engine.py:1151`, retry 491–588 | Pertahankan API-key Developer API `v1beta`, schema, retry, fallback model |
| Prompt | `clipping/engine.py:596` | Seluruh transkrip dalam satu prompt panjang; 23 field wajib per klip |
| Voiceover | `clipping/voiceover.py:221`, `clipping/runner.py:187` | Script langsung memakai Gemini; belum mengikuti provider analisis |
| Dependency | `requirements.txt`, `pyproject.toml`, `requirements-kaggle.txt` | `openai` belum ada di dua manifest utama, ada pada Kaggle |
| CLI | `main.py:133` | Google key masih diwajibkan; menghalangi vLLM tanpa kredensial cloud |
| Worker | `web/api/worker.py:97` | Pertahankan aturan cache bypass dan validasi NVIDIA/Gemini |
| Metadata | `clipping/metadata.py:104` | Ada normalisasi timestamp, belum ada validasi rentang lengkap |
| Whisper batch | `main.py:167`, `clipping/engine.py:392` | Instance bisa dipertahankan lintas video; urutan fase tidak otomatis membebaskan VRAM |
| Port | `docker-compose.yml:6`, `web/api/app.py:7` | Studio memakai 8000; vLLM same-host perlu port berbeda |
| API model | `web/api/models.py`, `config_adapter.py` | `nvidia_model`/`gemini_fallback_model` dibaca adapter tetapi belum dideklarasikan pada request |

Koreksi klaim lama: NIM tidak mempunyai loop retry tambahan di aplikasi, tetapi SDK dibuat tanpa override `max_retries`. SDK OpenAI mendokumentasikan dua retry default untuk error tertentu. Jangan menambah loop baru yang menggandakan retry atau mengubah default NIM saat ekstraksi. [SDK retries](https://github.com/openai/openai-python#retries).

Changelog relevan: v1.15.6 (routing Gemini), v1.15.1 (worker/cache/readiness), v0.9.13 (fallback NIM). `docs/architecture.md` dan `docs/modules.md` belum ada. Drift versi 1.12.0 vs v1.15.7 dicatat tetapi penertibannya dipisah dari fitur ini.

## 3. Arsitektur dan kompatibilitas

```text
CLI / Studio worker
        |
engine facade + prompt + validasi hasil
        |
registry -> Gemini / NVIDIA NIM / vLLM client
                                    |
                            HTTP private endpoint
                                    |
                           vLLM Linux GPU atau CPU
```

Target paket `clipping/providers/`: `base.py`, `registry.py`, `schema.py`, `validation.py`, `parsing.py`, `retry.py`, `gemini.py`, `openai_compat.py`, `nvidia.py`, `vllm.py`. Buat bertahap; `openai_compat.py` hanya berbagi transport/perilaku yang benar-benar sama. Ollama belum diregistrasikan atau ditampilkan di UI.

Kontrak minimum: `from_cfg(cfg)`, validasi konfigurasi/probe, dan `generate_clips(prompt, schema) -> list[dict]`. Tambahkan `generate_text` pada milestone voiceover. Gunakan konteks internal/collector untuk usage/attempt, bukan mengubah return facade publik menjadi tuple.

Aturan ekstraksi:

- Pertahankan fungsi `analyze_with_ai`, `analyze_with_gemini`, `analyze_with_nvidia`, `get_analysis_prompt`, `TARGET_ACCOUNTS`, dan konstanta retry publik.
- Engine dimuat lewat path oleh test: gunakan absolute lazy import dan verifikasi cara eksekusi test yang didukung.
- Nama facade saja tidak cukup. Prompt, sleep, dan konstanta yang di-monkeypatch harus tetap dipakai pada runtime; jangan hanya menyalin nilai pada import. Gunakan dependency injection bila perlu.
- Wrapper langsung Gemini/NIM menerima cfg minimal tanpa `ai_provider`.
- Canonical schema harus mempunyai adapter yang mempertahankan tipe, enum, `required`, dan `additionalProperties` cloud secara persis; snapshot request menjadi acuan.
- Hindari circular import: schema menerima daftar akun sebagai data atau memakai modul netral.
- Parsing/validasi baru diberlakukan untuk vLLM terlebih dahulu; memperketat output cloud bukan refactor murni.
- Capability thinking/format bergantung pada model, template, dan versi, bukan hanya nama provider.

## 4. Kontrak vLLM dan konfigurasi

vLLM menyediakan `/v1/chat/completions` yang dapat dipakai SDK OpenAI; chat model memerlukan template yang sesuai. Pin versi server/SDK, revisi model, dan template yang diuji. [Server API](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/).

- Model request harus cocok dengan `/v1/models`, termasuk alias `--served-model-name`.
- Gunakan `response_format: {type: "json_schema", json_schema: {name: "clips", schema: ...}}`; verifikasi bentuk lengkap melalui SDK nyata dan server.
- Envelope lokal `{"clips": [...]}` di-unpack menjadi list. Ini pilihan kontrak aplikasi, bukan klaim semua backend menolak root array.
- Prompt khusus vLLM harus konsisten dengan envelope; jangan menyisakan instruksi/contoh array yang bertentangan. Prompt cloud dipertahankan.
- Pilih instruct/non-thinking untuk spike; toggle thinking hanya dikirim jika model/template mendukung.
- `guided_json` lama dihapus sejak v0.12.0. Pakai `response_format` atau `structured_outputs` sesuai versi. Structured decoding tidak menjamin kebenaran timing/isi. [Structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/).

| Konfigurasi usulan | Aturan |
|---|---|
| `--ai-provider` | Default tetap `gemini`; tambah hanya `vllm` |
| `--vllm-model` / `VLLM_MODEL` | Eksplisit sampai preset model lolos eval |
| `--local-base-url` / `VLLM_BASE_URL` | Contoh same-host `http://127.0.0.1:8001`; remote memakai endpoint private operator |
| `VLLM_API_KEY` | Opsional; placeholder non-rahasia eksplisit untuk SDK jika server tanpa auth |
| `--local-max-tokens` / `VLLM_MAX_TOKENS` | Integer positif, eksplisit pada spike; preset setelah eval. Jangan menyalin 16384 NIM sebagai default universal |
| `VLLM_CONTEXT_LIMIT` | Batas client tambahan/limit server terverifikasi; bukan pengubah kapasitas server |
| Timeout lokal | Connect/probe singkat, inference configurable; usulan awal 5/10/900 detik, diuji pada server target |

Precedence CLI eksplisit > env > default terdokumentasi. Base URL menerima root atau satu suffix `/v1`, menangani trailing slash/proxy prefix, dan tidak menghasilkan `/v1/v1`. Simpan root service untuk `/tokenize`. Isi key/base URL eksplisit agar tidak mewarisi default cloud atau `OPENAI_API_KEY` secara tak sengaja.

Kebijakan fallback MVP: Gemini tetap memakai fallback model lama; NIM tetap beralih ke Gemini; vLLM tidak punya fallback cloud. Flag generic `--ai-fallback` ditunda.

Voiceover lokal adalah milestone berikutnya. Setelah tersedia, mode `auto` mengikuti vLLM untuk analisis vLLM; jalur cloud tetap Gemini dengan sampling/routing yang sama. Cache analisis tidak meniadakan kebutuhan provider bila script voiceover masih harus dibuat. Validasi mengikuti fitur yang benar-benar berjalan.

## 5. Konteks, validasi hasil, dan retry

### Budget konteks

```text
prompt_tokens setelah chat template + requested_output_tokens <= effective_context_limit
```

Gunakan `/tokenize` dengan messages, model, dan opsi template yang sama dengan generation. Endpoint berada pada root service, bukan `/v1/tokenize`. Respons memberikan jumlah token dan limit setelah preprocessing template. Jika gateway memblokirnya, gunakan tokenizer/revisi/template identik atau error konfigurasi yang jelas. [Tokenize protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/serve/tokenize/protocol/), [implementation](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/serve/tokenize/serving/).

`max_model_len` pada record `/v1/models` dapat null/hilang; validasi record model dan integer positif. Cocokkan limit metadata/tokenisasi/config; bila client ceiling dan server limit ada, pilih yang lebih kecil. Karakter/3.5 hanya perkiraan, bukan batas aman. `--max-model-len` meliputi input dan output. [ModelCard](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/engine/protocol.py), [serve configuration](https://docs.vllm.ai/en/stable/cli/serve/#--max-model-len).

Input terlalu besar menghasilkan `ContextOverflowError`, tanpa retry identik atau pemotongan diam-diam. Hitung ulang budget pada repair. Chunked/staged belum tersedia pada MVP.

### Validasi hasil sebelum cache/render

- Periksa content, JSON, envelope, schema, dan finish reason; `length` tidak dianggap sukses walau sebagian JSON bisa diparse.
- Reuse metadata normalization untuk alias/enrichment; validator lokal tetap terpisah agar perilaku cloud tidak berubah.
- Periksa angka finite, `0 <= start < end <= batas_sumber`, durasi menurut `MIN/MAX_CLIP_DURATION`, jumlah klip, dan overlap.
- Periksa juga `hook_start_time`, `hook_end_time`, `hook_v2.items`, `keep_segments`, dan timing B-roll sesuai kontraknya. Interval nested harus berada dalam klip, terurut, dan valid.
- Gunakan durasi sumber/segmen bila tersedia. Jika facade hanya menerima transkrip, parse seluruh baris tervalidasi sebagai batas konservatif; jangan hanya mengambil baris terakhir.
- Jangan clamp atau memperpanjang durasi otomatis agar tampak valid: perubahan itu dapat merusak narasi dan timing turunannya. Repair terbatas atau gagal dengan alasan jelas.
- Deduplikasi/pengurangan jumlah klip harus eksplisit dan terlapor; hasil kosong tidak boleh menjadi sukses semu.
- Fence/thinking dibersihkan hanya jika format dikenali dan lengkap; output kosong/thinking terpotong tetap error.

### Retry lokal

SDK vLLM memakai `max_retries=0`; adapter mengendalikan maksimum **3 generation attempt total**, termasuk initial, transport retry, dan repair, dengan deadline keseluruhan. Policy Gemini dan NIM tetap lama.

| Kondisi | Tindakan |
|---|---|
| Connection refused, config/auth/model salah | Fail-fast, pesan aman/actionable |
| Timeout, 429, 5xx sementara | Backoff terbatas dalam budget attempt yang sama |
| Context overflow | Tidak retry identik |
| Schema/semantic invalid | Repair dengan konteks cukup dan preflight ulang |
| `finish_reason=length` | Gagal atau ubah budget/beban yang benar-benar muat; jangan terima JSON parsial |

Timeout dapat menyisakan request yang sedang dikerjakan server; benchmark risiko retry menambah beban. Error hierarchy: `ProviderConfigError`, `ProviderUnavailableError`, `ContextOverflowError`, `InvalidOutputError`, dengan metadata status/retryable yang aman.

## 6. Wiring dan perubahan file

| Area | Perubahan |
|---|---|
| `scripts/eval_local_llm.py`, fixtures | Spike/eval transkrip tersimpan tanpa Whisper/render |
| `tests/test_providers_*.py` | Karakterisasi cloud, kontrak lokal, request SDK |
| `clipping/providers/*`, `clipping/engine.py` | Provider bertahap + facade/dispatch |
| `clipping/config.py`, `main.py` | Config lokal dan validasi sesuai execution path |
| `clipping/runner.py` | Preflight sebelum pekerjaan mahal jika relevan; metadata run |
| `clipping/voiceover.py` | Milestone text generation lokal; signature/default cloud tetap kompatibel |
| `web/api/models.py`, `config_adapter.py`, `worker.py` | Enum/config/override end-to-end, cache bypass |
| `web/api/routes/settings.py`, router provider | Setting operator, status/probe dan key masking |
| `web/dashboard/src/pages/NewJob.jsx`, `Settings.jsx` | Form vLLM |
| `docs/studio/new-job.html`, `settings.html` | Paritas Studio statis |
| `requirements.txt`, `pyproject.toml`, `requirements-test.txt` | Client OpenAI versi kompatibel yang diuji; tanpa dependency server vLLM |
| `docker-compose.yml`, `pytest.ini` | Profil GPU/CPU/network dan marker opt-in |

Untuk Studio, `JobCreateRequest` memuat `vllm_model` dan `local_max_tokens` dengan validasi tipe/rentang. Base URL, key, dan context ceiling berasal dari operator, bukan endpoint arbitrer per job. Awali dengan env operator; UI editor endpoint menunggu kontrol akses dan URL validation.

Jika Settings ditambahkan, wiring harus mencakup request/response, allowlist penyimpanan, pembacaan env, `env_overrides`, adapter, worker, dan kedua UI. Test round-trip sampai request provider. Definisikan arti field kosong vs tidak dikirim; response key hanya berupa `vllm_api_key_set`.

`GET /api/ai/providers` harus dipasang pada router/prefix yang tepat, dengan probe singkat yang tidak memblokir event loop. `/api/ready` tetap tidak memanggil inference/validasi kredensial. Story mode dipertahankan.

Cache `gemini_response.json` dan flag `load_gemini_json` tetap berlaku. `ai_run_meta.json` dapat mencatat provider/model aktual, attempt, usage bila tersedia, durasi, fallback, dan cache hit pada CLI/worker tanpa prompt/raw response. Cache hit bukan generation baru.

## 7. Fase implementasi

### Fase 0 — Hardware dan spike

- [ ] Catat CPU/ISA, RAM, GPU/VRAM bila ada, driver/OS, dan topologi aplikasi-server.
- [ ] Pin satu runtime yang tersedia, model/revisi, dtype/quantization, template, dan limit konteks.
- [ ] Uji model listing, tokenisasi, chat, lalu schema klip lengkap pada fixture pendek dengan 1–2 klip.
- [ ] Ukur latency, token, peak RAM/VRAM, error, dan batas konteks; jalankan tanpa Google/NVIDIA key.

Selesai jika minimal satu profil target berhasil dan konfigurasinya tercatat. Profil lain tetap belum diuji. Belum perlu refactor provider cloud atau menjalankan Whisper.

### Fase 1 — Karakterisasi dan dependency client

- [ ] Snapshot schema/request Gemini dan NIM, prompt, timeout, sampling, routing, parser, dan retry.
- [ ] Matriks dispatcher/key/fallback, cache bypass CLI/worker, cfg minimal, import lewat path.
- [ ] Tambah dependency `openai` sebagai perbaikan terpisah; uji kedua jalur instalasi utama.

Selesai jika baseline test hijau dan dampak dependency dijelaskan. Jangan menyebut perbaikan import NIM sebagai perubahan yang tidak memengaruhi perilaku.

### Fase 2 — Ekstraksi cloud

- [ ] Ekstrak provider minimal, pertahankan facade/titik patch dan payload cloud.
- [ ] Jangan sekaligus mengubah parsing, schema, fallback atau retry.

Selesai jika test lama lulus tanpa diubah dan snapshot request tetap sama.

### Fase 3 — vLLM CLI dan pengaman dasar

- [ ] Config/dispatch/provider, token preflight, validasi hasil, retry bounded, metadata run.
- [ ] Tidak memerlukan key cloud dan tidak fallback ke cloud.
- [ ] Tolak vLLM + voiceover sampai fitur lokal tersedia.
- [ ] Smoke analisis fixture lalu satu pipeline CLI end-to-end tanpa voiceover.

Selesai jika output valid dapat dirender dan error server/model/context/length ditangani; cloud tetap hijau.

### Fase 4 — Eval representatif dan voiceover

- [ ] Fixture pendek/sedang/panjang dan jumlah klip realistis; review sampel ID/EN, timing, dan koherensi.
- [ ] Ukur waktu total, token, memori, kegagalan dan repair rate; tetapkan preset dari hasil.
- [ ] Tambah voiceover lokal dengan `generate_text`, termasuk cache + voiceover.
- [ ] Chunking/staged hanya jika hasil menunjukkan kebutuhan.

Selesai jika batas penggunaan/preset terdokumentasi dan jalur voiceover yang ditawarkan benar-benar lulus.

### Fase 5 — Studio

- [ ] Config env operator, enum/request, status provider, dan kedua UI.
- [ ] Round-trip ke worker dan satu job nyata melalui UI.
- [ ] Editor endpoint Settings hanya setelah kontrol akses/URL validation tersedia.

### Fase 6 — Deployment dan dokumentasi

- [ ] External server dan profil GPU/CPU, networking/auth/alias model/port, lifecycle memori.
- [ ] Dokumentasi versi yang diuji dan changelog fitur yang benar-benar tersedia.

Ollama, provider generik, fallback chain, field cloud web yang hilang, dan penertiban versi adalah pekerjaan lanjutan terpisah.

## 8. Test dan evaluasi

- Gemini: pertahankan SDK request builder nyata yang sudah ada.
- NIM/vLLM: fake client untuk unit test **dan** SDK OpenAI nyata dengan HTTP mock transport untuk serialisasi, URL/auth, timeout, dan retry. Fake module saja tidak cukup.
- Uji base root/`/v1`/trailing slash/proxy prefix; alias model, optional key, tidak memakai default cloud.
- Uji context metadata null/hilang, template/messages identik, input+output boundary, dan budget repair.
- Uji empty/malformed JSON, schema invalid, thinking terpotong, finish reason length, 401/403/404/429/5xx, timeout, connection refused, dan total attempt.
- Uji timing top-level/nested, NaN/infinity, durasi sumber, overlap, jumlah berkurang/kosong; jangan mengubah cloud melalui kontrak baru yang dipaksakan ke semua provider.
- Uji cache bypass tanpa probe/key, cache + voiceover, dan tidak ada request cloud pada jalur lokal.
- Uji settings persistence/allowlist/override sampai cfg worker dan UI payload.
- Test biasa offline. Real-server test memakai marker `local_llm` dan opt-in. Jika opt-in aktif tetapi server mati, test **gagal**, bukan skip yang menyamarkan deployment rusak.

Eval melaporkan konfigurasi, hasil mentah vs setelah repair, validitas schema, timing/nested timing, jumlah/overlap klip, kualitas ID/EN dan koherensi (review manusia), latency, token, serta peak RAM/VRAM. Warning metadata hanyalah salah satu indikator kualitas. Baseline Gemini opsional dan eksplisit, tidak otomatis dijalankan oleh eval lokal. Ambang latency/kualitas ditentukan sebelum evaluasi akhir sesuai kebutuhan, tanpa mengarang benchmark.

## 9. Deployment Linux GPU dan CPU

| Aspek | GPU | CPU |
|---|---|---|
| Runtime | Build/image sesuai vendor dan driver | Build/image CPU sesuai arsitektur |
| Memori | Bobot + KV cache + overhead VRAM, serta RAM | Bobot + KV cache + overhead RAM |
| Pemeriksaan | GPU/VRAM/driver dan dukungan versi | CPU/ISA VM, RAM, core, NUMA bila relevan |
| Tuning | Context, concurrency, memory utilization | Context, concurrency, thread/core, KV cache |
| Kelayakan | Smoke test pada hardware target | Smoke test dan benchmark; latency belum dijamin |

Untuk Linux x86, AVX512 direkomendasikan; AVX2 mempunyai keterbatasan. Dtype/quantization harus cocok dengan CPU/model/versi, bukan menganggap semua FP16 atau model 4-bit kompatibel. Sisakan RAM bagi OS/layanan lain dan core untuk serving. Pin versi sebelum memilih flag KV cache; dokumentasi saat review merekomendasikan `--kv-cache-memory-bytes`, sedangkan `VLLM_CPU_KVCACHE_SPACE` adalah opsi legacy. [CPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/).

CPU-only didukung, tetapi belum tentu memenuhi latency workload ini. Mulai dari instruct kecil dan fixture pendek; Qwen/Gemma/Llama adalah kandidat, bukan default teruji. Profil CPU tidak didapat hanya dengan menghapus reservasi GPU dari image CUDA.

GPU memerlukan build sesuai vendor/driver. Windows native bukan target; percobaan di PC Windows nantinya memerlukan jalur Linux/WSL tersendiri. [GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).

Topologi contoh, belum diuji pada server pengguna:

- Studio port host 8000; vLLM same-host port 8001 untuk menghindari benturan.
- Antar-container Compose, gunakan DNS service seperti `http://vllm:8000/v1`; port internal yang sama tidak bertabrakan antar-container.
- Backend container ke host memakai alamat host sesuai platform; Linux dapat memerlukan mapping `host-gateway`.
- Backend Kaggle ke server Linux memakai endpoint eksternal/private yang terjangkau. `localhost` Kaggle tidak menunjuk server pengguna.

Pisahkan environment/container vLLM dan clipping untuk menghindari konflik dependensi native/PyTorch/CUDA. Jika berbagi GPU, anggarkan Whisper dan vLLM sekaligus; batch CLI mempertahankan Whisper. Menurunkan memory utilization tidak menjamin muat. Gunakan mesin/GPU terpisah atau lifecycle unload yang benar-benar teruji bila perlu. Spike memakai transkrip tersimpan tanpa mengubah lifecycle aplikasi.

Arahan terbaru memasukkan vLLM dalam sesi Kaggle dengan satu GPU 16 GB. Preset QAT Q4 memakai GPU untuk vLLM dan CPU/int8 untuk Whisper; peak memory dan kelayakan end-to-end tetap harus diukur. Model gated memerlukan akses/lisensi dan token download di server, bukan pada request aplikasi.

## 10. Keamanan dan privasi

- Self-hosted pada server cloud tetap mengirim transkrip melalui jaringan; bukan otomatis offline.
- Gunakan jaringan private/VPN atau proxy terkontrol, auth dan TLS sesuai topologi. Jangan mengekspos seluruh server publik hanya karena endpoint `/v1` diberi key.
- Memindahkan URL ke Settings saja belum mencegah SSRF. Batasi perubahan endpoint ke operator tepercaya; env operator menjadi default sampai kontrol akses tersedia.
- Untuk URL dinamis, gunakan allowlist host/port deployment; validasi DNS/IP IPv4/IPv6, tolak alamat metadata/link-local dan redirect tak terverifikasi. Loopback/LAN hanya target operator yang disengaja. Tolak credentials dalam URL, query, dan fragment.
- Nama model tidak digunakan untuk shell/download model. Validasi panjang/control characters dan cocokkan served model; regex bukan pengganti kontrol endpoint.
- Probe singkat/dibatasi frekuensi. Key hanya ditampilkan sebagai status; log tidak memuat transkrip, raw body, key, atau URL bertoken.
- Cache hasil pipeline adalah artefak eksplisit dan berbeda dari log diagnostik; pertahankan kebijakan cache yang terdokumentasi.

## 11. Dokumentasi

Perbarui saat fitur terkait tersedia:

- `README.md`/`README_ID.md`: vLLM, kebutuhan server, contoh CLI.
- `wiki/3-CLI-Reference.md`: flag/default nyata.
- `wiki/13-Troubleshooting-and-FAQ.md`: server/model/auth/context/length/RAM/VRAM.
- `wiki/12-Google-Colab-Guide.md`, `docs/KAGGLE_STUDIO.md`: koneksi server eksternal dan batas klaim offline.
- `docs/providers.md`: versi/preset diuji, CPU/GPU, voiceover/fallback, token budget, hasil eval.
- `.env.sample`: konfigurasi baru sesuai akses repo saat implementasi; isinya belum ditinjau pada review ini.
- `CHANGELOG.md`: perubahan nyata dan perbaikan dependency; pilih versi ketika rilis.

Gap `docs/architecture.md` dan `docs/modules.md` dapat ditutup terpisah. Review plan saja tidak memerlukan entri rilis seolah provider sudah berjalan.

## 12. Keputusan dan bukti yang masih diperlukan

| Hal | Status |
|---|---|
| vLLM dahulu, Ollama ditunda | Arahan pengguna |
| Linux/cloud GPU atau CPU-only | Target awal Kaggle satu GPU 16 GB; preset CPU ditunda |
| Service terpisah, adapter sama | Rekomendasi review |
| Fallback vLLM ke cloud default mati | Dipertahankan; MVP tanpa chain generik |
| Model/revisi/dtype/quantization/context | Preset eksperimen Gemma 4 QAT W4A16 di atas; validasi target belum dilakukan |
| Default output budget/jumlah klip | Output 4096; notebook mulai satu klip. Default global jumlah klip tidak diubah |
| Voiceover lokal | Implementasi tersedia; validasi Kaggle belum dilakukan |
| Chunking/staged | Berdasarkan kebutuhan eval |
| Editor endpoint web | Setelah kontrol akses/URL validation |
| Drift versi/field cloud web | Perubahan terpisah |

Risiko utama: CPU lambat/RAM kurang; VRAM berebut dengan Whisper; JSON valid tetapi timing/isi salah; input+output overflow; regresi cloud saat ekstraksi; SDK/server drift; dan endpoint dinamis menjadi SSRF. Bukti yang diperlukan masing-masing adalah benchmark target, pengukuran peak memory/lifecycle, validasi temporal + review manusia, tokenisasi/boundary test, snapshot/test lama, versi terpin + smoke test, dan kontrol operator/network.

Belum dilakukan: instalasi vLLM, koneksi server target, eksekusi test, benchmark CPU/GPU, atau eval model pada hardware pengguna. Kode aplikasi sudah berubah, tetapi belum menjadi bukti server tertentu mampu menjalankannya.

Langkah berikutnya dilakukan di Kaggle: preflight hardware, setup dua environment, test regresi offline, start server, smoke fixture pendek, lalu satu job CLI/Studio dan pemeriksaan hasil. Catat error/memori/latensi sebelum menaikkan jumlah klip atau panjang transkrip.
