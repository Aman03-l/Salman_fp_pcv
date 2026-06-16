# Platypus 

Game tembak-tembakan absurd yang dikendalikan cuma pakai gerakan tangan.  
Kamu adalah pesawat segitiga yang terbang mengikuti telapak tangan, menembak musuh beku berwarna biru, sambil menghindari peluru ungu.  
Semua berjalan *real-time* lewat webcam, tanpa mouse atau keyboard buat main.

---

## Deskripsi

Proyek ini adalah game interaktif berbasis visi komputer yang memanfaatkan kamera laptop/webcam.  
Pemain mengontrol pesawat dengan menggerakkan tangan di depan kamera, dan gestur jari digunakan untuk menembak, mengaktifkan slow motion, serta mengendalikan tempo permainan.  
Game dibuat menggunakan **Python**, **OpenCV**, **MediaPipe**, dan **NumPy** – dengan penekanan pada pengolahan citra *real-time* dan logika permainan sederhana.

---

## Fitur

- **Hand tracking real-time** – posisi tangan dideteksi setiap frame, pergerakan dihaluskan (*smoothing*) agar nyaman dimainkan.
- **Menembak otomatis** – tembakan muncul saat menunjukkan 3 atau 5 jari.
- **Slow motion** – musuh dan peluru melambat ketika pemain menunjukkan 4 atau 5 jari.
- **Sistem skor & level** – setiap musuh yang hancur memberi 1 poin, tiap 10 poin level naik, musuh makin kuat.
- **HP & invincibility** – pemain punya 5 nyawa; setelah kena damage, ada jeda 1 detik kebal.
- **Sprite overlay manual** – pesawat digambar poligon *pure NumPy*, lalu ditempelkan ke frame dengan **alpha blending custom**.
- **UI On-screen** – panel hitam menampilkan skor, level, dan jumlah jari yang terdeteksi.
- **Tiga state game** – Start Screen, Playing, Game Over.

---

## Kontrol Gestur

| Jumlah Jari Terdeteksi | Aksi                           |
|------------------------|--------------------------------|
| 0 – 2                  | Hanya bergerak                 |
| 3                      | Menembak + bergerak            |
| 4                      | Slow motion + bergerak         |
| 5                      | Menembak + Slow motion         |

- **Posisi tangan** = posisi pesawat. Gerakkan tangan ke kiri/kanan/atas/bawah, pesawat akan mengikuti.
- **Tekan SPACE** di layar Start atau Game Over untuk mulai/restart.
- **Tekan Q** untuk keluar.

---

## Cara Kerja (Pipeline)

### 1. Akuisisi Kamera
Webcam diakses menggunakan **cv2.VideoCapture(0)**.  
Setiap frame dibaca dengan `cap.read()`, lalu dibalik horizontal (`cv2.flip`) agar seperti cermin, dan sedikit dikoreksi kontrasnya.

### 2. Deteksi Tangan dengan MediaPipe
**MediaPipe Hands** mendeteksi 21 *landmark* tangan dalam satu frame.  
Pergelangan tangan (wrist) digunakan sebagai acuan posisi pesawat (`target_x`, `target_y`).  
Posisi ini kemudian diinterpolasi dengan *exponential smoothing* (`smoothness = 0.35`) agar gerakan tidak patah-patah.

### 3. Pengenalan Gestur Jari
Fungsi `count_fingers()` memeriksa posisi ujung jari terhadap sendi di bawahnya untuk menentukan jari tertekuk atau terangkat.  
- Jari telunjuk, tengah, manis, kelingking: dihitung terangkat jika ujungnya di atas sendi PIP (y lebih kecil).  
- Ibu jari: diperiksa berdasarkan sumbu x sesuai handedness (kanan/kiri).

### 4. Overlay Sprite Pesawat (Alpha Blending Manual)
Sprite pesawat dibuat *on-the-fly* menggunakan array NumPy (ukuran 60×60 piksel, 4 channel BGRA).  
Poligon diisi warna dengan `cv2.fillPoly`, kemudian sprite ditempelkan ke frame utama menggunakan fungsi **`overlay_sprite()`** yang melakukan alpha blending manual:
```python
alpha = sub_sprite[:, :, 3:4] / 255.0
blended = (sub_sprite[:, :, :3] * alpha) + (roi * (1 - alpha))
