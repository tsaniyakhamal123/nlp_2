# Implementasi Transformer (GPT) dari Nol dengan NumPy

Proyek ini merupakan implementasi arsitektur *decoder-only Transformer* (model gaya GPT) dari dasar (*from scratch*) menggunakan NumPy. [cite_start]Implementasi ini dibuat untuk memenuhi tugas mata kuliah dan sebagai sarana pembelajaran untuk memahami cara kerja internal arsitektur Transformer tanpa bergantung pada *library deep learning* seperti PyTorch atau TensorFlow[cite: 3].

[cite_start]Fokus utama proyek ini adalah membangun alur *forward pass*, mulai dari lapisan embedding hingga menghasilkan distribusi probabilitas untuk prediksi token berikutnya[cite: 4].

## Struktur Proyek

[cite_start]Kode diorganisir secara modular untuk memisahkan setiap komponen utama dari arsitektur Transformer[cite: 18]:

-   `main.py`: Skrip utama yang merakit semua komponen menjadi model GPT utuh. Berisi kelas `TransformerBlock` dan `GPT`, serta skrip untuk menjalankan uji coba sederhana.
-   `embedding.py`: Berisi implementasi untuk `TokenEmbedding`, `PositionalEncoding` (sinusoidal), dan `RotaryPositionalEncoding (RoPE)`.
-   `attention.py`: Mengimplementasikan mekanisme inti `ScaledDotProductAttention` dan `MultiHeadAttention`, lengkap dengan fungsi `causal_mask`.
-   `ffn.py`: Berisi implementasi `FeedForward` network dan fungsi `LayerNormalization`.
-   `requirements.txt`: Daftar dependensi yang dibutuhkan untuk menjalankan proyek.

## Fitur dan Arsitektur

[cite_start]Model ini mengimplementasikan semua komponen wajib dari arsitektur Transformer[cite: 6]:

-   [cite_start]**Token Embedding**: Memetakan token input ke dalam representasi vektor[cite: 7].
-   [cite_start]**Positional Encoding**: Menggunakan metode sinusoidal standar untuk menyuntikkan informasi posisi[cite: 8]. [cite_start]Juga tersedia **Rotary Positional Encoding (RoPE)** sebagai fitur tambahan[cite: 31].
-   [cite_start]**Multi-Head Self-Attention**: Mekanisme atensi yang memungkinkan model untuk menimbang relevansi token lain, dengan beberapa "kepala" atensi yang berjalan paralel[cite: 10].
-   [cite_start]**Causal Masking**: Memastikan bahwa saat memprediksi token pada posisi `t`, model hanya dapat mengakses informasi dari token sebelumnya (posisi `< t`)[cite: 13].
-   [cite_start]**Feed-Forward Network (FFN)**: Jaringan dua lapis dengan aktivasi ReLU yang diterapkan setelah mekanisme atensi[cite: 11].
-   [cite_start]**Residual Connections & Layer Normalization**: Digunakan di setiap sub-lapisan untuk menstabilkan training dan memungkinkan model yang lebih dalam[cite: 12].

### Fitur Tambahan (Bonus)

[cite_start]Proyek ini juga mengimplementasikan beberapa fitur tambahan untuk penilaian bonus[cite: 31]:
1.  **Weight Tying**: Matriks bobot pada lapisan embedding dan lapisan output akhir saling berbagi (di-transpose) untuk mengurangi jumlah parameter.
2.  **Rotary Positional Encoding (RoPE)**: Implementasi positional encoding alternatif yang lebih modern.
3.  **Visualisasi Attention**: Skrip utama akan menghasilkan plot visualisasi dari bobot atensi pada salah satu *head* untuk menunjukkan token mana yang "diperhatikan" oleh model.

## Dependensi

[cite_start]Proyek ini hanya memerlukan **NumPy** untuk operasi komputasi dan **Matplotlib** untuk visualisasi[cite: 16].

```
numpy
matplotlib
```

## Cara Menjalankan

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/](https://github.com/)[NAMA_USER_ANDA]/[NAMA_REPO_ANDA].git
    cd [NAMA_REPO_ANDA]
    ```

2.  **Instal Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Skrip Utama**
    [cite_start]Program akan menjalankan satu kali *forward pass* dengan input token sederhana dan mencetak hasilnya[cite: 23].
    ```bash
    python main.py
    ```

### Contoh Output

Setelah dijalankan, program akan mencetak dimensi output (logits dan probabilitas) dan menampilkan plot visualisasi atensi.

```
✅ Logits shape: (1, 4, 20)
✅ Probabilitas token terakhir: [0.049... 0.050... ... ]
✅ Jumlah probabilitas: 1.0
```
(Sebuah jendela plot yang menampilkan matriks atensi akan muncul)

## Penulis

- **[Nama Anda]** - [NIM Anda]
