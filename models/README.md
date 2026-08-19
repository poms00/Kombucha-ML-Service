# Kombucha fermentation model

Model XGBoost dibuat dari data historis berlabel. Jangan gunakan data sintetis untuk
model produksi.

Format CSV training (`data/kombucha_fermentation_training.csv`):

```csv
temperature_liquid,co2,ph,fermentation_stage
26.4,620,4.1,aktif
25.9,780,3.6,aktif
```

Setiap baris harus berasal dari satu observasi sensor historis yang memiliki label
tahap fermentasi hasil verifikasi. Diperlukan minimal 20 baris dan dua kelas label.
Kolom label juga dapat bernama `stage` untuk kompatibilitas dengan dataset lama.

Jalankan training:

```powershell
.\venv\Scripts\python.exe .\training\train_fermentation_model.py --dataset .\data\kombucha_fermentation_training.csv
```

Training akan membuat `kombucha_xgb.pkl` dan `kombucha_xgb_metadata.json` di
folder ini. Artefak tersebut otomatis digunakan oleh `services/xgboost_service.py`.

File `.pkl` hanya boleh dibuat dan dimuat dari lingkungan tepercaya karena format
pickle dapat mengeksekusi kode saat dibuka.
