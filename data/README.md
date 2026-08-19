# Dataset training kombucha

`kombucha_fermentation_training.csv` sengaja dibuat kosong selain header. Setiap
baris harus memuat pembacaan sensor nyata dan label tahap hasil verifikasi manual,
misalnya `awal`, `aktif`, atau `matang`.

Tambahkan observasi melalui FastAPI:

```http
POST /fermentators/kombucha_3/training-samples
```

```json
{
  "temperature_liquid": 26.4,
  "co2": 620,
  "ph": 4.1,
  "fermentation_stage": "aktif"
}
```

Endpoint menyimpan baris yang sama ke CSV lokal dan Firebase
`/fermentators/{fermentator_id}/training_samples`. Jangan memberi label otomatis
dari pH atau suhu; label harus berasal dari tahapan fermentasi yang benar-benar
terverifikasi.
