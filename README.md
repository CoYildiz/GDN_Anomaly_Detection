# GDN - Graph Deviation Network for Anomaly Detection

Bu proje, Graph Neural Networks (GNN) kullanarak siber-fiziksel sistemlerde (SWaT ve WADI) anomali tespiti yapmayı hedeflemiştir. Tasarım projesi çalışması altında yapılmış olup orijinal (https://arxiv.org/pdf/2106.06947) makalesinin replikasyonu amaçlamıştır ama makale ile aynı sonuçlar elde edilmemiştir bundan dolayı bu proje Çok değişkenli zaman serilerinde GDN kullanılarak anomali tespti üzerine deneysel bir çalışma olmuştur.
Aynı sonuçları alamama sebepleri kullanılan epoch sayısı, makalede belirtilmemiş yöntemler, hiperparametreler, makaleden farklı (yılların) datasetleri kullanılmış olması olabilir.

## Özellikler
- **Model:** Graph Deviation Network (GDN)
- **Veri Setleri:** SWaT ve WADI
- **Yöntem:** Zaman serisi verilerini graf yapısına dönüştürerek sensörler arası ilişkileri öğrenir.
- **Teknikler:** Min-Max Normalization, Clipping, Point Adjustment.

## Kurulum
Tavsiye edilen çalıştırma şekli virtual env kullanılmasıdır.
Bash

    pip install -r requirements.txt

    Veri Setleri: data/ klasörü içine SWaT_normal.csv, merged.csv, WADI_14days.csv ve WADI_attackdata.csv dosyalarını ekleyin. (Lisans gereği repo içinde paylaşılmamıştır).

Kullanım

1. Veri Ön İşleme:

Bash

    python3 preprocess.py

2. Modeli Eğitme: main.py içindeki DATASET değişkenini 'SWaT' veya 'WADI' olarak ayarlayın ve çalıştırın:

Bash

    python3 main.py

3. Değerlendirme: Eğitim bittikten sonra sonuçları görmek için:

Bash

    python3 evaluate.py
