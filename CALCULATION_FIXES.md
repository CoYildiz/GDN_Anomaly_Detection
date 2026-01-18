# Hesaplama Düzeltmeleri ve Doğrulama (Calculation Fixes and Verification)

Bu belge, GDN projesindeki hesaplama hatalarını ve yapılan düzeltmeleri açıklar.

## Özet (Summary)

Kod tabanında tespit edilen ve düzeltilen hesaplama hataları:

### 1. **IQR Normalizasyon Hatası** (`evaluate.py`)

**Sorun:**
```python
iqr = q3 - q1 + 1e-6  # YANLIŞ: Tüm değerlere 1e-6 ekliyor
```

**Açıklama:** 
Orijinal kod, sıfıra bölme hatasını önlemek için tüm IQR değerlerine 1e-6 ekliyordu. Bu, normalizasyon hesaplamalarını bozar çünkü:
- Büyük IQR değerleri gereksiz yere artırılır
- Normalizasyon skalaları bozulur
- Anomali skorları yanlış hesaplanır

**Çözüm:**
```python
iqr = q3 - q1
iqr[iqr < 1e-6] = 1e-6  # DOĞRU: Sadece küçük değerleri değiştir
```

Bu yaklaşım:
- Sadece çok küçük IQR değerlerini minimum bir threshold ile değiştirir
- Normal IQR değerlerini korur
- StandardScaler'daki yaklaşımla tutarlıdır

---

### 2. **Standartlaştırma Formülü Yorumu** (`utils.py`)

**Sorun:**
```python
# Formul = deger - ortalama) / standarsapma  # YANLIŞ: Eksik parantez
```

**Çözüm:**
```python
# Formul = (deger - ortalama) / standarsapma  # DOĞRU
```

Bu sadece bir yorum hatası ama formülü anlamayı zorlaştırıyordu.

---

### 3. **Embedding Normalizasyonu** (`models/model.py`)

**Sorun:**
```python
normalized_embedding = self.embedding.div(norm)  # Potansiyel sıfıra bölme
```

**Açıklama:**
Embedding vektörlerinin L2 normu teorik olarak sıfır olabilir veya çok küçük olabilir, bu durumda bölme işlemi numerik kararsızlığa yol açar.

**Çözüm:**
```python
normalized_embedding = self.embedding.div(norm + 1e-8)  # Epsilon ile koruma
```

---

### 4. **Veri Doğrulama Eksikliği**

Birçok fonksiyonda boş veri veya geçersiz girdi kontrolü yoktu. Eklenen doğrulamalar:

**`utils.py - StandardScaler`:**
```python
def fit(self, data):
    if len(data) == 0:
        raise ValueError("Cannot fit scaler with empty data")
    # ...

def transform(self, data):
    if self.mean is None or self.std is None:
        raise ValueError("Scaler must be fitted before transform")
    # ...
```

**`utils.py - TimeDataset`:**
```python
def __init__(self, ...):
    if len(raw_data) <= window:
        raise ValueError(f"Dataset too small: {len(raw_data)} samples, need at least {window + 1}")
    
    if mode != 'train' and scaler is None:
        raise ValueError("Test mode requires a fitted scaler")
    # ...
```

**`main.py - train/validate`:**
```python
def train(model, loader, optimizer, criterion):
    if len(loader) == 0:
        raise ValueError("Train loader is empty")
    # ...
```

**`evaluate.py - get_errors`:**
```python
def get_errors(model, loader):
    # ...
    if len(all_errors) == 0:
        raise ValueError("No errors computed - loader might be empty")
    return np.concatenate(all_errors), np.concatenate(all_labels)
```

---

### 5. **Yazım Hataları**

**`preprocess.py`:**
```python
# Önce: "...odel yanlış öğrenebilir."
# Sonra: "...model yanlış öğrenebilir."
```

---

## Hesaplama Doğruluğu Analizi

### IQR Tabanlı Normalizasyon Neden Önemli?

Anomali tespitinde IQR (Interquartile Range) kullanımı robust (sağlam) bir yöntemdir çünkü:

1. **Outlier'lara Dayanıklı:** Standart sapmanın aksine, aykırı değerlerden etkilenmez
2. **Simetrik Olmayan Dağılımlar:** Normal dağılım varsayımı gerektirmez
3. **Ölçek Bağımsız:** Farklı sensörleri normalize ederken tutarlıdır

**Doğru Formül:**
```python
normalized_error = (error - median) / IQR
```

**IQR Hesaplama:**
```python
Q1 = 25. persentil (alt çeyrek)
Q3 = 75. persentil (üst çeyrek)
IQR = Q3 - Q1
```

**Sıfıra Bölme Koruması:**
```python
# Sadece çok küçük IQR değerlerini değiştir
iqr[iqr < 1e-6] = 1e-6
```

### Eşik (Threshold) Seçimi

```python
threshold = np.percentile(train_scores_smooth, 99.5)
```

- Eğitim verisinin %99.5 persentilini eşik olarak kullanır
- Bu, eğitim verisinin üst %0.5'ini anomali olarak kabul eder
- Normal verinin çoğunluğu eşiğin altında kalır
- Test verisinde bu eşiği aşanlar anomali olarak işaretlenir

### Yumuşatma (Smoothing)

```python
scores_smooth = pd.Series(scores).rolling(window=10, min_periods=1).mean()
```

- 10 zaman adımlık hareketli ortalama
- Gürültüyü azaltır
- Anomali skorlarını daha kararlı hale getirir
- `min_periods=1`: İlk noktalarda bile hesaplama yapar

---

## Test Edilmesi Gerekenler

Bu düzeltmelerden sonra kontrol edilmesi gerekenler:

1. **Model Eğitimi:**
   ```bash
   python main.py
   ```
   - Eğitim ve validasyon loss'ları makul mı?
   - Loss azalıyor mu?
   - Model kaydediliyor mu?

2. **Model Değerlendirmesi:**
   ```bash
   python evaluate.py
   ```
   - F1-Score, Precision, Recall değerleri mantıklı mı?
   - Confusion matrix dengeli mi?
   - Threshold değeri makul bir aralıkta mı?

3. **Veri Ön İşleme:**
   ```bash
   python preprocess.py
   ```
   - Downsampling doğru çalışıyor mu?
   - Etiketler doğru mu?
   - Saldırı oranları beklenen aralıkta mı?

---

## Referanslar

- **GDN Paper:** Graph Neural Network-Based Anomaly Detection in Multivariate Time Series
- **IQR Method:** Robust normalization using Interquartile Range
- **PyTorch Geometric:** Graph neural network framework

---

## Değişiklik Günlüğü

**Tarih:** 2025-12-25

**Değiştirilen Dosyalar:**
- `utils.py`: StandardScaler ve TimeDataset validation
- `evaluate.py`: IQR calculation fix, error handling
- `models/model.py`: Embedding normalization stability
- `main.py`: Loader validation
- `preprocess.py`: Typo fix

**Etki:**
- Daha doğru anomali skorları
- Daha iyi numerik stabilite
- Daha iyi hata yakalama ve debugging
- Kodun bakımı daha kolay
