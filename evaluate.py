import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from utils import get_loaders
from models.model import GDN

# --- AYARLAR ---
DATASET = 'SWaT'  # 'SWaT' veya 'WADI'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = f'models/best_{DATASET}.pt'

if DATASET == 'SWaT':
    test_path = 'data/test.csv'
    train_path = 'data/train.csv'
    dim = 64
    slide_win = 5
    batch_size = 64
    topk = 15
elif DATASET == 'WADI':
    test_path = 'data/wadi_test.csv'
    train_path = 'data/wadi_train.csv'
    dim = 128
    slide_win = 5
    batch_size = 64
    topk = 30

print(f" Ayarlar: {DATASET} | Dim: {dim} | TopK: {topk}")

def get_errors(model, loader):
    model.eval()
    all_errors = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device).permute(0, 2, 1)
            y = batch['y'].to(device)
            label = batch['label'].to(device)
            preds = model(x)
            error = torch.abs(y - preds)
            all_errors.append(error.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    return np.concatenate(all_errors), np.concatenate(all_labels)

def point_adjustment(preds, labels):
    """
    Saldırı aralığının herhangi bir yerinde alarm varsa, 
    tüm aralığı 'bulundu' olarak işaretle (Makale Standardı).
    """
    pos_preds = preds.copy()
    actual_anomalies = np.where(labels == 1)[0]
    
    if len(actual_anomalies) == 0:
        return pos_preds
        
    # Anomali segmentlerini bul
    splits = np.where(np.diff(actual_anomalies) > 1)[0]
    splits = np.append(splits, len(actual_anomalies)-1)
    
    prev = 0
    for split in splits:
        segment = actual_anomalies[prev:split+1]
        # Eğer bu segmentte model herhangi bir 1 bulduysa
        if np.sum(preds[segment]) > 0:
            pos_preds[segment] = 1
        prev = split + 1
        
    return pos_preds

if __name__ == "__main__":
    print(f"--- DETAYLI MODEL ANALİZİ: {DATASET} ---")
    
    # 1. Yükle
    train_loader, test_loader = get_loaders(train_path, test_path, window=slide_win, batch_size=batch_size)
    sample = next(iter(train_loader))
    node_num = sample['x'].shape[2]
    
    # 2. Model
    model = GDN(edge_index_sets=None, node_num=node_num, dim=dim, 
                input_dim=slide_win, topk=topk).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(" Model yüklendi.")
    except:
        print(" Model bulunamadı! Lütfen önce main.py çalıştırın.")
        exit()
    
    # 3. Hata Hesapla
    print(" Hatalar hesaplanıyor...")
    train_errors, _ = get_errors(model, train_loader)
    test_errors, test_labels = get_errors(model, test_loader)

    # 4. Normalizasyon (Median/IQR)
    median = np.median(train_errors, axis=0)
    q1 = np.percentile(train_errors, 25, axis=0)
    q3 = np.percentile(train_errors, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr < 1e-6] = 1e-6 

    norm_errors_test = (test_errors - median) / iqr
    test_scores = np.max(norm_errors_test, axis=1)
    
    # Smoothing
    test_scores_smooth = pd.Series(test_scores).rolling(window=10, min_periods=1).mean().values

    # 5. BEST F1 SEARCH (Otomatik Eşik Belirleme)
    print("\n--- EN İYİ EŞİK DEĞERİ ARANIYOR ---")
    
    best_f1 = 0
    best_th = 0
    best_preds = None
    
    # Skorların min ve max aralığında 100 adım atarak tara
    min_score = np.min(test_scores_smooth)
    max_score = np.max(test_scores_smooth)
    # Çok yüksek outlier varsa aralığı daralt (Percentile 99.9'a kadar bak)
    max_score_capped = np.percentile(test_scores_smooth, 99.9)
    thresholds = np.linspace(min_score, max_score_capped, 100)
    
    for th in thresholds:
        preds = (test_scores_smooth > th).astype(int)
        # Ham F1'e göre optimize et
        f1 = f1_score(test_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = preds

    print(f"Bulunan En İyi Eşik: {best_th:.4f}")
    
    # 6. SONUÇLAR
    print("\n" + "="*40)
    print(" SONUÇLAR (RAW / HAM)")
    print("="*40)
    print(f"F1-Score  : {best_f1:.4f}")
    print(f"Precision : {precision_score(test_labels, best_preds):.4f}")
    print(f"Recall    : {recall_score(test_labels, best_preds):.4f}")
    print(f"AUC       : {roc_auc_score(test_labels, test_scores_smooth):.4f}")
    print("-" * 40)
    print(f"Confusion Matrix:\n{confusion_matrix(test_labels, best_preds)}")

    # 7. POINT ADJUSTMENT (MAKALE STANDARDI)
    print("\n" + "="*40)
    print(" SONUÇLAR (POINT ADJUSTMENT)")
    print(" (Anomali aralığının bir kısmı yakalanırsa tam puan)")
    print("="*40)
    
    pa_preds = point_adjustment(best_preds, test_labels)
    pa_f1 = f1_score(test_labels, pa_preds)
    pa_prec = precision_score(test_labels, pa_preds)
    pa_rec = recall_score(test_labels, pa_preds)
    
    print(f"F1-Score  : {pa_f1:.4f}")
    print(f"Precision : {pa_prec:.4f}")
    print(f"Recall    : {pa_rec:.4f}")
    print("-" * 40)
    
    # Grafik
    plt.figure(figsize=(15, 6))
    plt.plot(test_scores_smooth, label='Anomaly Score', color='blue', alpha=0.6)
    plt.axhline(y=best_th, color='red', linestyle='--', label='Best Threshold')
    # Gerçek saldırıları da boyayalım
    attack_indices = np.where(test_labels == 1)[0]
    plt.scatter(attack_indices, [min_score]*len(attack_indices), color='red', s=1, label='Actual Attack', alpha=0.5)
    
    plt.title(f'{DATASET} Anomaly Detection (Best F1)')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'final_analysis_{DATASET}_optimized.png')
    print(f"Grafik kaydedildi: final_analysis_{DATASET}_optimized.png")
