import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

DATA_FOLDER = 'data'

# WADI Saldırı Zamanları
WADI_ATTACK_RANGES = [
    ("2017-10-09 19:25:00", "2017-10-09 19:50:16"),
    ("2017-10-10 10:24:10", "2017-10-10 10:34:00"),
    ("2017-10-10 10:55:00", "2017-10-10 11:24:00"),
    ("2017-10-10 11:07:46", "2017-10-10 11:12:15"),
    ("2017-10-10 11:30:40", "2017-10-10 11:44:50"),
    ("2017-10-10 13:39:30", "2017-10-10 13:50:40"),
    ("2017-10-10 14:48:17", "2017-10-10 14:59:55"),
    ("2017-10-10 14:53:44", "2017-10-10 15:00:32"),
    ("2017-10-10 17:40:00", "2017-10-10 17:49:40"),
    ("2017-10-11 10:55:00", "2017-10-11 10:56:27"),
    ("2017-10-11 11:17:54", "2017-10-11 11:31:20"),
    ("2017-10-11 11:36:31", "2017-10-11 11:47:00"),
    ("2017-10-11 11:59:00", "2017-10-11 12:05:00"),
    ("2017-10-11 12:07:30", "2017-10-11 12:10:52"),
    ("2017-10-11 12:16:00", "2017-10-11 12:25:36"),
    ("2017-10-11 15:26:30", "2017-10-11 15:37:00"),
]

DATASETS = {
    'SWaT': {
        'train_raw': 'SWaT_normal.csv',
        'test_raw': 'merged.csv',
        'train_out': 'train.csv',
        'test_out': 'test.csv',
        'trim': 2160,
        'skiprows': 0,
    },
    'WADI':  {
        'train_raw':  'WADI_14days.csv',
        'test_raw': 'WADI_attackdata.csv',
        'train_out': 'wadi_train.csv',
        'test_out': 'wadi_test.csv',
        'trim': 2160,
        'train_skiprows': 4,
        'test_skiprows': 0,
    }
}

def clean_and_force_numeric(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')
    df = df.ffill().fillna(0)
    return df

def simplify_wadi_columns(df):
    new_columns = []
    for col in df.columns:
        col_str = str(col)
        if '\\' in col_str:
            col_str = col_str.split('\\')[-1]
        new_columns.append(col_str.strip())
    df.columns = new_columns
    return df

def create_wadi_labels(df):
    labels = np.zeros(len(df), dtype=int)
    # Date/Time bul
    date_col = next((c for c in df.columns if 'date' in str(c).lower()), None)
    time_col = next((c for c in df.columns if 'time' in str(c).lower()), None)
    
    if not date_col or not time_col:
        return pd.Series(labels)
    
    try:
        datetime_str = df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip()
        timestamps = pd.to_datetime(datetime_str, format='%m/%d/%Y %I:%M:%S.%f %p', errors='coerce')
        
        for start_str, end_str in WADI_ATTACK_RANGES:
            mask = (timestamps >= pd.to_datetime(start_str)) & (timestamps <= pd.to_datetime(end_str))
            labels[mask] = 1
    except:
        pass
    return pd.Series(labels)

def process_dataset(dataset_name):
    cfg = DATASETS[dataset_name]
    print(f"\n=== {dataset_name} İŞLENİYOR (CLIP FIX) ===")

    # 1. TRAIN YÜKLE
    train_path = os.path.join(DATA_FOLDER, cfg['train_raw'])
    print(f" [Train] Okunuyor: {cfg['train_raw']}")
    
    skip_tr = cfg.get('train_skiprows', cfg.get('skiprows', 0))
    try:
        df_train = pd.read_csv(train_path, skiprows=skip_tr, low_memory=False)
    except:
        df_train = pd.read_csv(train_path, skiprows=skip_tr, encoding='latin1', low_memory=False)
    
    df_train.columns = df_train.columns.astype(str).str.strip()
    if dataset_name == 'WADI':
        df_train = simplify_wadi_columns(df_train)
        drop_cols = ['Row', 'Date', 'Time']
        features_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
        labels_train = pd.Series([0] * len(features_train))
    else: # SWaT
        drop_cols = ['Timestamp', ' Timestamp', 'Normal/Attack']
        features_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
        labels_train = pd.Series([0] * len(features_train))

    features_train = clean_and_force_numeric(features_train)
    
    # --- NORMALİZASYON (FİT) ---
    print(" [Train] Scaler eğitiliyor...")
    scaler = MinMaxScaler()
    features_train_scaled = pd.DataFrame(scaler.fit_transform(features_train), columns=features_train.columns)
    
    # Train Downsampling
    n_groups = len(features_train_scaled) // 10
    features_train_down = features_train_scaled.iloc[:n_groups*10].groupby(np.arange(n_groups*10) // 10).median()
    labels_train_down = labels_train.iloc[:n_groups*10].groupby(np.arange(n_groups*10) // 10).apply(lambda x: x.mode()[0])
    
    if cfg['trim'] > 0:
        features_train_down = features_train_down.iloc[cfg['trim']:]
        labels_train_down = labels_train_down.iloc[cfg['trim']:]

    # Train Kaydet
    final_train = pd.concat([features_train_down, labels_train_down.rename('attack')], axis=1)
    final_train.to_csv(os.path.join(DATA_FOLDER, cfg['train_out']), index=False)
    print(f" ✅ Train Hazır: {final_train.shape}")


    # 2. TEST YÜKLE
    test_path = os.path.join(DATA_FOLDER, cfg['test_raw'])
    print(f" [Test] Okunuyor: {cfg['test_raw']}")
    
    skip_ts = cfg.get('test_skiprows', cfg.get('skiprows', 0))
    try:
        df_test = pd.read_csv(test_path, skiprows=skip_ts, low_memory=False)
    except:
        df_test = pd.read_csv(test_path, skiprows=skip_ts, encoding='latin1', low_memory=False)

    df_test.columns = df_test.columns.astype(str).str.strip()
    if dataset_name == 'WADI':
        df_test = simplify_wadi_columns(df_test)
        labels_test = create_wadi_labels(df_test)
        drop_cols = ['Row', 'Date', 'Time']
        features_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
    else: # SWaT
        drop_cols = ['Timestamp', ' Timestamp']
        if 'Normal/Attack' in df_test.columns:
            labels_test = df_test['Normal/Attack'].astype(str).str.strip().str.lower().apply(lambda x: 0 if x == 'normal' else 1)
            features_test = df_test.drop(columns=['Normal/Attack'] + [c for c in drop_cols if c in df_test.columns], errors='ignore')
        else:
            labels_test = pd.Series([0]*len(df_test))
            features_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

    features_test = clean_and_force_numeric(features_test)

    # --- KRİTİK ADIM: SÜTUN HİZALAMA VE CLIPPING ---
    print(" [Test] Normalizasyon ve CLIPPING uygulanıyor...")
    # 1. Sadece Train'de olan sütunları al (Garanti)
    features_test = features_test[features_train.columns]
    
    # 2. Transform et
    features_test_scaled = pd.DataFrame(scaler.transform(features_test), columns=features_test.columns)
    
    # 3. CLIP (0-1 ARASINA SIKIŞTIR) - İŞTE ÇÖZÜM BU!
    features_test_scaled = features_test_scaled.clip(0, 1)

    # Test Downsampling
    n_groups = len(features_test_scaled) // 10
    features_test_down = features_test_scaled.iloc[:n_groups*10].groupby(np.arange(n_groups*10) // 10).median()
    labels_test_down = labels_test.iloc[:n_groups*10].groupby(np.arange(n_groups*10) // 10).apply(lambda x: 1 if x.sum() > 0 else 0)

    if cfg['trim'] > 0:
        features_test_down = features_test_down.iloc[cfg['trim']:]
        labels_test_down = labels_test_down.iloc[cfg['trim']:]

    # Test Kaydet
    final_test = pd.concat([features_test_down, labels_test_down.rename('attack')], axis=1)
    final_test.to_csv(os.path.join(DATA_FOLDER, cfg['test_out']), index=False)
    
    attack_ratio = 100 * (final_test['attack'] == 1).sum() / len(final_test)
    print(f" ✅ Test Hazır: {final_test.shape} | Attack Oranı: %{attack_ratio:.2f}")

if __name__ == "__main__":
    process_dataset('SWaT') # SWaT veya WADI secilmeli
