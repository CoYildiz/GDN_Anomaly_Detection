import pandas as pd
import os

# Ayarlar
DATA_FOLDER = 'data'
FILES = {
    'WADI_Test': 'WADI_attackdata.csv',
    'WADI_Attack_Labels': 'attack_description.xlsx'
}

def inspect():
    print("--- VERİ İNCELEME MODU ---\n")

    # 1. WADI TEST CSV İNCELEME
    # ---------------------------------------------------------
    csv_path = os.path.join(DATA_FOLDER, FILES['WADI_Test'])
    if os.path.exists(csv_path):
        print(f"DOSYA: {FILES['WADI_Test']}")
        try:
            # Sadece ilk 5 satırı oku
            df = pd.read_csv(csv_path, nrows=5)
            print(df[['Date', 'Time']].head())
            print(f"Örnek Tarih Birleşimi: {df['Date'].iloc[0]} {df['Time'].iloc[0]}")
        except Exception as e:
            print(f"Okuma Hatası: {e}")
    else:
        print(f"HATA: {FILES['WADI_Test']} bulunamadı.")
    
    print("-" * 50 + "\n")

    # 2. SALDIRI DOSYASI (EXCEL) İNCELEME
    # ---------------------------------------------------------
    xlsx_path = os.path.join(DATA_FOLDER, FILES['WADI_Attack_Labels'])
    if os.path.exists(xlsx_path):
        print(f"DOSYA: {FILES['WADI_Attack_Labels']}")
        try:
            # Önceki kodda başlığın 1. satırda (index 1) olduğunu görmüştük.
            # Header=1 diyerek okuyalım ve altındaki veriye bakalım.
            df = pd.read_excel(xlsx_path, engine='openpyxl', header=1, nrows=5)
            
            # Tüm kolonları ve ilk satırların tiplerini görelim
            print("BULUNAN KOLONLAR:", list(df.columns))
            print("\nİLK 5 SATIR VERİSİ:")
            print(df.head())
            
            # Kritik Kolonların Tipi Ne?
            # 'Start Date (Time)' kolonunun ismini tam tutturamasak bile içinde 'Start' geçen kolonu bulup tipine bakalım
            start_col = [c for c in df.columns if 'Start' in str(c)][0]
            val = df[start_col].iloc[0]
            print(f"\nTip Analizi ({start_col}):")
            print(f"  Değer: {val}")
            print(f"  Tip:   {type(val)}")
            
        except Exception as e:
            print(f"Okuma Hatası: {e}")
    else:
        print(f"HATA: {FILES['WADI_Attack_Labels']} bulunamadı.")

if __name__ == "__main__":
    inspect()
