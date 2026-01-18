import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np 

class StandardScaler:
    def __init__(self, mean = None , std = None):
        self.mean = mean
        self.std = std 
    
    def fit(self, data):
        # Veri kontrolü
        if len(data) == 0:
            raise ValueError("Cannot fit scaler with empty data")
        
        self.mean = data.mean(0)
        self.std = data.std(0)

        # Küçük standart sapma değerlerini 1.0 ile değiştir (bölme hatasını önle)
        self.std[self.std < 1e-4] = 1.0

    def transform(self, data):
        # Formul = (deger - ortalama) / standarsapma
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        return (data - self.mean) / self.std

class TimeDataset(Dataset):
    def __init__(self, filepath , window=5, mode = 'train', scaler = None):
        self.window = window
        self.mode = mode 

        #veri okuma
        df = pd.read_csv(filepath)

        #son sutun etiket
        raw_data = df.iloc[:, :-1].values
        self.labels = df.iloc[:, -1].values

        # Veri boyutu kontrolü
        if len(raw_data) <= window:
            raise ValueError(f"Dataset too small: {len(raw_data)} samples, need at least {window + 1}")

        #normalizasyon

        if mode == 'train':
            self.scaler = StandardScaler() 
            self.scaler.fit(raw_data)
        else:
            if scaler is None:
                raise ValueError("Test mode requires a fitted scaler")
            self.scaler = scaler

        self.x = self.scaler.transform(raw_data)

        #Veri boyutu kontrolu
        self.n_samples = len(self.x) - self.window

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        
        start_idx = index 
        end_idx = index + self.window

        x_window = self.x[start_idx : end_idx]

        y_target = self.x[end_idx]


        label = self.labels[end_idx]

        return {
                'x': torch.tensor(x_window, dtype = torch.float32),
                'y': torch.tensor(y_target, dtype = torch.float32),
                'label': torch.tensor(label, dtype = torch.float32)
                }

def get_loaders(train_path, test_path , window= 5 , batch_size = 64):
    train_dataset = TimeDataset(train_path, window=window, mode='train')
    test_dataset = TimeDataset(test_path, window=window, mode='test', scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

