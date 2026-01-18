import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from utils import get_loaders
from models.model import GDN 

DATASET = 'WADI'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if DATASET == 'SWaT':
    train_path = 'data/train.csv'
    test_path  = 'data/test.csv'
    dim = 64          
    topk = 15         
    slide_win = 5     
    batch_size = 64   
    epochs = 50       # MAKALE STANDARDI 
elif DATASET == 'WADI':
    train_path = 'data/wadi_train.csv'
    test_path  = 'data/wadi_test.csv'
    dim = 128         
    topk = 30         
    slide_win = 5
    batch_size = 64
    epochs = 20

def train(model, loader, optimizer, criterion):
    model.train() 
    total_loss = 0
    if len(loader) == 0:
        raise ValueError("Train loader is empty")
    for i, batch in enumerate(loader):
        x = batch['x'].to(device).permute(0, 2, 1) 
        y = batch['y'].to(device) 
        optimizer.zero_grad()     
        preds = model(x)          
        loss = criterion(preds, y)
        loss.backward()           
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()          
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval() 
    total_loss = 0
    if len(loader) == 0:
        raise ValueError("Validation loader is empty")
    with torch.no_grad(): 
        for batch in loader:
            x = batch['x'].to(device).permute(0, 2, 1)
            y = batch['y'].to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    print(f"Veri Yükleniyor: {DATASET}...")
    train_loader, val_loader = get_loaders(train_path, test_path, window=slide_win, batch_size=batch_size)
    sample_batch = next(iter(train_loader))
    node_num = sample_batch['x'].shape[2] 
    
    model = GDN(edge_index_sets=None, node_num=node_num, dim=dim, 
                input_dim=slide_win, topk=topk).to(device)
    
    # Makale: Adam optimizer, lr=1e-3 [cite: 234]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"\n EĞİTİM BAŞLIYOR ({epochs} Epoch)...")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        loss_train = train(model, train_loader, optimizer, criterion)
        loss_val = validate(model, val_loader, criterion)
        
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), f'models/best_{DATASET}.pt')
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss_train:.5f} | Val Loss: {loss_val:.5f}")

    print(f"\nEğitim Tamamlandı! Model: 'models/best_{DATASET}.pt'")
