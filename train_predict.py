import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import StockPredictor, DirectionalLoss

def train_and_predict(X, y, X_infer, ticker, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    X_infer_tensor = torch.tensor(X_infer, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 适配升级后的数据流 (input_size 变成了 11 个)
    model = StockPredictor(input_size=11, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = DirectionalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{ticker}_lstm.pth")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            epochs = min(epochs, 10)
        except Exception:
            pass
            
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        prediction = model(X_infer_tensor).item()
        
    return prediction
