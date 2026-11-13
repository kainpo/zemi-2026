import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available. Running on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print("Preprocessing data...")
    # カテゴリ変数のエンコーディング
    le = LabelEncoder()
    df['subscription_type'] = le.fit_transform(df['subscription_type'])
    
    # 目的変数と説明変数の分離 (ここでは離脱予測モデルを想定)
    X = df[['age', 'subscription_type', 'login_frequency', 'total_purchase_amt', 'engagement_score']].values
    y = df['churn_risk'].values
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # スケーリング
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Tensorへの変換
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# 深層学習モデルの定義
class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ChurnPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_model():
    data_path = "/data/advanced_dataset.csv"  # DockerのVolumeマウント先のパス
    
    try:
        X_train, y_train, X_test, y_test = load_and_preprocess_data(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Make sure you mounted the volume correctly: -v /path/to/host/dir:/data")
        return
        
    # データローダーの作成
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # モデルの初期化
    model = ChurnPredictor(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 80
    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # テストデータでの評価
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            predictions = (test_outputs >= 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(dataloader):.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
