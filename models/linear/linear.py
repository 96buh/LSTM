import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random, time
import matplotlib.pyplot as plt

# 資料夾路徑設定與分類對應
LABEL_DIRS = {
    0: "./dataset/normal",
    1: "./dataset/abnormal/transformer_rust",
    2: "./dataset/abnormal/wire_rust",
    3: "./dataset/abnormal/wire_peeling"
}
RESULT_DIR = "./models/linear/result"
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_SEQ_LEN = 10

all_data = []
all_labels = []

def process_file(file_path, label, max_seq_len=MAX_SEQ_LEN):
    """
    讀取 CSV 檔案，取出 'current', 'voltage', 'power' 三個欄位，
    將數據切分為固定長度的片段，並將每個片段展平成一個一維向量。
    """
    df = pd.read_csv(file_path)
    current = df['current'].values
    voltage = df['voltage'].values
    power   = df['power'].values

    # 組合成 (N, 3) 的矩陣
    sequence = np.column_stack((current, voltage, power))
    seq_len = sequence.shape[0]
    num_chunks = seq_len // max_seq_len  # 僅取完整片段

    for i in range(num_chunks):
        start = i * max_seq_len
        end = start + max_seq_len
        chunk = sequence[start:end]
        chunk_flat = chunk.flatten()  # 將 (MAX_SEQ_LEN, 3) 變為一維向量
        all_data.append(chunk_flat)
        all_labels.append(label)

def load_all_data():
    """
    根據 LABEL_DIRS 中定義的各分類資料夾，讀取所有 CSV 檔案，
    並印出每個分類切分後的樣本數量。
    """
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                process_file(file_path, label)
    data_array = np.array(all_data, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)
    
    print("Data shape:", data_array.shape)
    print("Labels shape:", labels_array.shape)
    for i in sorted(LABEL_DIRS.keys()):
        print(f"Number of class {i} samples:", np.sum(labels_array == i))
    print("Unique labels:", np.unique(labels_array))
    return data_array, labels_array

class ChargeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ChargeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ChargeClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)


def main():
    # 載入數據
    data_array, labels_array = load_all_data()
    
    # 建立 Dataset 並切分成訓練集與驗證集 (80% / 20%)
    dataset = ChargeDataset(data_array, labels_array)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 建立模型，設定輸入維度與 4 個分類
    input_dim = MAX_SEQ_LEN * 3  # 例如：10*3=30
    num_classes = 4
    model = ChargeClassifier(input_dim, num_classes)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 訓練前先評估驗證集 (初始模型)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    print(f"Validation Accuracy before training: {100 * correct / total:.2f}%")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    start_time = time.time()
    num_epochs = 100
    for epoch in range(num_epochs):
        # 訓練階段：計算整個訓練集的平均 loss 與準確率
        model.train()
        train_loss_sum, correct_train, total_train = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * x_batch.size(0)
            _, predicted = torch.max(pred, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_loss = train_loss_sum / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 驗證階段：計算驗證集的平均 loss 與準確率
        model.eval()
        val_loss_sum = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss_sum += loss.item() * x.size(0)
                _, predicted = torch.max(pred, 1)
                correct_val += (predicted == y).sum().item()
                total_val += y.size(0)
        val_loss = val_loss_sum / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    end_time = time.time()
    print(f"⌛ Training time: {end_time - start_time:.2f} seconds")
    
    # 訓練後在驗證集上進行最終評估
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    print(f"Validation Accuracy after training: {accuracy:.2f}%")
    print(f"Validation Loss after training: {avg_loss:.4f}")
    
    # 儲存模型
    model_save_path = os.path.join(RESULT_DIR, "charge_classifier_mlp.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

     # ==== 繪製Loss與Accuracy圖 ====
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Accuracy')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULT_DIR, "training_metrics.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Metrics plot saved to {plot_path}")

if __name__ == "__main__":
    main()
