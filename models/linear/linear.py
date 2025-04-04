import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.preprocessing import StandardScaler

LABEL_DIRS = {
    0: "./dataset/normal",
    1: "./dataset/abnormal/transformer_rust",
    2: "./dataset/abnormal/wire_rust",
    3: "./dataset/abnormal/wire_peeling"
}
RESULT_DIR = "./models/linear/result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------
# 資料處理函數
# ---------------------------
def process_file(file_path, label, max_seq_len):
    """
    讀取 CSV 檔案，取出 'current', 'voltage', 'power' 三個欄位，
    並將數據切分為固定長度的片段，將每個片段展平成一維向量。
    """
    df = pd.read_csv(file_path)
    current = df['current'].values
    voltage = df['voltage'].values
    power   = df['power'].values

    # 組成 (N, 3) 的矩陣
    sequence = np.column_stack((current, voltage, power))
    seq_len = sequence.shape[0]
    num_chunks = seq_len // max_seq_len  
    data = []
    for i in range(num_chunks):
        start = i * max_seq_len
        end = start + max_seq_len
        chunk = sequence[start:end]
        chunk_flat = chunk.flatten()  # flatten 成一維向量
        data.append(chunk_flat)
    return data

def load_all_data(max_seq_len):
    """
    根據 LABEL_DIRS 讀取所有 CSV 檔案，
    並印出每個類別切分後的樣本數量。
    """
    all_data = []
    all_labels = []
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                segments = process_file(file_path, label, max_seq_len)
                all_data.extend(segments)
                all_labels.extend([label] * len(segments))
    data_array = np.array(all_data, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    # 使用 StandardScaler 對數據進行標準化
    scaler = StandardScaler()
    data_array = scaler.fit_transform(data_array)
    
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

# ---------------------------
# 網格搜尋與訓練流程
# ---------------------------
def grid_search():
    # 定義超參數範圍
    batch_size_values = [16, 32, 64]
    learning_rate_values = [1e-3, 1e-4, 1e-5]
    max_seq_len_values = [10, 20, 30]
    num_epochs = 100

    overall_results = []

    # 依照 MAX_SEQ_LEN 進行外層迴圈（此參數會影響資料切分）
    for max_seq in max_seq_len_values:
        print(f"\n=== Running experiments with MAX_SEQ_LEN = {max_seq} ===")
        data_array, labels_array = load_all_data(max_seq_len=max_seq)
        dataset = ChargeDataset(data_array, labels_array)
        
        # 使用 80% 為訓練集，20% 為驗證集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 內層超參數組合：Batch Size 與 Learning Rate
        for bs in batch_size_values:
            for lr in learning_rate_values:
                print(f"\n--- Experiment with Batch Size: {bs}, Learning Rate: {lr} ---")
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                
                # 根據 max_seq 更新輸入維度（max_seq * 3）
                input_dim = max_seq * 3
                num_classes = 4
                model = ChargeClassifier(input_dim, num_classes)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                train_losses = []
                val_losses = []
                train_accs = []
                val_accs = []
                start_time = time.time()
                
                for epoch in range(num_epochs):
                    # 訓練階段
                    model.train()
                    train_loss_sum = 0.0
                    correct_train = 0
                    total_train = 0
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
                    
                    # 驗證階段
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
                    
                    print(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                training_time = time.time() - start_time
                final_val_acc = val_accs[-1]
                final_val_loss = val_losses[-1]

                # 繪製並儲存指標曲線圖
                epochs_range = range(1, num_epochs+1)
                plt.figure(figsize=(12,5))
                plt.subplot(1,2,1)
                plt.plot(epochs_range, train_losses, label='Train Loss')
                plt.plot(epochs_range, val_losses, label='Val Loss')
                plt.title(f'Loss (MAX_SEQ_LEN: {max_seq}, BS: {bs}, LR: {lr})')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.subplot(1,2,2)
                plt.plot(epochs_range, train_accs, label='Train Acc')
                plt.plot(epochs_range, val_accs, label='Val Acc')
                plt.title(f'Accuracy (MAX_SEQ_LEN: {max_seq}, BS: {bs}, LR: {lr})')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.tight_layout()
                exp_id = f"seq_{max_seq}_bs_{bs}_lr_{lr}"
                plot_path = os.path.join(RESULT_DIR, f"{exp_id}_metrics.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved metrics plot to {plot_path}")

                # 記錄本次實驗最終結果
                overall_results.append({
                    'MAX_SEQ_LEN': max_seq,
                    'Batch Size': bs,
                    'Learning Rate': lr,
                    'Num Epochs': num_epochs,
                    'Final Val Loss': final_val_loss,
                    'Final Val Acc': final_val_acc,
                    'Training Time (s)': training_time
                })

    # 將所有實驗結果存成 CSV
    results_df = pd.DataFrame(overall_results)
    results_csv_path = os.path.join(RESULT_DIR, "grid_search_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nGrid search results saved to {results_csv_path}")


def best_result():
    results_csv_path = os.path.join(RESULT_DIR, "grid_search_results.csv")
    if not os.path.exists(results_csv_path):
        print("未發現網格搜尋結果 CSV，請先執行網格搜尋。")
        return None
    df = pd.read_csv(results_csv_path)
    # 假設以驗證準確率最高為最佳模型
    best_row = df.loc[df['Final Val Acc'].idxmax()]
    return best_row

def main():
    parser = argparse.ArgumentParser(description="Model Training and Grid Search")
    parser.add_argument("--best", action="store_true", help="若設置此選項，則只印出最佳模型指標")
    args = parser.parse_args()

    if args.best:
        best_model = best_result()
        if best_model is not None:
            print("最好的模型指標:")
            print(best_model)
    else:
        grid_search()
    

if __name__ == "__main__":
    main()
