import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import argparse
import time

from settings import *

os.makedirs(RESULT_DIR, exist_ok=True)


# === 資料處理 ===
def process_file(file_path, label, sequences, labels, max_seq_len=MAX_SEQ_LEN):
    """
    讀取 CSV 檔案並切分為長度為 max_seq_len 的小段。
    :param file_path: CSV檔案路徑
    :param label: 資料標籤（例如：0、1、2、3）
    :param sequences: 儲存切分後片段的 list
    :param labels: 儲存對應標籤的 list
    """
    df = pd.read_csv(file_path)
    # 假設 CSV 中有 'current', 'voltage', 'power' 三個欄位
    current = df['current'].values
    voltage = df['voltage'].values
    power = df['power'].values

    sequence = np.column_stack((current, voltage, power))  # shape: (N, 3)
    seq_len = sequence.shape[0]
    num_chunks = seq_len // max_seq_len
    for i in range(num_chunks):
        start = i * max_seq_len
        end = start + max_seq_len
        chunk = sequence[start:end]
        sequences.append(chunk)
        labels.append(label)


def load_data():
    sequences = []
    labels = []
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                process_file(file_path, label=label, sequences=sequences, labels=labels)
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print("sequences shape:", sequences.shape)
    print("labels shape:", labels.shape)
    for i in range(NUM_CLASSES):
        print(f"Number of class {i} samples:", np.sum(labels == i))
    print("Unique labels:", np.unique(labels))
    return sequences, labels


# === 自訂 Dataset ===
class ChargingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# === LSTM 模型定義 ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]   # 使用最後一個時間步的輸出
        out = self.dropout(out)
        out = self.fc(out)
        return out


# === 評估函數 ===
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = running_loss / total
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc


# === K-fold 訓練流程 ===
def kfold_training(sequences, labels):
    dataset = ChargingDataset(sequences, labels)
    kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fold_final_metrics = []
    all_folds_metrics = []  # 儲存每個 fold 的所有 epoch 指標 DataFrame

    fold_idx = 0
    for train_indices, test_indices in kfold.split(sequences, labels):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx} / {KFOLD_SPLITS} ===")
        # 切分訓練與測試資料
        train_sequences, train_labels = sequences[train_indices], labels[train_indices]
        test_sequences, test_labels = sequences[test_indices], labels[test_indices]

        train_dataset = ChargingDataset(train_sequences, train_labels)
        test_dataset = ChargingDataset(test_sequences, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 建立模型（未訓練狀態）
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 訓練前：用初始模型在測試集上的表現作為基線（Pre-train Metrics）
        pre_test_loss, pre_test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Pre-training  | Test Loss: {pre_test_loss:.4f}, Test Acc: {pre_test_acc:.4f}")
        print("===========================================")

        # 記錄各 epoch 指標
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        test_precision_list = []
        test_recall_list = []
        test_f1_list = []

        start_time = time.time()
        # 開始訓練
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total if total > 0 else 0
            epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader, criterion, device)

            train_loss_list.append(epoch_train_loss)
            train_acc_list.append(epoch_train_acc)
            test_loss_list.append(epoch_test_loss)
            test_acc_list.append(epoch_test_acc)

            # 計算其他指標（以 macro 平均）
            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs, 1)
                    preds.extend(predicted.cpu().numpy())
                    trues.extend(y_batch.cpu().numpy())
            test_precision = precision_score(trues, preds, average='macro', zero_division=0)
            test_recall = recall_score(trues, preds, average='macro', zero_division=0)
            test_f1 = f1_score(trues, preds, average='macro', zero_division=0)

            test_precision_list.append(test_precision)
            test_recall_list.append(test_recall)
            test_f1_list.append(test_f1)

            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f} | "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        end_time = time.time()
        print(f"⌛訓練時間：{end_time - start_time:.2f} 秒")

        # 訓練結束後，取得訓練後（Post-train）的測試集表現
        post_test_loss, post_test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Post-training | Test Loss: {post_test_loss:.4f}, Test Acc: {post_test_acc:.4f}")

        fold_final_metrics.append({
            'Fold': fold_idx,
            'Pre-train Loss': pre_test_loss,
            'Pre-train Accuracy': pre_test_acc,
            'Post-train Loss': post_test_loss,
            'Post-train Accuracy': post_test_acc,
            'Precision': test_precision_list[-1],
            'Recall': test_recall_list[-1],
            'F1-Score': test_f1_list[-1]
        })

        # 混淆矩陣 (使用所有類別)
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                trues.extend(y_batch.cpu().numpy())
        cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(NUM_CLASSES)))
        plt.figure(figsize=(4,4))
        disp.plot(values_format='d', cmap='Blues')
        plt.title(f"Fold {fold_idx} - Confusion Matrix")
        cm_pdf_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.pdf")
        cm_svg_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.svg")
        plt.savefig(cm_pdf_path, bbox_inches='tight')
        plt.savefig(cm_svg_path, bbox_inches='tight')
        plt.show()

        # 儲存模型
        model_save_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for fold {fold_idx} saved to {model_save_path}")

        # 將該 fold 的所有 epoch 指標記錄成 DataFrame
        epochs_range = range(1, NUM_EPOCHS+1)
        fold_metrics_df = pd.DataFrame({
            'Epoch': epochs_range,
            'Train Loss': train_loss_list,
            'Test Loss': test_loss_list,
            'Train Accuracy': train_acc_list,
            'Test Accuracy': test_acc_list,
            'Test Precision': test_precision_list,
            'Test Recall': test_recall_list,
            'Test F1-score': test_f1_list
        })
        fold_metrics_df['Fold'] = fold_idx
        all_folds_metrics.append(fold_metrics_df)

    final_metrics_df = pd.DataFrame(fold_final_metrics)
    print("\n=== Final Metrics Summary Across All Folds ===")
    print(final_metrics_df.to_string(index=False))
    return final_metrics_df, all_folds_metrics


def plot_metric_curves(all_folds_metrics):
    for fold_df in all_folds_metrics:
        fold = fold_df['Fold'].iloc[0]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Fold {fold} Metrics Curves', fontsize=16)
        epochs = fold_df['Epoch']
        
        # Train Loss
        axes[0, 0].plot(epochs, fold_df['Train Loss'], label='Train Loss')
        axes[0, 0].set_title('Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Test Loss
        axes[0, 1].plot(epochs, fold_df['Test Loss'], label='Test Loss', color='orange')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Train Accuracy
        axes[0, 2].plot(epochs, fold_df['Train Accuracy'], label='Train Accuracy', color='green')
        axes[0, 2].set_title('Train Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        
        # Test Accuracy
        axes[1, 0].plot(epochs, fold_df['Test Accuracy'], label='Test Accuracy', color='red')
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Test Precision
        axes[1, 1].plot(epochs, fold_df['Test Precision'], label='Test Precision', color='purple')
        axes[1, 1].set_title('Test Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        
        # Test F1-score
        axes[1, 2].plot(epochs, fold_df['Test F1-score'], label='Test F1-score', color='brown')
        axes[1, 2].set_title('Test F1-score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-score')
        axes[1, 2].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_pdf = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.pdf")
        plot_svg = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.svg")
        plt.savefig(plot_pdf, bbox_inches='tight')
        plt.savefig(plot_svg, bbox_inches='tight')
        plt.close(fig)


def count_chunks_in_folder(folder_path, max_seq_len=MAX_SEQ_LEN):
    """
    計算指定資料夾內所有 CSV 檔案，依據 max_seq_len 切分後的總片段數量。
    """
    total_chunks = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)
            chunks = num_rows // max_seq_len
            total_chunks += chunks
    return total_chunks

def plot_overlaid_metrics(all_folds_metrics):
    # 假設所有 fold 的 epoch 數量相同
    epochs = all_folds_metrics[0]['Epoch']
    
    # 定義你想疊加繪製的指標，key 為圖形標題、value 為 DataFrame 中的欄位名稱
    metrics = {
        'Train Accuracy': 'Train Accuracy',
        'Test Accuracy': 'Test Accuracy',
        'Train Loss': 'Train Loss',
        'Test Loss': 'Test Loss',
        'Test Precision': 'Test Precision',
        'Test Recall': 'Test Recall',
        'Test F1-score': 'Test F1-score'
    }
    
    for metric_title, col in metrics.items():
        plt.figure(figsize=(10, 6))
        for fold_df in all_folds_metrics:
            fold = fold_df['Fold'].iloc[0]
            plt.plot(epochs, fold_df[col], label=f'Fold {fold}')
        plt.title(f'Combined {metric_title}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_title)
        plt.legend()
        plt.tight_layout()
        
        # 儲存圖形
        pdf_path = os.path.join(RESULT_DIR, f"combined_{col}.pdf")
        svg_path = os.path.join(RESULT_DIR, f"combined_{col}.svg")
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Model Training and Data Count Check")
    parser.add_argument("--check-data", action="store_true", help="如果設置此選項，則只執行資料數量統計")
    args = parser.parse_args()
    if args.check_data:
        print(f"Max seq len = {MAX_SEQ_LEN}")
        # 檢查各類別資料的片段數量
        for label, folder in LABEL_DIRS.items():
            count = count_chunks_in_folder(folder)
            print(f"Label {label} ({folder}): {count} chunks")
    else:
        all_sequences, all_labels = load_data()
        final_metrics_df, all_folds_metrics = kfold_training(all_sequences, all_labels)
        # 繪製每個 fold 的指標曲線圖
        plot_metric_curves(all_folds_metrics)
        plot_overlaid_metrics(all_folds_metrics)
