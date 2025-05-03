import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 只從 settings 匯入超參數
from settings import (
    SEED, NUM_CLASSES, KFOLD_SPLITS,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
    RESULT_DIR, LABEL_DIRS
)

# 固定隨機種子
np.random.seed(SEED)
torch.manual_seed(SEED)

# === 資料切分與平衡 ===
def process_file(path, label, seqs, labs, max_len):
    df = pd.read_csv(path)
    arr = np.column_stack((df['current'].values,
                           df['voltage'].values,
                           df['power'].values))
    n_chunks = len(arr) // max_len
    for i in range(n_chunks):
        seqs.append(arr[i*max_len:(i+1)*max_len])
        labs.append(label)

def load_and_balance(max_len):
    seqs, labs = [], []
    for lbl, folder in LABEL_DIRS.items():
        for fn in os.listdir(folder):
            if fn.lower().endswith('.csv'):
                process_file(os.path.join(folder, fn), lbl, seqs, labs, max_len)
    X = np.array(seqs, dtype=np.float32)
    y = np.array(labs, dtype=np.int64)
    # 平衡各類別至最少樣本數
    counts = [np.sum(y==i) for i in range(NUM_CLASSES)]
    m = min(counts)
    idxs = np.hstack([
        np.random.choice(np.where(y==i)[0], m, replace=False)
        for i in range(NUM_CLASSES)
    ])
    np.random.shuffle(idxs)
    return X[idxs], y[idxs]

# === Dataset 與 Model ===
class ChargingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM, device=x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.drop(out[:, -1, :])
        return self.fc(out)

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

# === K-Fold 實驗 ===
def kfold_experiment(X, y, seq_len):
    skf = StratifiedKFold(KFOLD_SPLITS, shuffle=True, random_state=SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n>>> Fold {fold} / {KFOLD_SPLITS} (seq_len={seq_len})")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
        X_tr = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        X_te = scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

        train_loader = DataLoader(ChargingDataset(X_tr, y_tr),
                                  batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(ChargingDataset(X_te, y_te),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = LSTMClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # epoch-level記錄
        epochs = list(range(1, NUM_EPOCHS+1))
        t_loss, t_acc = [], []
        v_loss, v_acc = [], []
        prec_list, rec_list, f1_list = [], [], []

        for ep in epochs:
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            tl, ta = evaluate(model, train_loader, criterion, device)
            vl, va = evaluate(model, test_loader,  criterion, device)

            # precision/recall/f1 on 測試集
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb = Xb.to(device)
                    out = model(Xb).argmax(1).cpu().numpy()
                    preds.extend(out); trues.extend(yb.numpy())
            prec = precision_score(trues, preds, average='macro', zero_division=0)
            rec  = recall_score(   trues, preds, average='macro', zero_division=0)
            f1   = f1_score(       trues, preds, average='macro', zero_division=0)

            # 存到 list
            t_loss.append(tl);   t_acc.append(ta)
            v_loss.append(vl);   v_acc.append(va)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)

            # 即時印出
            print(f"Epoch {ep:2d}/{NUM_EPOCHS} | "
                  f"TrainLoss={tl:.4f}, TestLoss={vl:.4f} | "
                  f"TrainAcc={ta:.4f}, TestAcc={va:.4f} | "
                  f"Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # 組成 DataFrame
        df = pd.DataFrame({
            'epoch':      epochs,
            'train_loss': t_loss,
            'test_loss':  v_loss,
            'train_acc':  t_acc,
            'test_acc':   v_acc,
            'precision':  prec_list,
            'recall':     rec_list,
            'f1_score':   f1_list
        })

        # 存 csv
        fold_dir = os.path.join(RESULT_DIR, 'seq_len_test', f'seq{seq_len}', f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        df.to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)

        # 畫三張子圖：loss, accuracy, prf
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        # Loss
        axes[0].plot(epochs, df['train_loss'], label='Train Loss')
        axes[0].plot(epochs, df['test_loss'],  label='Test Loss')
        axes[0].set_title('Loss'); axes[0].legend()
        # Accuracy
        axes[1].plot(epochs, df['train_acc'], label='Train Acc')
        axes[1].plot(epochs, df['test_acc'],  label='Test Acc')
        axes[1].set_title('Accuracy'); axes[1].legend()
        # Precision/Recall/F1
        axes[2].plot(epochs, df['precision'], label='Precision')
        axes[2].plot(epochs, df['recall'],    label='Recall')
        axes[2].plot(epochs, df['f1_score'],  label='F1-score')
        axes[2].set_title('Precision / Recall / F1'); axes[2].legend()

        plt.tight_layout()
        fig.savefig(os.path.join(fold_dir, 'metrics_plots.pdf'))
        plt.close(fig)

        # 混淆矩陣
        cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix'); plt.colorbar()
        plt.xlabel('Pred'); plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i,j], ha='center', va='center')
        plt.savefig(os.path.join(fold_dir, 'confusion_matrix.pdf'))
        plt.close()

        # 收集最終 epoch 指標
        fold_results.append({
            'fold':       fold,
            'train_loss': t_loss[-1],
            'test_loss':  v_loss[-1],
            'train_acc':  t_acc[-1],
            'test_acc':   v_acc[-1],
            'precision':  prec_list[-1],
            'recall':     rec_list[-1],
            'f1_score':   f1_list[-1]
        })

    return pd.DataFrame(fold_results)

# === 主程式 ===
if __name__ == '__main__':
    seq_lengths = [5, 10, 15, 20, 30, 40]  # 可自行調整
    for seq in seq_lengths:
        print(f"\n=== 開始 seq_len = {seq} 的實驗 ===")
        X, y = load_and_balance(seq)
        summary = kfold_experiment(X, y, seq)
        summary_dir = os.path.join(RESULT_DIR, 'seq_len_test', f'seq{seq}')
        summary.to_csv(os.path.join(summary_dir, 'summary_metrics.csv'), index=False)
        print(f"已儲存 summary_metrics.csv 到 {summary_dir}")
