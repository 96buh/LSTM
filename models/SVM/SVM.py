import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# 資料夾路徑設定與分類對應
LABEL_DIRS = {
    0: "./dataset/normal",
    1: "./dataset/abnormal/transformer_rust",
    2: "./dataset/abnormal/wire_rust",
    3: "./dataset/abnormal/wire_peeling"
}

RESULT_DIR = "./models/SVM/result"
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_SEQ_LEN = 10

def process_file(file_path, label, max_seq_len=MAX_SEQ_LEN):
    """
    讀取 CSV 檔案，取出 'current', 'voltage', 'power' 三個欄位，
    將數據切分為固定長度的片段，並將每個片段展平成一個一維向量。
    回傳該檔案的所有片段資料與對應的標籤。
    """
    df = pd.read_csv(file_path)
    current = df['current'].values
    voltage = df['voltage'].values
    power   = df['power'].values

    # 組合成 (N, 3) 的矩陣
    sequence = np.column_stack((current, voltage, power))
    seq_len = sequence.shape[0]
    num_chunks = seq_len // max_seq_len  # 僅取完整片段

    file_data = []
    file_labels = []
    for i in range(num_chunks):
        start = i * max_seq_len
        end = start + max_seq_len
        chunk = sequence[start:end]
        chunk_flat = chunk.flatten()  # 將 (max_seq_len, 3) 變為一維向量
        file_data.append(chunk_flat)
        file_labels.append(label)
    return file_data, file_labels

def load_all_data(max_seq_len=MAX_SEQ_LEN):
    """
    根據 LABEL_DIRS 中定義的各分類資料夾，讀取所有 CSV 檔案，
    並印出每個分類切分後的樣本數量。
    使用區域變數來累積所有資料與標籤，避免全域變數累積的問題。
    """
    all_data = []
    all_labels = []
    
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                file_data, file_labels = process_file(file_path, label, max_seq_len)
                all_data.extend(file_data)
                all_labels.extend(file_labels)

    data_array = np.array(all_data, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    scaler = StandardScaler()
    data_array = scaler.fit_transform(data_array)
    
    print("Data shape:", data_array.shape)
    print("Labels shape:", labels_array.shape)
    for i in sorted(LABEL_DIRS.keys()):
        print(f"Number of class {i} samples:", np.sum(labels_array == i))
    print("Unique labels:", np.unique(labels_array))
    return data_array, labels_array

def main():
    seq_list = [5, 10, 20, 40]
    for seq in seq_list: 
        print("===============")
        print(f"長度為 {seq}")
        data, labels = load_all_data(max_seq_len=seq)
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )

        model = SVC(kernel='rbf', C=1.0, gamma="scale", random_state=42)

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"⌛ Training time: {end_time - start_time:.2f} seconds")
        
        y_pred = model.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        # 如需要更詳細的評估結果，可啟用下行
        # print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
