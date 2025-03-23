import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import serial 
import time
from settings import *  

# LSTM 模型定義（與訓練時一致）
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=2, dropout_rate=0.3):
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
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最後時刻的輸出
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 載入訓練好的模型
def load_model(model_path, input_dim, hidden_dim, num_layers, num_classes, device):
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 設置為評估模式
    return model

# 實時數據處理
class RealTimePredictor:
    def __init__(self, model, max_seq_len, input_dim, device):
        self.model = model
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.device = device
        self.buffer = deque(maxlen=max_seq_len)  # 用於儲存最近 MAX_SEQ_LEN 筆數據
        
    def add_data(self, current, voltage, power):
        """將單筆數據加入緩衝區"""
        self.buffer.append([current, voltage, power])
        
    def predict(self):
        """當緩衝區滿時進行預測"""
        if len(self.buffer) < self.max_seq_len:
            return None  # 數據不足，無法預測
        
        # 轉換為模型輸入格式: (1, MAX_SEQ_LEN, input_dim)
        sequence = np.array(self.buffer, dtype=np.float32)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, MAX_SEQ_LEN, 3)
        
        # 預測
        with torch.no_grad():
            output = self.model(sequence_tensor)  # (1, num_classes)
            probabilities = torch.softmax(output, dim=1)  # 轉為機率
            _, predicted = torch.max(output, 1)  # 預測類別
            return predicted.item(), probabilities.cpu().numpy()[0]  # 返回預測標籤和機率

# 主程式
def main():
    # 參數設置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(RESULT_DIR, "fold_0_model.pth")  

    # 載入模型
    model = load_model(model_path, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, device)
    predictor = RealTimePredictor(model, MAX_SEQ_LEN, INPUT_DIM, device)

    # 假設單晶片通過串口傳送數據 (例如 COM3, 9600 baud rate)
    serial_port = 'COM3'  # 根據你的設備調整
    baud_rate = 9600
    ser = serial.Serial(serial_port, baud_rate, timeout=1)

    print("開始實時異常檢測... (按 Ctrl+C 結束)")
    try:
        while True:
            # 從單晶片讀取一行數據，假設格式為 "current,voltage,power" (例如 "1.2,3.5,4.2")
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    current, voltage, power = map(float, line.split(','))
                    predictor.add_data(current, voltage, power)
                    
                    # 預測
                    result = predictor.predict()
                    if result is not None:
                        predicted_label, probabilities = result
                        label_str = "異常" if predicted_label == 1 else "正常"
                        print(f"檢測結果: {label_str} | 正常機率: {probabilities[0]:.4f} | 異常機率: {probabilities[1]:.4f}")
                except ValueError as e:
                    print(f"數據格式錯誤: {line} | 錯誤: {e}")
            time.sleep(0.1)  
    except KeyboardInterrupt:
        print("結束檢測")
    finally:
        ser.close()  # 關閉串口

if __name__ == "__main__":
    main()