"""
一次讀取全部在dataset的csv檔案，並繪製圖表儲存在對應的figures資料夾中。
沒事不要執行，不然重複的檔案會有編號(1)、(2)、(3)、...。
"""
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 設置 Matplotlib 與 LaTeX 的兼容性
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
})

# 設置圖表尺寸（基於 LaTeX 列寬）
width_pt = 345  # 替換為您的列寬，例如 \textwidth 的值
inches_per_pt = 1.0 / 72.27  # 點數轉英寸
golden_ratio = (5**0.5 - 1) / 2  # 黃金比例
width_in = width_pt * inches_per_pt
height_in = width_in * golden_ratio

def get_unique_filename(folder, filename):
    """
    如果檔案名稱已存在，則添加編號 (1), (2), ... 以生成唯一檔案名稱。
    
    參數：
        folder (str): 目標資料夾路徑。
        filename (str): 原始檔案名稱（包含副檔名）。
    
    回傳：
        str: 唯一的檔案名稱。
    """
    base_name, ext = os.path.splitext(filename)  # 分離檔案名和副檔名
    new_filename = filename
    counter = 1
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{base_name}({counter}){ext}"
        counter += 1
    return new_filename

def plot_and_save_csv(csv_path, category, figures_base_dir="./figures"):
    """
    讀取 CSV 檔案並繪製圖表，然後保存為 PDF。
    
    參數：
        csv_path (str): CSV 檔案的完整路徑。
        category (str): 'normal' 或 'abnormal'，決定儲存圖表的子資料夾。
        figures_base_dir (str): 基本圖表儲存資料夾。
    """
    # 確定目標儲存資料夾
    output_dir = os.path.join(figures_base_dir, category)
    os.makedirs(output_dir, exist_ok=True)  # 確保目標資料夾存在
    
    # 獲取原始檔案名稱
    original_filename = os.path.basename(csv_path)
    
    # 去掉 .csv 後綴並生成 PDF 檔案名稱
    base_filename = os.path.splitext(original_filename)[0]
    pdf_filename = f"{base_filename}.pdf"
    
    # 生成唯一的 PDF 檔案名稱
    unique_pdf_filename = get_unique_filename(output_dir, pdf_filename)
    output_pdf_path = os.path.join(output_dir, unique_pdf_filename)
    
    # 讀取資料
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"讀取 {csv_path} 時發生錯誤: {e}")
        return
    
    # 繪製圖表
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.plot(data.index, data['voltage'], label='Voltage (V)')
    ax.plot(data.index, data['current'], label='Current (A)')
    ax.plot(data.index, data['power'], label='Power (W)')
    
    # 添加標題和標籤
    ax.set_title('Voltage, Current, and Power Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    # 將圖例放置在下方，水平排列，並去除外框
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    # 僅保留 Y 軸網格
    ax.grid(axis='y', alpha=0.7)
    
    # 自動調整圖表邊界以適應圖例
    fig.tight_layout()
    
    # 儲存圖表為 PDF
    fig.savefig(output_pdf_path, bbox_inches='tight')
    plt.close(fig)  # 關閉圖表以釋放記憶體
    print(f"圖表已保存為 {output_pdf_path}")

def main():
    """
    主程式，遍歷 dataset/normal 和 dataset/abnormal 資料夾中的所有 CSV 檔案，
    並繪製圖表儲存在對應的 figures/normal 或 figures/abnormal 資料夾中。
    """
    dataset_dir = "./dataset"
    categories = ["normal", "abnormal"]
    figures_dir = "./figures"
    
    # 確保 figures 的基礎資料夾存在
    os.makedirs(figures_dir, exist_ok=True)
    
    for category in categories:
        csv_folder = os.path.join(dataset_dir, category)
        if not os.path.isdir(csv_folder):
            print(f"警告：資料夾 {csv_folder} 不存在，跳過。")
            continue
        
        # 使用 glob 搜尋所有 CSV 檔案
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        
        print(f"正在處理 {category} 資料夾中的 {len(csv_files)} 個 CSV 檔案...")
        
        for csv_file in csv_files:
            # 獲取原始檔案名稱
            original_filename = os.path.basename(csv_file)
            plot_and_save_csv(csv_file, category, figures_base_dir=figures_dir)
        
        print(f"已完成處理 {category} 資料夾。\n")

if __name__ == "__main__":
    main()
