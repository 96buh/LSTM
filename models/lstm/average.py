import os
import pandas as pd
import matplotlib.pyplot as plt

path = "./models/lstm/result/seq_len_test"
dir_list = sorted(os.listdir(path))
records = []

for d in dir_list:
    csv_path = os.path.join(path, d, "summary_metrics.csv")
    if not os.path.isfile(csv_path):
        continue
    data = pd.read_csv(csv_path)
    mean_ser = data.mean()[1:]
    # 將 mean_ser 轉成 dict，並加入一個新的 'seq_folder' 欄位
    rec = mean_ser.to_dict()
    rec["seq_folder"] = d
    records.append(rec)

summary_df = pd.DataFrame(records)
print(summary_df)
plt.figure(figsize=(12,8))

plt.subplot(331)
plt.bar(summary_df['seq_folder'], summary_df['train_loss'])
plt.title("train loss")

plt.subplot(332)
plt.bar(summary_df['seq_folder'], summary_df['test_loss'])
plt.title("test loss")

plt.subplot(333)
plt.bar(summary_df['seq_folder'], summary_df['train_acc'])
plt.title("train acc")

plt.subplot(334)
plt.bar(summary_df['seq_folder'], summary_df['test_acc'])
plt.title("test acc")

plt.subplot(335)
plt.bar(summary_df['seq_folder'], summary_df['precision'])
plt.title("precision")

plt.subplot(336)
plt.bar(summary_df['seq_folder'], summary_df['recall'])
plt.title("recall")

plt.subplot(337)
plt.bar(summary_df['seq_folder'], summary_df['f1_score'])
plt.title("f1 score")

plt.tight_layout()
plt.savefig("TEST.png")

print(f"最低loss: {summary_df['test_loss'].min()}")