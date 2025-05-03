import matplotlib.pyplot as plt
import pandas as pd



normal_data = pd.read_csv("dataset/normal/70.csv")
# print(normal_voltage['voltage'][:30])
wire_rust_data = pd.read_csv("dataset/abnormal/wire_rust/326_頭正常線生鏽.csv")
wire_peeling_data = pd.read_csv("dataset/abnormal/wire_peeling/228_頭正常線剝落.csv")
transformer_rust_data = pd.read_csv("dataset/abnormal/transformer_rust/70_開螢幕_變壓器生鏽.csv")

columns = ['voltage', 'current', 'power']

for col in columns:
    plt.figure(figsize=(10, 6))
    plt.plot(normal_data[col][:30], label="normal", color="green")
    plt.plot(wire_rust_data[col][:30], label="wire rust", color="blue", linestyle="--")
    plt.plot(wire_peeling_data[col][:30], label="wire peeling", color="orange", linestyle=":")
    plt.plot(transformer_rust_data[col][:30], label="transformer rust", color="purple", linestyle="-.")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(col)
    plt.grid(alpha=0.3)

    plt.savefig(f'{col}.png', bbox_inches='tight')