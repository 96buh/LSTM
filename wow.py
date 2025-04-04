import pandas as pd

data = pd.read_csv("./models/lstm/result/overall_experiment_log.csv")
print(data['post_train_acc'].max())