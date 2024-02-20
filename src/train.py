from dataset import *
from sklearn.model_selection import train_test_split
import wandb

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.01,
#         "architecture": "Linear Regression",
#         "dataset": "insurance.csv",
#         "epochs": 10,
#     },
# )

# 80-10-10 split

dataset = InsuranceDataset("data/insurance.csv", transform=EncodingToTensor())

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
test_data, dev_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
