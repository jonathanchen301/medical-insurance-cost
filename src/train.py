from dataset import *
from model import *
from sklearn.model_selection import train_test_split

# wandb.init(
#     project="medical-insurance",
#     config={
#         "learning_rate": 0.0001,
#         "architecture": "Linear Regression",
#         "dataset": "insurance.csv",
#         "epochs": 200,
#     },
# )

dataset = InsuranceDataset("data/insurance.csv", transform=EncodingToTensor())

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
test_data, dev_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
dev_dataloader = DataLoader(dev_data, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

lr = 0.0001
num_epochs = 200

model = LinearRegression(next(iter(train_dataloader))[0].size(1))
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_ft = torch.nn.MSELoss()

model.learn(train_dataloader, dev_dataloader, num_epochs, optimizer, loss_ft)
model.evaluate(test_dataloader, loss_ft)
