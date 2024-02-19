from dataset import *

# Import the data

dataset = InsuranceDataset("data/insurance.csv", transform=EncodingToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
