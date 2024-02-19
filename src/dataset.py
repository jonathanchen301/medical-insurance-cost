import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class InsuranceDataset(Dataset):

    def __init__(self, root, transform=None):
        # Data Loading
        dataset = pd.read_csv(root)
        self.nsamples = dataset.shape[0]
        self.transform = transform
        self.X = dataset.iloc[:, :-1]
        self.y = dataset.iloc[:, -1]

    def __getitem__(self, index):
        # Allow indexing (dataset[0])
        sample = (self.X.iloc[index, :], self.y.iloc[index])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.nsamples


class EncodingToTensor:
    def __call__(self, sample):
        x_copy = sample[0].copy()
        x_copy.replace(
            {
                "male": 0,
                "female": 1,
                "no": 0,
                "yes": 1,
                "southwest": 0,
                "southeast": 1,
                "northwest": 2,
                "northeast": 3,
            },
            inplace=True,
        )

        return torch.tensor(x_copy), torch.tensor(sample[1])
