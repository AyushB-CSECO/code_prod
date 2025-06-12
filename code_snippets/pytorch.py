import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Create a pytorch dataset
    """
    def __init__(self, dataframe,y_col):
        self.data = dataframe
        self.labels = dataframe[y_col].values 
        self.features = dataframe[[i for i in dataframe.columns if i!=y_col]].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label