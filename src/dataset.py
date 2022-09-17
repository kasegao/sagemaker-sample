import numpy as np
import torch
from torch.utils.data import Dataset

from params import DatasetParams


class MyDataset(Dataset):
    def __init__(self, csv_path: str, dataset_params: DatasetParams = DatasetParams()):
        self.cav_path = csv_path
        self.dataset = np.loadtxt(
            csv_path,
            delimiter=",",
            skiprows=dataset_params.skiprows,
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
        Returns
        -------
        (torch.Tensor([a, b]), a+b)
        """

        row = self.dataset[index]
        x = torch.from_numpy(row[:2]).clone()
        return x, row[2]



def get_dataset(csv_path: str, dataset_params: DatasetParams) -> MyDataset:
    return MyDataset(csv_path, dataset_params)
