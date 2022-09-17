import numpy as np
import pytest
import torch
from dataset import MyDataset
from params import DatasetParams


@pytest.fixture(scope="session")
def csv_file(tmpdir_factory):
    arg = np.arange(4).reshape((-1, 2))
    ans = np.sum(arg, axis=1)
    data = np.concatenate([arg, ans.reshape((-1, 1))], axis=1)
    filename = str(tmpdir_factory.mktemp("testdata").join("data.csv"))
    np.savetxt(filename, data, delimiter=",", header="a,b,ans", comments="")
    return filename


def test_mydataset(csv_file: str):
    ds = MyDataset(csv_file, DatasetParams(skiprows=1))
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype=np.float32)
    x, y = ds[0]
    want_x = torch.from_numpy(data[0, :2]).clone()
    assert torch.eq(x, want_x).all()
    assert y == data[0, 2]
