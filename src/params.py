from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DatasetParams:
    skiprows: int = 1


@dataclass_json
@dataclass
class HyperParams:
    epochs: int = 10
    batch_size: int = 8
    lr: float = 0.001
    dataset_params: DatasetParams = DatasetParams()
