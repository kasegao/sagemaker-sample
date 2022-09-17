from params import DatasetParams, HyperParams


def test_model_params():
    dataset_params = DatasetParams(skiprows=5)
    assert dataset_params.to_json() == '{"skiprows": 5}'


def test_hyper_params():
    dataset_params = DatasetParams(skiprows=5)
    hyper_params = HyperParams(
        epochs=1, batch_size=2, lr=0.1, dataset_params=dataset_params
    )
    assert (
        hyper_params.to_json()
        == '{"epochs": 1, "batch_size": 2, "lr": 0.1, "dataset_params": {"skiprows": 5}}'
    )
