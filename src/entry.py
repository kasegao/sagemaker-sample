import argparse
import csv
import datetime
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import get_model
from params import HyperParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--hyper_params", type=str, default="{}")

    # Data directories
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    # Artifacts, /opt/ml/output/data
    parser.add_argument(
        "--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    # Container environment
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_known_args()


def model_fn(model_dir):
    """
    Load the model for inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(get_model())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
        train_inputs = torch.tensor(request)
        return train_inputs


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.float()).numpy()[0]


def train(args):
    """
    Train the PyTorch model
    """

    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")

    params: HyperParams = HyperParams.from_json(args.hyper_params)
    train_ds = get_dataset(os.path.join(args.data_dir, "train.csv"), params.dataset_params)
    test_ds = get_dataset(os.path.join(args.data_dir, "test.csv"), params.dataset_params)
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=params.batch_size, shuffle=False)

    logger.debug(
        "batch_size = {}, epochs = {}, learning rate = {}".format(
            params.batch_size, params.epochs, params.lr
        )
    )

    model = get_model().to(device)
    model = torch.nn.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    digit = len(str(params.epochs))
    train_loss = []
    test_loss = []
    for epoch in range(params.epochs):
        # train
        model.train()
        running_train_loss = 0.0
        with torch.set_grad_enabled(True):
            for data, target in train_dl:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                running_train_loss += loss.item()
                loss.backward()
                optimizer.step()
        train_loss.append(running_train_loss / len(train_ds))

        # eval
        running_test_loss, save_labels, save_outputs = test(
            model, test_dl, criterion, device
        )
        test_loss.append(running_test_loss)

        # logging
        logger.info(
            "epoch: {}, train_loss: {:.6f}, test_loss: {:.6f}, date: {}".format(
                str(epoch).zfill(digit),
                train_loss[-1],
                test_loss[-1],
                datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S"),
            )
        )

        # save results
        save_labels = np.concatenate(save_labels)
        save_outputs = np.concatenate(save_outputs)
        save_result = np.stack([save_labels, save_outputs], axis=0).T
        np.savetxt(
            os.path.join(args.output_dir, f"{str(epoch).zfill(digit)}.csv"),
            save_result,
            delimiter=",",
            header="labels,outputs",
            comments="",
        )

    result_loss = [[train_loss[i], test_loss[i]] for i in range(len(train_loss))]
    with open(os.path.join(args.output_dir, "loss.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss", "test_loss"])
        writer.writerows(result_loss)

    save_model(model, args.model_dir)


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    save_labels = []
    save_outputs = []
    with torch.set_grad_enabled(False):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            save_labels.append(target.to("cpu").detach().numpy().copy())
            save_outputs.append(output.to("cpu").detach().numpy().copy())

    test_loss /= len(test_loader.dataset)
    return test_loss, save_labels, save_outputs


if __name__ == "__main__":
    args, _ = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(args)
