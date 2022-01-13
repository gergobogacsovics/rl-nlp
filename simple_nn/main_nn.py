# %%
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from net import Network, NetworkLSTM
from dataset import DNNDataset, DNNDatasetPerfectYs, RNNDatasetPerfectYs, DNNDataset_90Percentage, DNNDataset_80Percentage
from tqdm import tqdm
import sys

sys.path.append('..')
from util_tensorboard import TensorboardLoggerSimple

CONFIG = {
    "mode": "train",  # hptuning, train, test,
    "tag": "windowsize=5",
    "train": {
        "lr": 0.001, # 0.001, # 3e-4,
        "train_csv": "../../data/BloombergNRG_train.csv",
        "val_csv": "../../data/BloombergNRG_train.csv",
        "train_bs": 32,
        "val_bs": 32,
        "n_epochs": 250
    }
}

_dataset_class = DNNDataset
window_size=5 # DNN-nél: 5, RNN: 1 (because that is rolled)

logger = TensorboardLoggerSimple(log_dir="tb_logs", run_name=f'{CONFIG["mode"]}_{CONFIG["tag"]}')


# %%
def hptuning():
    def schedule_lr(e):
        return 1e-5 * (10 ** (e / 20))

    def update_learning_rate(optimizer_, new_lr):
        for param_group in optimizer_.param_groups:
            param_group['lr'] = new_lr

    epochs = list(range(100))
    lrs = [schedule_lr(e) for e in epochs]

    train_dataset = _dataset_class(csv_file=CONFIG["train"]["train_csv"], window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train"]["train_bs"],
                              shuffle=True)

    total_steps_train_dataset = len(train_dataset)

    net = Network(feature_dim=window_size, out_dim=1)
    net.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)

    for epoch in range(0, 99 + 1):
        update_learning_rate(optimizer, lrs[epoch])

        total_loss_train = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item() * inputs.size(0)

        avg_loss_train = total_loss_train / total_steps_train_dataset

        print(
            f"Epoch [{epoch}/100] Avg Train Loss: {avg_loss_train}")

        logger.write_metadata(epoch=epoch, key="loss", value=total_loss_train)
        logger.write_metadata(epoch=epoch, key="avg_loss", value=avg_loss_train)


# %%
def train():
    train_dataset = _dataset_class(csv_file=CONFIG["train"]["train_csv"], window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["train"]["train_bs"],
                              shuffle=True)

    val_dataset = _dataset_class(csv_file=CONFIG["train"]["val_csv"], window_size=window_size)
    val_loader = DataLoader(train_dataset, batch_size=CONFIG["train"]["val_bs"],
                            shuffle=False)

    total_steps_train_dataset = len(train_dataset)
    total_steps_val_dataset = len(val_dataset)

    net = Network(feature_dim=window_size, out_dim=1)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=CONFIG["train"]["lr"])

    for epoch in range(1, CONFIG["train"]["n_epochs"] + 1):
        net.train()

        total_loss_train = 0
        label_output_matches_train = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item() * inputs.size(0)

            with torch.no_grad():
                predictions = outputs > 0.5
                label_output_matches_train += (predictions == labels).sum().item()

        avg_loss_train = total_loss_train / total_steps_train_dataset
        acc_train = label_output_matches_train / total_steps_train_dataset

        net.eval()

        total_loss_val = 0
        label_output_matches_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.float(), labels.float()

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                total_loss_val += loss.item() * inputs.size(0)

                # _, predictions = torch.max(outputs, 1)
                predictions = outputs > 0.5
                label_output_matches_val += (predictions == labels).sum().item()

        avg_loss_val = total_loss_val / total_steps_val_dataset
        acc_val = label_output_matches_val / total_steps_val_dataset

        print(
            f"Epoch [{epoch}/{CONFIG['train']['n_epochs']}] Avg Train Loss: {avg_loss_train}, Train Acc: {acc_train}, Avg Val Loss: {avg_loss_val}, Val Acc: {acc_val}")

        logger.write_metadata(epoch=epoch, key="training_loss", value=total_loss_train)
        logger.write_metadata(epoch=epoch, key="training_avg_loss", value=avg_loss_train)
        logger.write_metadata(epoch=epoch, key="training_ACC", value=acc_train)
        logger.write_metadata(epoch=epoch, key="validation_loss", value=total_loss_val)
        logger.write_metadata(epoch=epoch, key="validation_avg_loss", value=avg_loss_val)
        logger.write_metadata(epoch=epoch, key="validation__ACC", value=acc_val)

    print('Finished Training')


# %%
def test():
    test_dataset = _dataset_class(csv_file=CONFIG["test"]["csv"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["test"]["bs"],
                             shuffle=False)

    total_steps_test_dataset = len(test_dataset)

    net = Network()
    # TODO: Weight loading
    net.eval()

    label_matches = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            outputs = net(inputs)

            _, predictions = torch.max(outputs, 1)
            label_matches += (predictions == labels).sum().item()

        acc = label_matches / total_steps_test_dataset
        print("ACC: ", acc)

        logger.write_metadata(epoch=0, key="acc", value=acc)


# %%
if CONFIG["mode"] == "hptuning":
    print("Entering hyperparameter-tuning mode.")
    hptuning()
elif CONFIG["mode"] == "train":
    print("Entering training mode.")
    train()
elif CONFIG["mode"] == "test":
    print("Entering test mode.")
    test()
else:
    print(f"Error. Unknown mode '{CONFIG['mode']}'.")
    exit(-1)
