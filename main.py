import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from GoDataset import GoDataset
# from ParmFinder import ParmFinder
from Trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 11032006
set_seed(seed)


goDataset = GoDataset("data/train/dan_train.csv")

config = {
    "input_dim": 19 * 19 * 3,
    "output_dim": 19 * 19,
    "num_heads": 3,
    "ffn_dim": 512,
    "num_layers": 3,
    "depthwise_conv_kernel_size": 3,
    "dropout": 0.1,
    "use_group_norm": False,
    "convolution_first": False,
    "lr": 0.00001,
    "gen_path": "data/models/gen",
    "dis_path": "data/models/dis",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "device": torch.device("cpu"),
    "batch_size": 25,
    "clip_value": 1,
    "data_len": goDataset.get_longest_game(),
    "epochs": 1,
    "early_stop": 1,
    "selected": 0
}

train_len = int(0.9 * len(goDataset))
val_len = len(goDataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(
    goDataset, [train_len, val_len]
)
train_loader = DataLoader(
    train_dataset,
    batch_size=int(config["batch_size"]),
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=int(config["batch_size"]), shuffle=False, pin_memory=True
)
trainer = Trainer(config, train_loader, val_loader)
statistic = trainer.run()


for key, value in statistic.items():
    plt.plot(value, label=key)
    plt.legend()
    plt.savefig(f"plot/{key}.png")
    plt.clf()
