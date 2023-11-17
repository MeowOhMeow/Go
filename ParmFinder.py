import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from GoDataset import GoDataset
from Trainer import Trainer

domain = {
    "input_dim": [2 * 19 * 19],
    "num_heads": [1, 2],
    "ffn_dim": [64, 128, 256, 512],
    "num_layers": [2, 4, 8],
    "depthwise_conv_kernel_size": [3, 5, 7],
    "dropout": [0, 0.1, 0.2, 0.3, 0.4],
    "use_group_norm": [True, False],
    "convolution_first": [True, False],
    "lr": [0.0001, 0.001, 0.01],
    "gen_path": ["data/models/gen.pth"],
    "dis_path": ["data/models/dis.pth"],
    "device": [torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    "batch_size": [128],
    "clip_value": [1],
    "data_len": [4, 8, 16, 32],
    "epochs": [10],
    "early_stop": [5],
}

# # print domain count
# count = 1
# for key, value in domain.items():
#     count *= len(value)
# print(f"Total combinations: {count}")


class ParmFinder:
    def __init__(self, domain: dict) -> None:
        # Initialize parameters and data structures
        self.best_ratio = float("inf")
        self.best_params = None
        self.G_parms = []
        self.G_train_losses = []
        self.G_ratios = []
        self.D_parms = []
        self.domain = domain
        self.max_iter = 500
        self.max_epoch = 5
        self.G_history_path = "data/G_history.csv"

    def __random_sample(self):
        # Randomly sample parameters from the given domain
        params = {}
        for key, value in self.domain.items():
            params[key] = np.random.choice(value)

        # print(f"Current params: {params}")

        goDataset = GoDataset("data/train/dan_train.csv", params["data_len"])
        train_len = int(0.8 * len(goDataset))
        val_len = len(goDataset) - train_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            goDataset, [train_len, val_len]
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=int(params["batch_size"]), shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=int(params["batch_size"]), shuffle=False, pin_memory=True
        )

        return params

    def __save_G(self):
        # Save G_parms, G_train_losses, and G_ratios to a CSV file and best G model
        header = list(self.domain.keys()) + ["train_loss", "loss_ratio"]
        df = pd.DataFrame(self.G_parms, columns=header)
        df["train_loss"] = self.G_train_losses
        df["loss_ratio"] = self.G_ratios
        df.sort_values(by="loss_ratio", ascending=False, inplace=True)
        df.to_csv(self.G_history_path, index=False)

    def __evaluate_G(self, trainer: Trainer):
        # Evaluate generator performance over multiple epochs
        train_loss = 0
        loss_ratio = 0
        for epoch in range(self.max_epoch):
            G_losses = []
            G_val_losses = []
            trainer.normal_train_G(G_losses)
            trainer.normal_evaluate_G(G_val_losses)

            train_loss = np.mean(G_losses)
            val_loss = np.mean(G_val_losses)
            loss_ratio = train_loss / val_loss

            print(
                f"Epoch {epoch+1}/{self.max_epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Loss Ratio: {loss_ratio}"
            )
        
        if loss_ratio < self.best_ratio:
            self.best_ratio = loss_ratio
            self.best_params = trainer.config
                
        self.G_parms.append(trainer.config)
        self.G_train_losses.append(train_loss)
        self.G_ratios.append(loss_ratio)
        self.__save_G()

    def find(self):
        # Iterate for a maximum number of iterations
        for _ in range(self.max_iter):
            params = self.__random_sample()
            trainer = Trainer(params, self.train_loader, self.val_loader)
            self.__evaluate_G(trainer)
            torch.cuda.empty_cache()
            gc.collect()

        return self.best_params, self.train_loader, self.val_loader
