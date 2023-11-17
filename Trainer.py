import datetime
import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Predictor import Predictor


class Trainer:
    def __init__(
        self,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        version: str = "v1",
        load_model: bool = False,
        from_epoch: int = 0,
        path: str = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.from_epoch = from_epoch
        self.version = version
        
        if not os.path.exists(f"models/{version}"):
            os.makedirs(f"models/{version}")

        if load_model:
            self.pre = torch.load(path)
        else:
            self.pre = Predictor(
                input_dim=config["input_dim"],
                max_len=config["data_len"],
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                num_layers=config["num_layers"],
                depthwise_conv_kernel_size=config["depthwise_conv_kernel_size"],
                dropout=config["dropout"],
                use_group_norm=config["use_group_norm"],
                convolution_first=config["convolution_first"],
            )
        self.pre.to(self.config["device"])

        self.optimizer = torch.optim.Adam(self.pre.parameters(), lr=config["lr"])

        self.criterion = nn.MSELoss()

        self.clip_value = config["clip_value"]

        self.early_count = 0
        self.best_val_loss = float("inf")

    def evaluate(self, val_losses: list):
        print(f"Evaluating:")

        self.pre.eval()

        total_loss = 0
        total_correct = 0

        for i, (x, y) in enumerate(tqdm(self.val_loader)):
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])

            with torch.no_grad():
                output = self.pre(x)

            total_correct += (
                torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).item()
            )

            loss = self.criterion(output, y).item()
            total_loss += loss

        val_losses.append(total_loss / len(self.val_loader))

        print(f"Validation accuracy: {total_correct / len(self.val_loader.dataset)}")
        print(f"Validation loss: {total_loss / len(self.val_loader)}")

    def train(self, train_losses: list):
        print(f"Training:")

        self.pre.train()

        total_loss = 0

        for i, (x, y) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            x = x.to(self.config["device"])
            y = y.to(self.config["device"])

            output = self.pre(x)
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pre.parameters(), max_norm=self.clip_value)
            self.optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(self.train_loader))
        
        print(f"Training loss: {total_loss / len(self.train_loader)}")


    def run(self):
        trian_losses = []
        val_losses = []

        for epoch in range(self.config["epochs"]):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            self.train(trian_losses)
            self.evaluate(val_losses)

            gc.collect()
            torch.cuda.empty_cache()

            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.pre,
                    f"models/{self.version}/epoch_{epoch + 1}.pth",
                )

            if self.early_count >= self.config["early_stop"]:
                break

        return {
            "train_losses": trian_losses,
            "val_losses": val_losses,
        }
