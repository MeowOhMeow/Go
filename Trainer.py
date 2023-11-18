import datetime
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import Predictor


class Trainer:
    def __init__(
        self,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_model: bool = False,
        from_epoch: int = 0,
        path: str = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.from_epoch = from_epoch
        self.version = f'v{config["selected"]}'
        
        if not os.path.exists(f"models/{self.version}"):
            os.makedirs(f"models/{self.version}")

        if load_model:
            self.pre = torch.load(path)
        else:
            self.pre = Predictor(
                input_dim=config["input_dim"],
                output_dim=config["output_dim"],
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

        self.criterion = nn.MSELoss(reduction="mean")

        self.clip_value = config["clip_value"]

        self.early_count = 0
        self.best_val_loss = float("inf")

    def evaluate(self, val_losses: list, val_accs: list):
        print(f"Evaluating:")

        self.pre.eval()

        total_loss = 0
        total_correct = 0
        total_predictions = 0

        for i, (data, max_len, label) in enumerate(tqdm(self.val_loader)):
            data = data.to(self.config["device"])
            label = label.to(self.config["device"])
            max_len = max_len.to(self.config["device"])

            with torch.no_grad():
                output = self.pre(data, max_len)

            loss = self.criterion(output, label)
            total_loss += loss.item()

            total_correct += ( (output.argmax(2) == label.argmax(2)) * (label.argmax(2) != 0) ).sum().item()
            total_predictions += max_len.sum().item()
            
        val_losses.append(total_loss / len(self.val_loader))
        val_accs.append(total_correct / total_predictions)

        print(f"Validation accuracy: {total_correct / total_predictions}")
        print(f"Validation loss: {total_loss / len(self.val_loader)}")

    def train(self, train_losses: list):
        print(f"Training:")

        self.pre.train()

        total_loss = 0

        for i, (data, max_len, label) in enumerate(tqdm(self.train_loader)):
            data = data.to(self.config["device"])
            label = label.to(self.config["device"])
            max_len = max_len.to(self.config["device"])

            self.optimizer.zero_grad()

            output = self.pre(data, max_len)
            
            loss = self.criterion(output, label)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pre.parameters(), self.clip_value)
            self.optimizer.step()


        train_losses.append(total_loss / len(self.train_loader))
        
        print(f"Training loss: {total_loss / len(self.train_loader)}")


    def run(self):
        trian_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(self.config["epochs"]):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            self.train(trian_losses)
            self.evaluate(val_losses, val_accs)

            gc.collect()
            torch.cuda.empty_cache()


            torch.save(
                self.pre,
                f"models/{self.version}/epoch_{epoch + 1}.pth",
            )

            if self.early_count >= self.config["early_stop"]:
                break

        return {
            "train_losses": trian_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }
