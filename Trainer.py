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

        self.criterion = nn.MSELoss()

        self.clip_value = config["clip_value"]

        self.early_count = 0
        self.best_val_loss = float("inf")

    def evaluate(self, val_losses: list, val_accs: list):
        print(f"Evaluating:")

        self.pre.eval()

        total_loss = 0
        total_correct = 0

        for i, (data, max_len, label) in enumerate(tqdm(self.val_loader)):
            data = data.to(self.config["device"])
            label = label.to(self.config["device"])
            max_len = max_len.to(self.config["device"])

            with torch.no_grad():
                for j in range(0, max_len.max()):
                    # Check if any data point in the batch has reached its max_len
                    unfinished_data = max_len > j

                    if unfinished_data.any():
                        current_data = torch.flip(data[unfinished_data, : j + 1], [1])

                        output = self.pre(current_data, torch.ones_like(max_len[unfinished_data]) * (j + 1))

                        current_label = label[unfinished_data, j]

                        loss = self.criterion(output, current_label)

                        total_loss += loss.item()

                        total_correct += (
                            (output.argmax(dim=1) == current_label.argmax(dim=1))
                            .sum()
                            .item()
                        )

            if i == 100:
                break
            
        val_losses.append(total_loss / len(self.val_loader))
        val_accs.append(total_correct / len(self.val_loader.dataset))

        print(f"Validation accuracy: {total_correct / len(self.val_loader.dataset)}")
        print(f"Validation loss: {total_loss / len(self.val_loader)}")

    def train(self, train_losses: list):
        print(f"Training:")

        self.pre.train()

        total_loss = 0

        for i, (data, max_len, label) in enumerate(tqdm(self.train_loader)):
            data = data.to(self.config["device"])
            label = label.to(self.config["device"])
            max_len = max_len.to(self.config["device"])

            for j in range(0, max_len.max()):
                # Check if any data point in the batch has reached its max_len
                unfinished_data = max_len > j

                if unfinished_data.any():
                    self.optimizer.zero_grad()

                    current_data = torch.flip(data[unfinished_data, : j + 1], [1])

                    output = self.pre(current_data, torch.ones_like(max_len[unfinished_data]) * (j + 1))

                    loss = self.criterion(output, label[unfinished_data, j])

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.pre.parameters(), self.clip_value
                    )

                    self.optimizer.step()

                    total_loss += loss.item()
            
            if i == 900:
                break


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
