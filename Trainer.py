import datetime
import gc

import torch
import torch.nn as nn
from IPython.display import clear_output
from torch.utils.data import DataLoader
from tqdm import tqdm

from Predictor import Predictor


class Trainer:
    def __init__(
        self,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_model: bool = False,
        from_epoch: int = 0,
        G_path: str = None,
        D_path: str = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.from_epoch = from_epoch

        if load_model:
            self.pre = torch.load(G_path)
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

    def normal_evaluate_G(self, G_val_losses: list):
        self.pre.eval()
        for i, (x, y) in enumerate(tqdm(self.val_loader)):
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])
            output = self.pre(x)
            loss = self.criterion(output, y)
            G_val_losses.append(loss.item())

    def evaluate_G(self, G_val_losses: list, G_accs: list, epoch):
        print(f"Evaluating generator:")

        # Set the generator and discriminator in evaluation mode
        self.pre.eval()
        self.dis.eval()
        total_loss = 0
        total_correct = 0

        # Iterate through the validation loader
        for i, (x, y) in enumerate(tqdm(self.val_loader)):
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])

            with torch.no_grad():
                # Generate fake data and conditioning information from the generator
                output = self.pre(x)

                # Pass fake data and conditioning information through the discriminator
                fake_pred = self.dis(x, output)

            # Determine the predicted classes for fake and real samples
            fake_indices = torch.argmax(fake_pred, dim=1)
            real_indices = torch.argmax(y, dim=1)

            # Count correct predictions
            correct = torch.sum(fake_indices == real_indices)
            total_correct += correct.item()

            # Compute generator loss for both the image output and the discriminator predictions
            loss = self.criterion(output, y) + -torch.mean(fake_pred)
            total_loss += loss.item()

        # Calculate and store the average generator validation loss
        average_loss = total_loss / len(self.val_loader)
        G_val_losses.append(average_loss)

        # Calculate and store the validation accuracy
        accuracy = total_correct / len(self.val_loader.dataset)
        G_accs.append(accuracy)
        print(f"G Validation accuracy: {accuracy}")

        if average_loss < self.best_val_loss:
            self.best_val_loss = average_loss
            self.early_count = 0
        else:
            self.early_count += 1

        if (epoch + self.from_epoch + 1) % 10 == 0:
            torch.save(
                self.pre,
                self.config["gen_path"]
                + "/"
                + str(self.config["selected"])
                + "_"
                + f'epoch{epoch + self.from_epoch + 1}'
                + ".pth",
            )
            torch.save(
                self.dis,
                self.config["dis_path"]
                + "/"
                + str(self.config["selected"])
                + "_"
                + f'epoch{epoch + self.from_epoch + 1}'
                + ".pth",
            )

    def evaluate_D(self, D_val_losses: list, D_accs: list):
        print(f"Evaluating discriminator:")

        # Set the generator and discriminator in evaluation mode
        self.pre.eval()
        self.dis.eval()

        total_loss = 0
        total_correct = 0
        total_fake_loss = 0
        total_real_loss = 0

        # Iterate through the validation loader
        for i, (x, y) in enumerate(tqdm(self.val_loader)):
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])
            
            with torch.no_grad():
                # Generate fake data and conditioning information from the generator
                output = self.pre(x)
                # Pass fake data and conditioning information through the discriminator
                fake_pred = self.dis(x, output)
                real_pred = self.dis(x, y)

            total_correct += torch.sum(fake_pred < 0.5) + torch.sum(real_pred > 0.5)

            total_loss += -torch.mean(real_pred) + torch.mean(fake_pred)
            total_fake_loss += torch.mean(fake_pred)
            total_real_loss += -torch.mean(real_pred)

        # Calculate and store the average discriminator validation loss
        average_loss = total_loss / len(self.val_loader)
        D_val_losses.append(average_loss.item())
        print(f"Discriminator loss: {average_loss}")
        print(f"Fake loss: {total_fake_loss / len(self.val_loader)}")
        print(f"Real loss: {total_real_loss / len(self.val_loader)}")

        # Calculate and store the validation accuracy
        accuracy = total_correct / len(self.val_loader.dataset) / 2
        D_accs.append(accuracy.item())
        print(f"D Validation accuracy: {accuracy}")

        return accuracy

    def normal_train_G(self, G_losses: list):
        self.pre.train()
        total_loss = 0
        for i, (x, y) in enumerate(tqdm(self.train_loader)):
            self.G_normal_optimizer.zero_grad()
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])
            output = self.pre(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.G_normal_optimizer.step()
            total_loss += loss.item()

        G_losses.append(total_loss / len(self.train_loader))
        print(f"Normal Generator loss: {total_loss / len(self.train_loader)}")

    def train_G(self, x, y):
        x = x.to(self.config["device"])

        # Generate fake data and conditioning information from the generator
        fake_output = self.pre(x)

        # Pass fake data and conditioning information through the discriminator
        fake_pred = self.dis(x, fake_output)

        # Total loss for the generator: discriminator loss + normal loss
        loss = -torch.mean(fake_pred)

        # Backpropagation and optimization step
        self.G_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pre.parameters(), max_norm=self.clip_value)
        self.G_optimizer.step()

        return loss.item()

    def cal_gradient_penalty(self, real, fake, condition, lambda_gp=10):
        batch_size = real.size(0)
        alpha = torch.rand((batch_size, 1), dtype=real.dtype, device=real.device)

        # Interpolate between real and fake samples based on alpha
        interpolates = alpha * real + (1 - alpha) * fake
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        # Pass the interpolated samples through the discriminator
        disc_interpolates = self.dis(condition, interpolates)

        # Compute gradients of the interpolated samples with respect to inputs
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten and calculate the norm of the gradients for each sample in the batch
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Calculate gradient penalty based on the Lipschitz constraint formula
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        # Scale the gradient penalty by lambda_gp and add it to the loss
        return lambda_gp * gradient_penalty

    def train_D(self, x, y):
        # Move real data and labels to the specified device
        x = x.to(self.config["device"])
        y = y.to(self.config["device"])

        # Generate fake data and conditioning information from the generator
        fake_output = self.pre(x)

        # Pass fake data and conditioning information through the discriminator
        fake_pred = self.dis(x, fake_output)
        real_pred = self.dis(x, y)

        # Calculate the gradient penalty
        gradient_penalty = self.cal_gradient_penalty(
            y, fake_output, x
        )

        # Calculate the total loss: -real + fake + gradient penalty
        loss = -torch.mean(real_pred) + torch.mean(fake_pred) + gradient_penalty

        # Backpropagation and optimization step
        self.D_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=self.clip_value)
        self.D_optimizer.step()
        

        return loss.item()

    def train(self):
        G_losses = []
        G_val_losses = []
        G_accs = []
        D_losses = []
        D_val_losses = []
        D_accs = []

        D_acc = self.evaluate_D(D_val_losses, D_accs)
            

        for epoch in range(self.config["epochs"]):
            self.pre.train()
            self.dis.train()
            D_total_loss = 0
            G_total_loss = 0
            print(f'Epoch {epoch+1}/{self.config["epochs"]}')
            for i, (x, y) in enumerate(tqdm(self.train_loader)):
                if D_acc < 0.8:
                    D_loss = self.train_D(x, y)
                    D_total_loss += D_loss

  
                G_loss = self.train_G(x, y)

                G_total_loss += G_loss

            self.normal_train_G(G_losses)

            print(f"Discriminator loss: {D_total_loss / len(self.train_loader)}")
            D_losses.append(D_total_loss / len(self.train_loader))
            print(f"GAN Generator loss: {G_total_loss / len(self.train_loader)}")
            G_losses[-1] = G_total_loss / len(self.train_loader) + G_losses[-1]

            D_acc = self.evaluate_D(D_val_losses, D_accs)
            self.evaluate_G(G_val_losses, G_accs, epoch)

            gc.collect()
            torch.cuda.empty_cache()

            if (epoch + 1) % 5 == 0:
                clear_output(wait=True)

            if self.early_count >= self.config["early_stop"]:
                break

        return {
            "G_losses": G_losses,
            "G_val_losses": G_val_losses,
            "G_accs": G_accs,
            "D_losses": D_losses,
            "D_val_losses": D_val_losses,
            "D_accs": D_accs,
        }
