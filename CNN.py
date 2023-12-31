import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN model for board state recognition.
    This is a simple implementation of CNN model.
    Args:
        output_dim (int): Dimension of output vector
        dropout (float, optional): Dropout ratio. (default=0.0)
    """

    def __init__(
        self,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # convoluiton layer
        self.conv = nn.Sequential(
            # # (B, 3, 19, 19) -> (B, 9, 19, 19)
            nn.Conv2d(
                3,
                32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            # (B, 9, 19, 19) -> (B, 32, 19, 19)
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        self.transform = nn.Sequential(
            nn.Linear(32 * 19 * 19 + 1, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, color: torch.Tensor):
        """
        Forward pass of the Generator (Conformer model).

        Args:
            x (torch.Tensor): Input tensor with shape `(B, T, 19, 19, 3)`.

        Returns:
            torch.Tensor: Output tensor with shape `(B, T, output_dim)`.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # convert from (B, T, 19, 19, 3) to (B, T, 3, 19, 19)
        x = x.permute(0, 1, 4, 2, 3)
        # convert from (B, T, 3, 19, 19) to (B * T, 3, 19, 19)
        x = x.reshape(batch_size * seq_len, 3, 19, 19)
        # Pass the input through the convolution layer
        cnn_output = self.conv(x)
        # convert from (B * T, channels, transformed_height, transformed_width)
        #           to (B, T, transformed_height, transformed_width, channels)
        cnn_output = cnn_output.permute(0, 2, 3, 1)
        # flatten the output into 1D
        flattened_output = cnn_output.reshape(batch_size, seq_len, -1)
        # concatenate color information
        flattened_output = torch.cat([flattened_output, color], dim=-1)
        # Pass the input through the transformation layer
        return self.transform(flattened_output)
