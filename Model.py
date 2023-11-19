import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn

from Conformer import Conformer
from CNN import CNN


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.

    Args:
        input_dim (int): input dimension.
        max_len (int): maximum length of input sequence.
        dropout (float, optional): dropout probability. (Default: 0.1)
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim)
        )
        if input_dim % 2 != 0:
            pe = torch.zeros(1, max_len, input_dim + 1)
        else:
            pe = torch.zeros(1, max_len, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        if input_dim % 2 != 0:
            pe = pe[:, :, :-1]

        # Register buffer rather than a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor with shape `(B, T, D)`.
                B: batch size, T: sequence length, D: input dimension

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Predictor(nn.Module):
    """
    Classifier model using Conformer architecture.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads in each Conformer layer.
        ffn_dim (int): Hidden layer dimension of feedforward networks in Conformer layers.
        num_layers (int): Number of Conformer layers.
        depthwise_conv_kernel_size (int): Kernel size of depthwise convolution in Conformer layers.
        dropout (float, optional): Dropout probability. (Default: 0.1)
        use_group_norm (bool, optional): Use GroupNorm instead of BatchNorm1d in Conformer layers. (Default: False)
        convolution_first (bool, optional): Apply convolution module ahead of attention module. (Default: False)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        max_len,
        num_heads,
        ffn_dim,
        num_layers,
        depthwise_conv_kernel_size,
        dropout=0.1,
        use_group_norm=False,
        convolution_first=False,
    ):
        super(Predictor, self).__init__()

        self.CNN = CNN(input_dim, dropout)
        
        # Instantiate the PositionalEncoding module
        self.positional_encoding = PositionalEncoding(input_dim, max_len, dropout)

        # Instantiate the Conformer module
        self.conformer = Conformer(
            input_dim,
            num_heads,
            ffn_dim,
            num_layers,
            depthwise_conv_kernel_size,
            dropout,
            use_group_norm,
            convolution_first,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, color: torch.Tensor, lengths: torch.Tensor):
        """
        Forward pass of the Generator (Conformer model).

        Args:
            x (torch.Tensor): Input tensor with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in ``x``.

        Returns:
            torch.Tensor: Output tensor with shape `(B, output_dim)`.
        """
        # Pass the input through the CNN
        cnn_output = self.CNN(x, color)

        # Pass the input through the positional encoding layer
        preprocessed_output = self.positional_encoding(cnn_output)

        # Pass the input through the Conformer layers
        conformer_output, _ = self.conformer(preprocessed_output, lengths)

        # truncate the output to the first token
        output = conformer_output[:, 0, :]

        # Pass the output through the linear layer
        output = self.output_layer(output)

        return output