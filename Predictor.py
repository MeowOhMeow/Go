import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """
    Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError(
                "depthwise_kernel_size must be odd to achieve 'SAME' padding."
            )

        # Layer normalization for input
        self.layer_norm = nn.LayerNorm(input_dim)

        # Sequential layers: 1x1 Conv, GLU, Depthwise Conv, Normalization, Activation, 1x1 Conv, Dropout
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),  # SiLU activation function (Sigmoid Linear Unit)
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conformer convolution module.

        Args:
            input (torch.Tensor): Input tensor with shape `(B, T, D)`.
            B: Batch size, T: Sequence length, D: Input dimension

        Returns:
            torch.Tensor: Output tensor with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        # Transpose to shape `(B, D, T)` for 1D convolutions
        x = x.transpose(1, 2)
        x = self.sequential(x)  # Apply sequential layers
        return x.transpose(1, 2)  # Transpose back to shape `(B, T, D)`


class FeedForwardModule(nn.Module):
    """
    Feedforward module with Layer Normalization, Linear layers, SiLU activation, and Dropout.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float, optional): Dropout probability. (Default: 0.1)
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # SiLU activation function (Sigmoid Linear Unit)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the FeedForwardModule.

        Args:
            x (torch.Tensor): Input tensor with shape `(B, T, D)`.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        return self.module(x)


class ConformerBlock(nn.Module):
    """
    Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of the feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of the depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.1)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim,
        ffn_dim,
        num_attention_heads,
        depthwise_conv_kernel_size,
        dropout=0.1,
        use_group_norm=False,
        convolution_first=False,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.ffn2 = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.conv = ConvModule(
            input_dim,
            input_dim,
            depthwise_conv_kernel_size,
            dropout,
            use_group_norm=use_group_norm,
        )
        self.self_attn = nn.MultiheadAttention(
            input_dim, num_attention_heads, dropout=dropout
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_conv(self, x):
        """
        Apply the convolution module.

        Args:
            x (torch.Tensor): Input tensor with shape `(T, B, D)`.

        Returns:
            torch.Tensor: Output tensor after applying the convolution module.
        """
        residual = x
        # Transpose to shape `(B, T, D)` for 1D convolutions
        x = x.transpose(0, 1)
        x = self.conv(x)
        x = x.transpose(0, 1)  # Transpose back to shape `(T, B, D)`
        x = x + residual
        return x

    def forward(self, x):
        """
        Forward pass of the ConformerBlock.

        Args:
            x (torch.Tensor): Input tensor with shape `(T, B, D)`.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        residual = x
        x = self.ffn1(x)  # First feedforward module
        x = x * 0.5 + residual  # Residual connection and scaling

        if self.convolution_first:
            x = self._apply_conv(x)  # Apply convolution module if specified

        residual = x
        x = self.self_attn_layer_norm(x)  # Layer normalization
        x, _ = self.self_attn(x, x, x, need_weights=False)  # Multihead self-attention
        x = self.self_attn_dropout(x)
        x = x + residual  # Residual connection

        if not self.convolution_first:
            x = self._apply_conv(x)  # Apply convolution module if specified

        residual = x
        x = self.ffn2(x)  # Second feedforward module
        x = x * 0.5 + residual  # Residual connection and scaling
        x = self.layer_norm(x)  # Final layer normalization
        return x


class Conformer(nn.Module):
    """
    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.1)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        ffn_dim,
        num_layers,
        depthwise_conv_kernel_size,
        dropout=0.1,
        use_group_norm=False,
        convolution_first=False,
    ):
        super().__init__()

        # Instantiate Conformer blocks
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout,
                    use_group_norm,
                    convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Generator (Conformer model).

        Args:
            x (torch.Tensor): input with shape `(B, T, input_dim)`.

        Returns:
            torch.Tensor: output with shape `(B, T, input_dim)`.
        """

        x = x.transpose(0, 1)  # Transpose to shape `(T, B, input_dim)`

        # Pass input through Conformer blocks
        for layer in self.conformer_blocks:
            x = layer(x)

        x = x.transpose(0, 1)  # Transpose back to shape `(B, T, input_dim)`

        return x


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
        pe = torch.zeros(1, max_len, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

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
        # convoluiton layer
        self.conv = nn.Sequential(
            nn.Conv2d(
                2,
                4,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            nn.Conv2d(
                4,
                8,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )

        self.transform = nn.Sequential(
            nn.Linear(input_dim * 8, input_dim * 4),
            nn.SiLU(),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.SiLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

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
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Generator (Conformer model).

        Args:
            x (torch.Tensor): Input tensor with shape `(B, T, input_dim)`.

        Returns:
            torch.Tensor: Output tensor with shape `(B, output_dim)`.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Pass the input through the convolution layer
        x = x.view(batch_size * seq_len, 2, 19, 19)
        cnn_output = self.conv(x)
        flattened_output = cnn_output.view(batch_size, seq_len, -1)

        # Pass the input through the linear layer
        transformed_output = self.transform(flattened_output)

        # Pass the input through the positional encoding layer
        preprocessed_output = self.positional_encoding(transformed_output)

        # Pass the input through the Conformer layers
        conformer_output = self.conformer(preprocessed_output)

        # truncate the output to the last time step
        output = conformer_output[:, -1, :]

        # Pass the output through the linear layer
        output = self.output_layer(output)

        return output