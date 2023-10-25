import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ConvModule2D(torch.nn.Module):
    r"""Conformer convolution module.

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
                "depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class FeedForwardModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.module(x)


class ConformerBlock(nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
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
        self.conv = ConvModule2D(
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
        self.layer_norm = nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def apply_conv(self, x):
        residual = x
        x = x.transpose(0, 1)
        x = self.conv(x)
        x = x.transpose(0, 1)
        x = x + residual
        return x

    def forward(self, x):
        residual = x
        x = self.ffn1(x)
        x = 0.5 * x + residual

        if self.convolution_first:
            x = self.apply_conv(x)

        residual = x
        x = self.layer_norm(x)
        x, _ = self.self_attn(x, x, x)
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self.apply_conv(x)

        residual = x
        x = self.ffn2(x)
        x = 0.5 * x + residual
        x = self.layer_norm(x)
        return x


class Generator(nn.Module):
    r"""
    Model: Conformer

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)"""

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
        output_dim = input_dim
        self.generator_output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        r"""
        Args:
            x (torch.Tensor): input with shape `(B, T, input_dim)`.

        Returns:
            torch.Tensor: output with shape `(B, T, input_dim)`.
        """

        x = x.transpose(0, 1)
        for layer in self.conformer_blocks:
            x = layer(x)
        x = x.transpose(0, 1)

        return self.generator_output(x)


# Define the batch size, number of channels, height, and width
batch_size = 32
num_channels = 3
height = 64
width = 64

# Generate random test data with the specified shape
test_data = torch.randn(batch_size, num_channels, height, width)
print(f'original shape: {test_data.shape}')
test_data = test_data.view(batch_size, num_channels, height * width)
print(f'flattened shape: {test_data.shape}')

conformer = Generator(
    input_dim=height * width,
    num_heads=4,
    ffn_dim=32,
    num_layers=2,
    depthwise_conv_kernel_size=3,
    dropout=0.1,
    use_group_norm=False,
    convolution_first=False,
)

# print(conformer)

# Pass the test data to the model
output = conformer(test_data)
print(f'output shape: {output.shape}')
output = output.view(batch_size, num_channels, height, width)
print(f'reshaped output shape: {output.shape}')
