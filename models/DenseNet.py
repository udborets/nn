import torch.nn as nn
import torch


class _TransitionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        in_channels,
        out_channels: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwards)
        self.act = nn.ReLU(inplace=True)
        self.in_features = in_features
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        out = self.bn(x)
        out = self.act(out)
        out = self.conv(out)
        out = self.avg_pool(out)
        return out


class _DenseBlock(nn.Module):
    pass


class DenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        conv_num_in_blocks,
        stack_channels: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.classifier = nn.Linear()
        # for i, num_conv_layers in enumerate(conv_num_in_blocks):
        # self.add_module("conv_1")

    @staticmethod
    def from_config(config_path: str) -> DenseNet:
        pass

    @staticmethod
    def craete_config(save_to: str) -> None:
        pass
