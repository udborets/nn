from typing import Optional
import torch.nn as nn
import torch


class _TransitionLayer(nn.ModuleList):
    def __init__(
        self,
        in_channels,
        out_channels: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU(inplace=True)
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


class _DenseBlock(nn.ModuleList):
    def __init__(
        self,
        in_channels: int,
        conv_num: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        for i in range(conv_num):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            bn = nn.BatchNorm2d(in_channels)
            act = nn.ReLU(inplace=True)
            self.add_module(f"layer_{i + 1}", nn.Sequential(bn, act, conv))

    def forward(self, x: torch.Tensor):
        outputs = [x]
        for layer in self:
            out = layer(x)
            outputs.append(out)
        return torch.concat(outputs, 1)


# DenseNet(5, 3, 256, [2, 2, 3], 32)
class DenseNet(nn.ModuleList):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        in_features: int,
        conv_num_in_blocks,
        stack_channels: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.upsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=stack_channels,
            kernel_size=1,
            stride=1,
        )
        self.num_classes = num_classes

        num_channels = stack_channels
        for i in range(len(conv_num_in_blocks) - 1):
            dense = _DenseBlock(num_channels, conv_num_in_blocks[i])
            self.add_module(f"dense_{i + 1}", dense)
            num_channels += num_channels * conv_num_in_blocks[i]
            self.add_module(
                f"transition_{i + 1}",
                _TransitionLayer(
                    in_channels=num_channels,
                ),
            )
        dense = _DenseBlock(num_channels, conv_num_in_blocks[-1])
        num_channels += num_channels * conv_num_in_blocks[-1]
        self.add_module(f"dense_{len(conv_num_in_blocks)}", dense)
        self.add_module(
            "downsample",
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
            ),
        )
        self.add_module(
            "global_avg_pool",
            nn.AvgPool2d(kernel_size=in_features // 2 ** (len(conv_num_in_blocks) - 1)),
        )
        self.add_module("flatten", nn.Flatten())

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self:
            out = layer(out)
            print(layer, out.shape)
        return out

    @staticmethod
    def from_config(config_path: str) -> "DenseNet":
        pass

    @staticmethod
    def craete_config(save_to: str) -> None:
        pass
