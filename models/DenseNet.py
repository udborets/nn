from typing import Optional
import torch.nn as nn
import torch
import yaml


class _TransitionLayer(nn.Module):
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
        out = x
        for _, module in self.named_modules():
            out = module(out)
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
        out = x
        outputs = [out]
        for _, module in self.named_modules():
            out = module(out)
            outputs.append(out)
        return torch.concat(outputs, 1)


class DenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        in_features: int,
        conv_num_in_blocks: list[int],
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
        self.in_channels = in_channels
        self.in_features = in_features
        self.conv_num_in_blocks = conv_num_in_blocks
        self.stack_channels = stack_channels

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
        for _, module in self.named_modules():
            out = module(out)
        return out

    @staticmethod
    def from_config(config_path: str) -> "DenseNet":
        try:
            with open(config_path, "r", encoding="utf-8") as read_stream:
                config = yaml.safe_load(read_stream)
                read_stream.close()
                return DenseNet(
                    config["num_classes"],
                    config["in_channels"],
                    config["in_features"],
                    config["conv_num_in_blocks"],
                    config["stack_channels"],
                )
        except yaml.YAMLError as err:
            print(err)

    @staticmethod
    def create_config(
        config_save_path: str,
        num_classes: int,
        in_channels: int,
        in_features: int,
        conv_num_in_blocks: list[int],
        stack_channels: int,
    ) -> None:
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(
                f"""num_classes: {num_classes}\
                \nin_channels: {in_channels}\
                \nin_features: {in_features}\
                \nconv_num_in_blocks: {conv_num_in_blocks}\
                \nstack_channels: {stack_channels}\
                    """
            )
            f.close()

    def save_config(
        self,
        config_save_path: str,
    ) -> None:
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(
                f"""num_classes: {self.num_classes}\
                \nin_channels: {self.in_channels}\
                \nin_features: {self.in_features}\
                \nconv_num_in_blocks: {self.conv_num_in_blocks}\
                \nstack_channels: {self.stack_channels}\
                    """
            )
            f.close()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
