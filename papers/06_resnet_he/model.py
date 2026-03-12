import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int,  out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, padding=1, kernel_size = (3,3), bias = False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, kernel_size = (3,3), out_channels =  out_channels, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, stride = stride, bias = False, kernel_size = (1,1)),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Sequential(

            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1(x)
        bn1_out = self.batchnorm1(conv1_out)
        act1_out = self.relu_act(bn1_out)
        conv2_out = self.conv2(act1_out)
        bn2_out = self.batchnorm2(conv2_out)
        res_add = bn2_out + self.shortcut(x)
        final_res = self.relu_act(res_add)

        return final_res



