import torch
from torch import nn
from blocks import ResidualBlock, ResidualBottleneckBlock
from utils import get_norm

from typing import Literal


resnet_config = {
    18: {
        "block": ResidualBlock,
        "num_blocks": [2, 2, 2, 2]
    },
    34: {
        "block": ResidualBlock,
        "num_blocks": [3, 4, 6, 3]        
    },
    50: {
        "block": ResidualBottleneckBlock,
        "num_blocks": [3, 4, 6, 3]
    },
    101: {
        "block": ResidualBottleneckBlock,
        "num_blocks": [3, 4, 23, 3]
    },
    152: {
        "block": ResidualBottleneckBlock,
        "num_blocks": [3, 8, 36, 3]
    }
}

    
class CIFARResNet(nn.Module):
    def __init__(self, 
                 configuration: int = 34, 
                 in_channels: int = 3, 
                 num_classes: int = 10, 
                 norm: Literal["batch", "group"] = "batch", 
                 num_groups: int = 32, 
                 base_channels: int = 64) -> None:
        super(CIFARResNet, self).__init__()
        
        config = resnet_config[configuration]
        block = config['block']
        num_blocks = config['num_blocks']
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.norm = norm
        self.num_groups = num_groups
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=base_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            get_norm(norm, base_channels, num_groups),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(block, base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * block.expansion, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 2 * block.expansion, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 4 * block.expansion, base_channels * 8, num_blocks[3], stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        features = x.view(x.size(0), -1)
        
        out = self.fc(features)
        return out
        
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):
        if stride != 1 or in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, 
                          out_channels=out_planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                get_norm(self.norm, out_planes * block.expansion, self.num_groups)
            )
        else:
            downsample = None
        layers = [block(in_planes, out_planes, stride=stride, downsample=downsample, norm=self.norm, num_groups=self.num_groups)]
        for _ in range(1, num_blocks):
            layers.append(block(out_planes * block.expansion, out_planes, stride=1, downsample=None, norm=self.norm, num_groups=self.num_groups))
            
        return nn.Sequential(*layers)