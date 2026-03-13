import torch
import torch.nn as nn

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=4, padding=2, kernel_size=11),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, padding=2, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_pass = self.features(x)
        pool_pass = self.avgpool(feature_pass)
        flatten_pass = torch.flatten(pool_pass, 1)
        classifier_pass = self.classifier(flatten_pass)

        return classifier_pass