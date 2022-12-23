import torch.nn as nn


# defines the convolutional neural network
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            self._block(3, 16, kernel_size=5, stride=1, padding=2),
            self._block(16, 32, kernel_size=5, stride=1, padding=2),
        )

        # Linear Layer: Input Dimension --> 32x7x7
        self.flat = nn.Sequential(
            nn.Linear(32*7*7, 100),
            # Output Dimension --> 1x100
            nn.LeakyReLU(),

            nn.Linear(100, 27),
            # Output Dimension --> 1x27
            # nn.Sigmoid()
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.cnn(x)

        # flatten the dataset
        flattened = self.flat(out.reshape(-1, 32*7*7))

        return flattened
