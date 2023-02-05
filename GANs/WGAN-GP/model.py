from typing import Any

import torch
import torch.nn as nn


# Critic(Discriminator) Model
class Critic(nn.Module):
    def __init__(self, img_shape: tuple[int, ...], features_d: int, ngpu: int):
        super(Critic, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is batch_size x (nc) x 64 x 64
            nn.Conv2d(img_shape[0], features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 32 x 32

            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 16 x 16

            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 8 x 8

            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 4 x 4

            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, input: Any):                              # type: ignore
        return self.main(input)


def test_critic():
    img_shape = (3, 64, 64)
    features_d = 64
    ngpu = 1
    critic = Critic(img_shape, features_d, ngpu)
    x = torch.randn(1, 3, 64, 64)

    assert critic(x).shape == (1, 1, 1, 1)
    print(critic(x).shape)
    print(critic)


# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int, ...], features_g: int, ngpu: int):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                latent_dim, features_g * 16, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(),
            # state size. (features_g*16) x 4 x 4

            nn.ConvTranspose2d(
                features_g * 16, features_g * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),
            # state size. (features_g*8) x 8 x 8

            nn.ConvTranspose2d(
                features_g * 8, features_g * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            # state size. (features_g*4) x 16 x 16

            nn.ConvTranspose2d(
                features_g * 4, features_g * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            # state size. (features_g*2) x 32 x 32

            nn.ConvTranspose2d(
                features_g * 2, img_shape[0], 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input: Any):                              # type: ignore
        return self.main(input)


def test_generator():
    latent_dim = 100
    img_shape = (3, 64, 64)
    features_g = 64
    ngpu = 1
    generator = Generator(latent_dim, img_shape, features_g, ngpu)
    x = torch.randn(1, 100, 1, 1)

    assert generator(x).shape == (1, 3, 64, 64)
    print(generator(x).shape)
    print(generator)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    test_critic()
    test_generator()
