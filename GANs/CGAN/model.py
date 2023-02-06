from typing import Any

import torch
import torch.nn as nn


# The Critic Model
class Critic(nn.Module):
    def __init__(self, img_shape: tuple[int, ...], num_classes: int, features_d: int, ngpu: int):
        super(Critic, self).__init__()

        self.ngpu = ngpu
        self.width, self.height = img_shape[1], img_shape[2]

        self.main = nn.Sequential(
            # input is (nc + 1) x 128 x 128
            nn.Conv2d(img_shape[0] + 1, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 64 x 64

            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 32 x 32

            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 16 x 16

            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 8 x 8

            nn.Conv2d(features_d * 8, features_d * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*16) x 4 x 4

            nn.Conv2d(features_d * 16, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            # State size. 1 x 1 x 1
        )

        self.label_embedding = nn.Embedding(
            num_classes, self.width * self.height
        )

    def forward(self, inputs: Any, labels: Any):                              # type: ignore
        labels = self.label_embedding(labels)
        labels = labels.view(labels.size(0), 1, self.width, self.height)

        return self.main(torch.cat([inputs, labels], 1))


def test_critic():
    img_shape = (3, 128, 128)
    num_classes = 10
    features_d = 128
    ngpu = 1

    critic = Critic(img_shape, num_classes, features_d, ngpu)
    print(critic)

    x = torch.randn(1, 3, 128, 128)
    labels = torch.randint(0, 10, (1,))

    assert critic(x, labels).shape == (1, 1, 1, 1)
    print(critic(x, labels).shape)


# Generator Model
class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_shape: tuple[int, ...],
        num_classes: int,
        embedding_dim: int,
        features_g: int,
        ngpu: int
    ):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.width, self.height = img_shape[1], img_shape[2]
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                latent_dim + embedding_dim, features_g * 32, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(features_g * 32),
            nn.ReLU(),
            # state size. (features_g*32) x 4 x 4

            nn.ConvTranspose2d(
                features_g * 32, features_g * 16, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(),
            # state size. (features_g*16) x 8 x 8

            nn.ConvTranspose2d(
                features_g * 16, features_g * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),
            # state size. (features_g*8) x 16 x 16

            nn.ConvTranspose2d(
                features_g * 8, features_g * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            # state size. (features_g*4) x 32 x 32

            nn.ConvTranspose2d(
                features_g * 4, features_g * 2, 4, 2, 1, bias=False
            ),
            nn.ReLU(),
            # state size. (features_g*2) x 64 x 64

            nn.ConvTranspose2d(
                features_g * 2, img_shape[0], 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, inputs: Any, labels: Any):                              # type: ignore
        # Latent vector z: (batch_size, latent_dim, 1, 1)
        labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)

        return self.main(torch.cat((inputs, labels), dim=1))


def test_generator():
    latent_dim = 100
    img_shape = (3, 128, 128)
    num_classes = 10
    embedding_dim = 128
    features_g = 128
    ngpu = 1

    generator = Generator(
        latent_dim, img_shape, num_classes, embedding_dim, features_g, ngpu
    )
    print(generator)

    x = torch.randn(1, latent_dim, 1, 1)
    y = torch.randint(0, num_classes, (1,))
    assert generator(x, y).shape == (1, 3, 128, 128)
    print(generator(x, y).shape)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    test_critic()
    test_generator()
