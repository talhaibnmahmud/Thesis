from typing import Any

import torch
import torch.nn as nn

from model import Critic


def gradient_penalty(
    critic: nn.Module,
    labels: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10,
) -> torch.Tensor:
    """Calculates the gradient penalty loss for WGAN GP.

    Args:
        critic (nn.Module): The critic.
        labels (torch.Tensor): The labels.
        real (torch.Tensor): The real images.
        fake (torch.Tensor): The fake images.
        device (torch.device): The device type.
        lambda_gp (float, optional): The gradient penalty lambda hyperparameter.
            Defaults to 10.

    Returns:
        torch.Tensor: The gradient penalty loss.
    """
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)
    gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gp


def test_gradient_penalty():
    img_shape = (3, 128, 128)
    num_classes = 10
    features_d = 128
    ngpu = 1
    critic = Critic(img_shape, num_classes, features_d, ngpu)
    x = torch.randn(1, 3, 128, 128)
    labels = torch.randint(0, 10, (1,))
    assert critic(x, labels).shape == (1, 1, 1, 1)

    real = torch.randn(1, 3, 128, 128)
    fake = torch.randn(1, 3, 128, 128)
    device = torch.device("cpu")
    gp = gradient_penalty(critic, labels, real, fake, device)
    assert gp.shape == torch.Size([])
    assert gp.item() >= 0

    print(gp.shape)
    print(gp.item())
    print(gp)


# Save generator and discriminator to checkpoint
def save_checkpoint(
    checkpoint_file: str, generator: nn.Module, discriminator: nn.Module, optimizerG: torch.optim.Optimizer, optimizerD: torch.optim.Optimizer
) -> None:
    print("=> Saving checkpoint")
    checkpoint: dict[str, Any] = {
        "generator": generator.state_dict(),                    # type: ignore
        "discriminator": discriminator.state_dict(),            # type: ignore
        "optimizerG": optimizerG.state_dict(),                  # type: ignore
        "optimizerD": optimizerD.state_dict(),                  # type: ignore
    }
    torch.save(checkpoint, checkpoint_file)                     # type: ignore


# Load generator and discriminator from checkpoint
def load_checkpoint(
    checkpoint_file: str, generator: nn.Module, discriminator: nn.Module, optimizerG: torch.optim.Optimizer, optimizerD: torch.optim.Optimizer
) -> None:
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)                    # type: ignore
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])        # type: ignore
    optimizerD.load_state_dict(checkpoint["optimizerD"])        # type: ignore


if __name__ == "__main__":
    test_gradient_penalty()
