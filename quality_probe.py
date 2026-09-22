"""
quality_probe.py
----------------
Small CIFAR-10 classifier used as the fixed quality probe in the cycle-eval
and conditional-accuracy parts of the task. Train it once during setup, then
use ``load_probe`` to call it as a frozen judge on your generated images. Do
NOT retrain it as part of your task -- the probe must remain a fixed external
judge across all your sampler / step-count combinations.

Setup (run once):

    python quality_probe.py [--epochs N] [--device cpu|cuda|mps] [--out_path PATH]

Then in your code:

    from quality_probe import load_probe
    probe = load_probe(device="cuda")
    logits = probe(images_in_minus1_to_1)  # (B, 3, 32, 32) -> (B, 10)

The default 8-epoch run reaches ~70-75% val accuracy. On CPU it takes a few
minutes; on a single GPU it is much faster.
"""

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from prepare_dataset import PairedCIFAR10, prepare

PROBE_WEIGHTS = os.path.join(os.path.dirname(__file__), "probe_weights.pt")


class Probe(nn.Module):
    """Small CNN that classifies CIFAR-10 images in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                                 # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        :param x: (B, 3, 32, 32) tensor in [-1, 1].
        :return: (B, 10) class logits.
        """
        return self.net(x)


def load_probe(device: str = "cpu", path: str = PROBE_WEIGHTS) -> Probe:
    """Load the trained probe in eval mode with frozen parameters.

    :param device: Target torch device string.
    :param path: Path to the saved ``probe_weights.pt``.
    :return: A frozen ``Probe`` ready for inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run `python quality_probe.py` to train it.")
    model = Probe().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def train(*, epochs: int = 8, device: str = "cpu", out_path: str = PROBE_WEIGHTS) -> None:
    """Train the probe on the paired CIFAR-10 splits and save its weights.

    :param epochs: Number of training epochs.
    :param device: Torch device string.
    :param out_path: Destination path for the weights.
    """
    prepare()
    train_dataset = PairedCIFAR10("train")
    val_dataset = PairedCIFAR10("val")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)

    model = Probe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, _, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, _, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        print(f"epoch {epoch + 1}: val acc = {correct / total:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"saved {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """argparse parser for the quality-probe training CLI."""
    parser = argparse.ArgumentParser(description="train the CIFAR-10 quality probe")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--out_path", default=PROBE_WEIGHTS, help="output weights path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(epochs=args.epochs, device=args.device, out_path=args.out_path)


if __name__ == "__main__":
    main()
