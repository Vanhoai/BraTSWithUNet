import torch
import torch.nn as nn
import torch.nn.functional as F

from models import UNet


def main():
    model = UNet(n_channels=3, n_classes=2, bilinear=True)

    images = torch.randn(2, 3, 572, 572)
    masks = torch.randint(0, 2, (2, 1, 572, 572)).float()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad()

    outputs = model(images)
    if outputs.shape[-2:] != masks.shape[-2:]:
        outputs = F.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

    loss = criterion(outputs, masks)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        pred = model(images)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).float()

    print(f"Prediction shape: {pred_mask.shape}")


if __name__ == "__main__":
    main()
