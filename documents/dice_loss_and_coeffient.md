#### Dice Coefficient and Dice Loss

- Cross Entropy Loss is widely used for segmentation tasks, but it may not always be the best choice, especially when
  dealing with imbalanced datasets. In such cases, the Dice Coefficient and Dice Loss can be more effective.
- The Dice Coefficient is a measure of overlap between two samples. It ranges from 0 (no overlap) to 1 (perfect
  overlap).

Formula:
$$
\text{Dice Coefficient} = \frac{2 |A \cap B|}{|A| + |B|}
$$

Where:

- \(A\) is the set of predicted pixels (predicted mask).
- \(B\) is the set of ground truth pixels (true mask).

#### Manual Calculation Steps:

1. Flatten the predicted and ground truth masks into 1D arrays.
2. Calculate the intersection (the number of pixels where both masks are 1).
   Formula for Intersection:
   $$
   |A \cap B| = \sum_{i=1}^{N} p_i \times g_i
   $$
3. Calculate the total number of pixels in each mask.
   Formula for Total Pixels:
   $$
   |A| = \sum_{i=1}^{N} p_i
   $$
   $$
   |B| = \sum_{i=1}^{N} g_i
   $$
4. Plug these values into the Dice Coefficient formula.
   Formula for Dice Coefficient:
   $$
   \text{Dice Coefficient} = \frac{2 \times |A \cap B|}{|A| + |B|}
   $$
5. Calculate Dice Loss as:
   $$
   \text{Dice Loss} = 1 - \text{Dice Coefficient}
   $$

#### Python Implementation:

```python
def dice_loss(pred, target, eps=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

    return 1 - dice


import torch

# Example usage
pred = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.float32)
target = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32)

loss = dice_loss(pred, target)
print(f"Dice Loss: {loss.item()}")
```

#### Multi-Class Dice Loss

For multi-class segmentation:

1. One-hot encode the ground truth labels.
2. Softmax outputs
3. Calculate Dice Loss for each class
4. Average the losses across classes

```python
def multiclass_dice_loss(pred, target, eps=1e-6):
    # pred softmax: [B, C, H, W]
    # target one-hot: [B, C, H, W]
    dice = 0
    num_classes = pred.shape[1]

    for c in range(num_classes):
        p = pred[:, c].contiguous().view(-1)
        t = target[:, c].contiguous().view(-1)

        intersection = (p * t).sum()
        dice_c = (2 * intersection + eps) / (p.sum() + t.sum() + eps)
        dice += 1 - dice_c

    return dice / num_classes


# Example usage for multi-class
import torch
import torch.nn.functional as F

# Example usage
pred = torch.randn(4, 3, 256, 256)  # shape: [B, C, H, W]
target = torch.randint(0, 3, (4, 256, 256))  # shape: [B, H, W]
target_one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

pred_softmax = F.softmax(pred, dim=1)
loss = multiclass_dice_loss(pred_softmax, target_one_hot)
print(f"Multiclass Dice Loss: {loss.item()}")
```
