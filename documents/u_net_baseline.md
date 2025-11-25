#### U-NET Baseline

U-Net is a convolutional neural network architecture designed for biomedical image segmentation. It was introduced by
Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their 2015 paper "U-Net: Convolutional Networks for Biomedical
Image Segmentation". The architecture is named "U-Net" due to its U-shaped structure, which consists of a contracting
path (encoder) and an expansive path (decoder).

#### Architecture

The U-Net architecture can be summarized as follows:

1. Contracting Path (Encoder):
    - Purpose: to capture context, extract features and reduce spatial dimensions.
    - Structure:
        - 2 convolutional layers 3 x 3 followed by a ReLU activation function.
        - 1 max-pooling layer 2 x 2 (stride 2) for downsampling.
        - The number of feature channels is doubled after each downsampling step: 64 -> 128 -> 256 -> 512 -> 1024.

    - From input, while throughing the max-pooling layers:
        - Size: 572×572 -> 284×284 -> 140×140 -> 68×68 -> 32×32
        - Channels: 1 -> 64 -> 128 -> 256 -> 512 -> 1024

2. Bottleneck:
    - Purpose: to connect the encoder and decoder.
    - Structure:
        - 2 convolutional layers 3 x 3 followed by a ReLU activation function.
        - No pooling layer.
    - Size: 32×32
    - Channels: 1024

3. Expansive Path (Decoder):

- Purpose: to enable precise localization and reconstruct the spatial dimensions.
- Structure:
    - Up-convolution (transposed convolution) 2 x 2 to upsample the feature map.
    - Skip connections from the corresponding layers in the contracting path to concatenate feature maps.
    - 2 convolutional layers 3 x 3 followed by a ReLU activation function
    - The number of feature channels is halved after each upsampling step: 1024 -> 512 -> 256 -> 128 -> 64.

4. Output Layer:

- Convolutional layer 1 x 1 to map the feature maps to the desired number of classes (e.g., 2 for binary segmentation).
- Example: For binary segmentation, the output layer will have 2 channels representing the background and the object of
  interest.

#### Key Factors

1. Skip Connections: U-Net uses skip connections to concatenate feature maps from the contracting path to the
   corresponding layers in the expansive path. This helps retain spatial information that may be lost during
   downsampling.
2. Valid Convolutions: U-Net employs valid convolutions (no padding) to ensure that the output size is smaller than the
   input size. This helps in reducing the number of parameters and computational complexity.

#### Understand Loss Calculation (with OxfordIIIT Pet Dataset)

##### Cross Entropy Loss

For segmentation tasks, we use Cross Entropy Loss, which measures how well the predicted probability distribution
matches the true distribution.

Formula:

$$
\text{Cross Entropy Loss} = -\sum_{i=1}^{N} y_i \log(p_i)
$$

Python Implementation:

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
```

##### Example with one pixel

```python
import numpy as np

# Suppose for ONE pixel at position (i, j):

# Model output (logits) for 3 classes:
logits = np.array([2.5, 1.0, 0.3])  # Raw scores (unnormalized)

# True label:
true_label = 0  # This pixel belongs to class 0 (pet)
label_one_hot = np.array([1, 0, 0])  # One-hot encoded

# Step 1: Convert logits to probabilities using Softmax
exp_logits = np.exp(logits)  # [12.18249396, 2.71828183, 1.34985881]

sum_exp = sum(exp_logits)  # 16.25063459673852

# probabilities = [e / sum_exp for e in exp_logits]
probabilities = exp_logits / sum_exp
# probabilities = [0.7496626601368837, 0.16727234941364075, 0.0830649904494756]
#                  ^^^^
#                  ~75% confidence it's class 0 (pet)

# Step 2: Calculate loss for this pixel
# 1: loss = -np.log(probabilities[true_label])
# 2: loss = -np.sum(label_one_hot * np.log(probabilities))
# 3: Manual calculation
# loss = 0
# for c in range(len(probabilities)):
#     loss += -label_one_hot[c] * np.log(probabilities[c])

loss = -np.sum(label_one_hot * np.log(probabilities))
print("Cross-Entropy Loss for the pixel:", loss)  # 0.28813196012021874
```

##### Visual Representation of Loss Calculation

```
Input Image (3x256x256)           Ground Truth Mask (256x256)
┌─────────────────┐               ┌─────────────────┐
│                 │               │  0 0 0 0 1 1 1  │
│   [Cat Image]   │               │  0 0 0 1 1 1 1  │
│                 │               │  0 0 0 1 1 1 2  │
│                 │               │  0 0 1 1 1 2 2  │
└─────────────────┘               └─────────────────┘
        ↓                         0: Pet
   UNet Model                     1: Background
        ↓                         2: Border
Logits (3x256x256)
┌─────────────────┐
│ Class 0: [...]  │  ← Raw scores for "Pet" at each pixel
│ Class 1: [...]  │  ← Raw scores for "Background"
│ Class 2: [...]  │  ← Raw scores for "Border"
└─────────────────┘
        ↓
    Softmax (implicit in CrossEntropyLoss)
        ↓
Probabilities (3x256x256)
┌─────────────────┐
│ Class 0: [0.8, 0.2, ...]  │
│ Class 1: [0.15, 0.7, ...] │
│ Class 2: [0.05, 0.1, ...] │
└─────────────────┘
        ↓
Compare with Ground Truth
        ↓
Calculate -log(p_correct) for each pixel
        ↓
Average over all pixels
        ↓
    Final Loss Value
```

##### Complete Loss Calculation Example

```python
import torch
import torch.nn as nn

# Simulate model output for a small image (2x2 pixels, 3 classes)
batch_size = 1
num_classes = 3
height, width = 2, 2

# Model outputs (logits) - shape: [batch, classes, height, width]
logits = torch.tensor([
    [  # Batch 0
        [  # Class 0 (Pet)
            [3.0, 1.0],
            [2.0, 0.5]
        ],
        [  # Class 1 (Background)
            [1.0, 3.5],
            [1.5, 3.0]
        ],
        [  # Class 2 (Border)
            [0.5, 0.8],
            [0.3, 2.5]
        ]
    ]
], dtype=torch.float32)

# Ground truth labels - shape: [batch, height, width]
labels = torch.tensor([
    [  # Batch 0
        [0, 1],  # Row 0: pixel(0,0)=class 0, pixel(0,1)=class 1
        [0, 2],  # Row 1: pixel(1,0)=class 0, pixel(1,1)=class 2
    ]
], dtype=torch.long)

print("Logits shape:", logits.shape)  # [1, 3, 2, 2]
print("Labels shape:", labels.shape)  # [1, 2, 2]

# Calculate Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# Let's manually calculate to understand:
total_loss = 0.0
for i in range(height):
    for j in range(width):
        pixel_logits = logits[0, :, i, j]  # [3] - logits for all classes
        true_label = labels[0, i, j].item()

        # Softmax
        exp_logits = torch.exp(pixel_logits)
        probs = exp_logits / exp_logits.sum()

        # Loss for this pixel
        pixel_loss = -torch.log(probs[true_label])
        print(f"Pixel ({i},{j}): True class={true_label}")
        print(f"  Logits: {pixel_logits.tolist()}")
        print(f"  Probabilities: {probs.tolist()}")
        print(f"  Prob of correct class: {probs[true_label]:.4f}")
        print(f"  Loss: {pixel_loss:.4f}\n")

        total_loss += pixel_loss.item()

print("Total Loss: ", loss.item())  # 0.48539113998413086
print("Manually Calculated Total Loss: ", total_loss / (height * width))  # 0.48539111018180847
```

##### Brief Summary: Loss Calculation Flow

```
1. Model Output (Logits)
   ↓
   [batch, 3, 256, 256] - raw scores

2. Apply Softmax (implicit in CrossEntropyLoss)
   ↓
   [batch, 3, 256, 256] - probabilities (sum to 1 per pixel)

3. For each pixel, select probability of true class
   ↓
   [batch, 256, 256] - probability of correct class at each pixel

4. Calculate -log(probability)
   ↓
   [batch, 256, 256] - loss per pixel

5. Average over all pixels and batches
   ↓
   scalar - final loss value

6. Backpropagate
   ↓
   Compute gradients for all weights

7. Update weights
   ↓
   Model learns to predict better masks
```
