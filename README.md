#### U-NET Baseline

U-Net is a convolutional neural network architecture designed for biomedical image segmentation. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in their 2015 paper "U-Net: Convolutional Networks for Biomedical Image Segmentation". The architecture is named "U-Net" due to its U-shaped structure, which consists of a contracting path (encoder) and an expansive path (decoder).

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
- Example: For binary segmentation, the output layer will have 2 channels representing the background and the object of interest.

#### Key Factors

1. Skip Connections: U-Net uses skip connections to concatenate feature maps from the contracting path to the corresponding layers in the expansive path. This helps retain spatial information that may be lost during downsampling.

2. Valid Convolutions: U-Net employs valid convolutions (no padding) to ensure that the output size is smaller than the input size. This helps in reducing the number of parameters and computational complexity.
