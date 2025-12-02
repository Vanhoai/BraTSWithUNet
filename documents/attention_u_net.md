#### What is attention ?

Attention, in the context of image segmentation, is a way to highlight only the relevant activations during training. This reduce the computational resources wasted on irrelevant activations, providing the network with better generalization power. So the network can pay "attetion" to the certain parts of the image.

##### Hard Attention:

Reference: (What are the main difference between hard attention and soft attention ?)[https://eitca.org/artificial-intelligence/eitc-ai-adl-advanced-deep-learning/attention-and-memory/attention-and-memory-in-deep-learning/examination-review-attention-and-memory-in-deep-learning/what-are-the-main-differences-between-hard-attention-and-soft-attention-and-how-does-each-approach-influence-the-training-and-performance-of-neural-networks/]

Hard attention is a mechanism that selects a specific part of the input to focus on, effectively making a discrete choice about where to allocate attention. This selection is typically non-differentiable, requiring methods such as reinforcement learning or the use of the REINFORCE algorithm to train the model. The process involves sampling from a distribution to decide which part of the input to attend to, making it inherently stochastic.

Characteristics of Hard Attention:

1. Discrete Selection: Hard attention makes a binary decision about which parts of the input to focus on. For example, in image processing, it might select specific pixels or regions.
2. Non-Differentiability: The discrete nature of hard attention means that the gradient cannot be directly calculated through the attention mechanism. This necessitates alternative training methods.
3. Efficiency: By focusing on specific parts of the input, hard attention can be more computationally efficient. It processes only the relevant parts, potentially reducing the computational load.
4. Sparsity: Hard attention leads to sparse attention maps, where only a few elements are attended to at any given time.

Training Hard Attention:

Training hard attention is challenging due to its non-differentiable nature. Common approaches include:

– Reinforcement Learning: Techniques such as policy gradients can be used to train the attention mechanism. The model learns to maximize a reward signal, which is often related to the performance on the task.
– REINFORCE Algorithm: This is a Monte Carlo method for optimizing the expected reward. It involves sampling actions (attention choices) and updating the policy based on the observed rewards.

##### Soft Attention:

Soft attention, in contrast, is a differentiable mechanism that assigns a continuous weight to each part of the input. This results in a weighted sum of the input features, where the weights represent the importance of each part. Soft attention is fully differentiable, allowing for end-to-end training using backpropagation.

Characteristics of Soft Attention:

1. Continuous Weights: Soft attention assigns continuous weights to all parts of the input, creating a weighted combination of features.
2. Differentiability: The continuous nature of soft attention allows gradients to be calculated directly through the attention mechanism, facilitating end-to-end training.
3. Comprehensive: Unlike hard attention, soft attention considers all parts of the input, albeit with different degrees of importance.
4. Interpretability: The attention weights can provide insights into which parts of the input are most influential in the model's decision-making process.

Training Soft Attention:

Training soft attention is more straightforward compared to hard attention, as it can be integrated seamlessly into the backpropagation process. The attention weights are learned alongside the other parameters of the model.

#### Influence on Training and Performance

The choice between hard and soft attention has significant implications for the training and performance of neural networks.

Training Complexity:
– Hard Attention: Requires more complex training methods such as reinforcement learning or the REINFORCE algorithm. These methods can be more difficult to implement and may require more computational resources.
– Soft Attention: Can be trained using standard backpropagation, making it easier to implement and more computationally efficient in terms of training.

Computational Efficiency:
– Hard Attention: Can be more efficient during inference, as it processes only the selected parts of the input. This can lead to faster inference times and reduced memory usage.
– Soft Attention: Processes the entire input, which can be computationally intensive, especially for large inputs such as high-resolution images or long sequences.

Performance:
– Hard Attention: Can be more effective in scenarios where focusing on specific parts of the input is important. However, the stochastic nature of hard attention can lead to instability during training.
– Soft Attention: Generally provides more stable training and can achieve high performance across a variety of tasks. The continuous attention weights allow for more nuanced focus on different parts of the input.

Interpretability:
– Hard Attention: The discrete nature of hard attention can make it easier to interpret which parts of the input the model is focusing on, as it makes clear, binary decisions.
– Soft Attention: The continuous attention weights provide a more detailed view of the model's focus, but this can sometimes be less intuitive to interpret compared to the binary decisions of hard attention.

#### Why is attention needed in U-Net ?

During upsampling in the expanding path, spatial information recreated is imprecise. To counteract this problem, the U-Net uses skip connections that combine spatial information from the downsampling path with the upsampling path. However, this brings across many redundant low-level feature extractions, as feature representation is poor in the initial layers.

Soft attention implemented at the skip connections will actively suppress activations in irrelevant regions, reducing the number of redundant features brought across.

#### Attention U-Net Architecture

The attention gates introduced by Oktay et al. uses additive soft attention.

1. The attention gate takes in two inputs, vector "x" and "g".
    - "x" is the feature map from the contracting path (skip connection).
    - "g" is the feature map from the expansive path (coarser scale).
2. Both inputs are passed through a 1x1 convolution to reduce the number of channels to an intermediate value "F_int".
3. The outputs are summed element-wise.
4. The summed output is passed through a ReLU activation function.
5. The activated output is passed through another 1x1 convolution to reduce the channels to 1.
6. A sigmoid activation function is applied to obtain the attention coefficients "α".
7. The attention coefficients "α" are multiplied element-wise with the input feature map "x" to produce the output feature map "x_out".

#### Example

```
x: (batch_size, 64, 128, 128)  # Feature map from contracting path
g: (batch_size, 128, 64, 64)   # Feature map from expansive path

F_int = 32  # Intermediate number of channels

# 1x1 convolutions to reduce channels
W_x = Conv2D(F_int, kernel_size=1)(x)  # Shape: (batch_size, 32, 128, 128)
W_g = Conv2D(F_int, kernel_size=1)(g)  # Shape: (batch_size, 32, 64, 64)

# Upsample W_g to match W_x spatial dimensions
W_g_up = UpSampling2D(size=(2, 2))(W_g)  # Shape: (batch_size, 32, 128, 128)

# Element-wise sum
f = W_x + W_g_up  # Shape: (batch_size, 32, 128, 128)

# ReLU activation
f_relu = ReLU()(f)  # Shape: (batch_size, 32, 128, 128)

# 1x1 convolution to reduce channels to 1
psi = Conv2D(1, kernel_size=1)(f_relu)  # Shape: (batch_size, 1, 128, 128)

# Sigmoid activation to get attention coefficients
α = Sigmoid()(psi)  # Shape: (batch_size, 1, 128, 128)

# Element-wise multiplication with input feature map x
x_out = x * α  # Shape: (batch_size, 64, 128, 128)
```
