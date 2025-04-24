# CIFAR-10 Classification with ResNet-inspired CNN

A modified ResNet-style convolutional neural network for image classification on CIFAR-10, implemented in PyTorch with residual connections and advanced training techniques.

## Key Enhancements (ResNet-inspired)
- **Residual Blocks** with identity mapping shortcuts
- **Channel Expansion** in downsampling layers (1x1 convolutions)
- **Strided Convolutions** for spatial reduction
- **Bottleneck-style** architecture with sequential conv layers

## Architecture Details
- **3 Residual Blocks** with increasing filters (64→128→256)
- **Dual-path Processing** (main path + shortcut connection)
- **Channel Matching** through 1x1 convolutional shortcuts
- **Progressive Downsampling** (32x32 → 16x16 → 8x8 → 1x1)

## Residual Block Structure
```python
# Example block implementation
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, 3, padding=1),
    nn.BatchNorm2d(out_channels)
)
# With matching shortcut
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1, stride=2),
    nn.BatchNorm2d(out_channels)
)