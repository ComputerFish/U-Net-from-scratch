# U-Net Implementation from Scratch

A comprehensive implementation of U-Net and hybrid ResNet+U-Net models for semantic segmentation, built from scratch using PyTorch.


## Overview

This repository contains a from-scratch implementation of U-Net, a fully convolutional network designed for semantic segmentation tasks. Additionally, we provide a hybrid architecture that combines ResNet backbone with U-Net decoder for improved feature extraction and segmentation performance.

U-Net was introduced in the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Ronneberger et al. (2015), and has become a standard architecture for image segmentation in medical imaging and other domains.

## Architecture

### U-Net

The classical U-Net architecture consists of:

- **Encoder (Contracting Path)**: A series of convolutional blocks that progressively downsample the input image while increasing the number of feature channels. Each block typically contains:
  - Two 3×3 convolutional layers
  - ReLU activation functions
  - Max pooling (2×2) for downsampling

- **Bottleneck**: The deepest layer where the spatial dimensions are smallest but feature richness is highest

- **Decoder (Expanding Path)**: A series of upsampling blocks that progressively upsample the feature maps while decreasing the number of channels. Each block contains:
  - Upsampling (transposed convolution or interpolation)
  - Concatenation with corresponding encoder feature maps (skip connections)
  - Two 3×3 convolutional layers
  - ReLU activation functions

- **Final Output Layer**: A 1×1 convolution that maps feature maps to the desired number of classes

#### Key Characteristics:
- **Skip Connections**: Direct connections between encoder and decoder at each level preserve spatial information and aid gradient flow
- **Symmetric Architecture**: The decoder mirrors the encoder structure
- **End-to-End Training**: Fully convolutional approach allowing variable input sizes (with padding considerations)

### ResNet+U-Net Hybrid

The hybrid model combines the strengths of both architectures:

- **Encoder**: ResNet backbone (ResNet18, ResNet34, ResNet50, etc.) for powerful feature extraction
  - Pre-trained weights can be optionally loaded for transfer learning
  - Extracts multi-scale features at different depth levels

- **Decoder**: U-Net-style decoder with skip connections
  - Receives feature maps from multiple ResNet stages
  - Progressive upsampling with feature concatenation
  - Reduces channels progressively toward output

#### Advantages:
- Better feature representation through ResNet's residual connections
- Leverages pre-trained ImageNet weights for improved performance
- Maintains the benefit of U-Net's skip connections for detailed segmentation
- More efficient training on limited datasets



## Usage

### Basic U-Net Model

```python
import torch
from models.unet import UNet

# Create a U-Net model
# Parameters: in_channels (input channels), out_channels (output classes), features (base feature count)
model = UNet(in_channels=3, out_channels=1, features=64)

# Forward pass
x = torch.randn(1, 3, 572, 572)  # Batch, Channels, Height, Width
output = model(x)
print(output.shape)  # torch.Size([1, 1, 388, 388])
```

### ResNet+U-Net Hybrid Model

```python
import torch
from models.resnet_unet import ResNetUNet

# Create ResNet+U-Net hybrid model
# Parameters: num_classes, pretrained, depth (18, 34, 50, etc.)
model = ResNetUNet(num_classes=1, pretrained=True, depth=50)

# Forward pass
x = torch.randn(1, 3, 512, 512)
output = model(x)
print(output.shape)  # torch.Size([1, 1, 512, 512])
```

### Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet

# Initialize model
model = UNet(in_channels=3, out_channels=1, features=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "unet_model.pth")
```

## Model Architecture Details

### U-Net Architecture

```
Input (572x572x3)
    ↓
[Conv 3x3 + ReLU] × 2 → (570x570x64)
    ↓ MaxPool 2x2
[Conv 3x3 + ReLU] × 2 → (284x284x128)
    ↓ MaxPool 2x2
[Conv 3x3 + ReLU] × 2 → (140x140x256)
    ↓ MaxPool 2x2
[Conv 3x3 + ReLU] × 2 → (68x68x512)
    ↓ MaxPool 2x2
[Conv 3x3 + ReLU] × 2 → (32x32x1024)  [Bottleneck]
    ↑
[UpConv] → concat with encoder feature map
[Conv 3x3 + ReLU] × 2 → (68x68x512)
    ↑
[UpConv] → concat with encoder feature map
[Conv 3x3 + ReLU] × 2 → (140x140x256)
    ↑
[UpConv] → concat with encoder feature map
[Conv 3x3 + ReLU] × 2 → (284x284x128)
    ↑
[UpConv] → concat with encoder feature map
[Conv 3x3 + ReLU] × 2 → (572x572x64)
    ↓
Conv 1x1 → (572x572x1)  [Output]
```

### ResNet+U-Net Hybrid Architecture

```
Input (512x512x3)
    ↓
ResNet Encoder (Pre-trained backbone)
├─ Layer 1: stride-1 → features[64]
├─ Layer 2: stride-2 → features[128]
├─ Layer 3: stride-4 → features[256]
└─ Layer 4: stride-8 → features[512]
    ↓
U-Net Decoder with Skip Connections
├─ Upsample + concat [512 features]
├─ [Conv blocks + ReLU]
├─ Upsample + concat [256 features]
├─ [Conv blocks + ReLU]
├─ Upsample + concat [128 features]
├─ [Conv blocks + ReLU]
├─ Upsample + concat [64 features]
├─ [Conv blocks + ReLU]
└─ Final Conv 1x1
    ↓
Output (512x512xnum_classes)
```

## Training

Refer to the `train.py` script for a complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Loss function selection
- Optimization strategies
- Validation and metric calculation
- Checkpoint saving

### Key Training Considerations:

1. **Loss Functions**:
   - Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for binary segmentation
   - Cross-Entropy Loss (CrossEntropyLoss) for multi-class segmentation
   - Dice Loss for imbalanced datasets

2. **Optimization**:
   - Adam optimizer recommended for most tasks
   - Learning rate: typically 1e-4 to 1e-3
   - Learning rate scheduling for improved convergence

3. **Data Augmentation**:
   - Random rotations
   - Elastic deformations
   - Brightness/contrast adjustments
   - Horizontal/vertical flips

4. **Batch Size**: 2-16 depending on GPU memory and image size

## References

- **Original U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- **ResNet Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition." CVPR 2015.

---

**Created by**: ComputerFish  
**Last Updated**: January 2026
