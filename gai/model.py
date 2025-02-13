import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FiLMBlock(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) block """
    def __init__(self, context_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(context_dim, num_features)  # Scale factor
        self.beta = nn.Linear(context_dim, num_features)   # Shift factor

    def forward(self, x, context):
        gamma = self.gamma(context).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, C, 1, 1]
        beta = self.beta(context).unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta  # Apply FiLM modulation

class FiLMUNet(nn.Module):
    def __init__(self, hparams=None, encoder_name='resnet18', context_dim=10, num_classes=3):
        super().__init__()

        if hparams is not None:
            encoder_name = hparams.get('encoder_name', encoder_name)
            context_dim = hparams.get('context_dim', context_dim)
            num_classes = hparams.get('num_classes', num_classes)

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )

        # FiLM layers for context modulation at different feature map depths
        self.film1 = FiLMBlock(context_dim, num_features=64)
        self.film2 = FiLMBlock(context_dim, num_features=64)
        self.film3 = FiLMBlock(context_dim, num_features=128)
        self.film4 = FiLMBlock(context_dim, num_features=256)
        self.film5 = FiLMBlock(context_dim, num_features=512)

    def forward(self, image, context):
        """ Forward pass: f(image, context) -> image """

        # U-Net encoder (extracts feature maps)
        encoder_features = self.unet.encoder(image)

        # Apply FiLM modulation at different levels
        encoder_features[1] = self.film1(encoder_features[1], context)
        encoder_features[2] = self.film2(encoder_features[2], context)
        encoder_features[3] = self.film3(encoder_features[3], context)
        encoder_features[4] = self.film4(encoder_features[4], context)
        encoder_features[5] = self.film5(encoder_features[5], context)

        # Decode the modified feature maps
        output_image = self.unet.decoder(*encoder_features)

        return output_image

# # Example usage
# image = torch.randn(1, 3, 32, 32)  # CIFAR-like image
# context = torch.randn(1, 10)  # Example 10D context vector
#
# model = FiLMUNet()
# output_image = model(image, context)
# print(output_image.shape)  # Should be [1, 3, 32, 32]
