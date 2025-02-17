import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FiLMBlock(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) block """
    def __init__(self, context_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(context_dim, num_features)
        self.beta = nn.Linear(context_dim, num_features)

    def forward(self, x, context):
        gamma = self.gamma(context).unsqueeze(-1).unsqueeze(-1)
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

        # Get the number of feature channels dynamically
        encoder_channels = self.unet.encoder.out_channels  # Example: [3, 64, 64, 128, 256, 512]

        # Create FiLM layers dynamically
        self.film_blocks = nn.ModuleList([
            FiLMBlock(context_dim, num_features) for num_features in encoder_channels[1:]
        ])

    def forward(self, image, context):
        """ Forward pass: f(image, context) -> image """
        encoder_features = self.unet.encoder(image)

        # Apply FiLM modulation at each level (skipping first input feature)
        for i in range(1, len(encoder_features)):
            encoder_features[i] = self.film_blocks[i - 1](encoder_features[i], context)

        # Decode the modified feature maps
        output_image = self.unet.decoder(*encoder_features)
        output_image = self.unet.segmentation_head(output_image)

        return output_image


# # Example usage
# image = torch.randn(1, 3, 32, 32)  # CIFAR-like image
# context = torch.randn(1, 10)  # Example 10D context vector
#
# model = FiLMUNet()
# output_image = model(image, context)
# print(output_image.shape)  # Should be [1, 3, 32, 32]
