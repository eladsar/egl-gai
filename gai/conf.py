from beam.config import ExperimentConfig, BeamParam


# encoder_name="resnet18", context_dim=10, num_classes=3):
class EGLExperimentConfig(ExperimentConfig):
    parameters = [
        BeamParam('encoder_name', str, 'resnet18', 'Name of the encoder architecture'),
        BeamParam('context_dim', int, 10, 'Dimensionality of the context vector'),
        BeamParam('num_classes', int, 3, 'Number of output classes (should be 3 for RGB images)'),
        BeamParam('eps-max', float, 0.5, 'Maximum value of epsilon'),
        BeamParam('eps-min', float, 0.001, 'Minimum value of epsilon'),
    ]