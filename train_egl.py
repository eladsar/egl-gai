from gai.conf import EGLExperimentConfig
from gai.alg import ExplicitGradientGenerativeAlgorithm
from gai.dataset import CIFAR10Dataset
from beam import Experiment, logger


if __name__ == '__main__':
    conf = EGLExperimentConfig()
    dataset = CIFAR10Dataset(conf)
    exp = Experiment(conf)
    alg = exp.fit(alg=ExplicitGradientGenerativeAlgorithm, dataset=dataset)
    logger.info('Training finished')

