from torch import Tensor

from typing import Callable, Tuple
import random

from .. import INAugmentation
from ....data.samples import Sample


class TVNormalize(INAugmentation):
    def __init__(self, torchvision_transform: Callable) -> None:
        self.augmentation = torchvision_transform

    def _augment(self, sample: Sample) -> Sample:
        sample.image = self.augmentation(sample.image)
        return sample
