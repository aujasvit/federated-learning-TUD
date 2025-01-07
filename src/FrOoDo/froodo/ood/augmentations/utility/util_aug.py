import random

from ..utils import init_augmentation
from .. import Augmentation
from ....data.samples import Sample


class ProbabilityAugmentation(Augmentation):
    def __init__(self, augmentation, prob=0.5) -> None:
        super().__init__()
        assert isinstance(augmentation, Augmentation)
        self.augmentation = augmentation
        self.prob = prob

    def __call__(self, sample: Sample) -> Sample:

        sample = init_augmentation(sample)
        if random.random() >= self.prob:
            return sample
        return self.augmentation(sample)


class Nothing(Augmentation):
    def __init__(self, create_ood_mask: bool = False) -> None:
        super().__init__()
        self.create_ood_mask = create_ood_mask

    def __call__(self, sample: Sample) -> Sample:
        sample = init_augmentation(sample, self.create_ood_mask)
        return sample


class NTimesAugmentation(Augmentation):
    def __init__(self, augmentation: Augmentation, n) -> None:
        super().__init__()
        self.augmentation = augmentation
        self.n = n

    def __call__(self, sample: Sample) -> Sample:
        for _ in range(self.n):
            sample = self.augmentation(sample)
        return sample
