
from torchvision.transforms.functional import InterpolationMode, rotate
from torch import Tensor
import random

from typing import List, Tuple

from ..types import INAugmentation
from ....data.samples import Sample


class InRotate(INAugmentation):
    def __init__(self, angles: List[int] = [0,90,180,270]) -> None:
        super().__init__()
        self.angles = angles

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        angle = random.choice(self.angles)
        sample.image = rotate(sample.image.unsqueeze(0),angle=angle,interpolation=InterpolationMode.BILINEAR).squeeze()
        for k, v in sample.label_dict.items():
            sample[k] = rotate(v.unsqueeze(0).unsqueeze(0),angle=angle,interpolation=InterpolationMode.NEAREST).squeeze()
        return sample

class InRotateImg(INAugmentation):
    def __init__(self, angles: List[int] = [0,90,180,270]) -> None:
        super().__init__()
        self.angle = random.choice(angles)

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        if hasattr(sample, "inverse"):
            inverse = sample.inverse
        else:
            inverse = 1
        sample.image = rotate(sample.image.unsqueeze(0),angle=inverse*self.angle,interpolation=InterpolationMode.BILINEAR).squeeze()
        for k, v in sample.label_dict.items():
            if len(v.shape) == 2:
                sample[k] = rotate(v.unsqueeze(0).unsqueeze(0),angle=inverse*self.angle,interpolation=InterpolationMode.NEAREST).squeeze()
            elif len(v.shape) == 3:
                sample[k] = rotate(v.unsqueeze(0),angle=inverse*self.angle,interpolation=InterpolationMode.NEAREST).squeeze()
        return sample