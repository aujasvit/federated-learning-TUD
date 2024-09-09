from torch import Tensor
from skimage.filters import gaussian

from typing import Tuple, Optional, List
import random
import numpy as np
import torch
from torchvision.transforms import Resize


from ....augmentations import OODAugmentation, SizeAugmentation, SamplableAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample
from ....augmentations.indistribution.rotate import InRotateImg


class OverlapAugmentation(OODAugmentation, SizeAugmentation, SamplableAugmentation):
    def __init__(
        self,
        overlap=50,
        severity: SeverityMeasurement = None,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        keep_ignorred: bool=True,
    ) -> None:
        super().__init__()
        self.overlap = overlap
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(10, 100)]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred

    def _get_parameter_dict(self):
        return {"overlap": self.overlap}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"overlap": self.sample_intervals}, True
        )

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        # rotate image to start folding from random side
        rotate = InRotateImg()
        sample = rotate(sample)

        # get image
        per_img = sample.image.permute(1, 2, 0)


        start = int(
            random.uniform(
                2,
                per_img.shape[0] - 2 * self.overlap,
            )
        )

        part1 = torch.ones((per_img.shape[0] - self.overlap, per_img.shape[1], 3))
        part2 = torch.ones((per_img.shape[0] - self.overlap, per_img.shape[1], 3))

        # if flip:
        #     per_img = per_img.permute(1, 0, 2)

        shape = np.arange(per_img.shape[0])

        # create masks to add random noise at edges
        PIMask1 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PImask2 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PIMask1[start + self.overlap :, shape] = True
        PImask2[: start + self.overlap, ...] = True

        # to maintain format of the image
        PMask1 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask2 = torch.full((per_img.shape[0] - self.overlap, per_img.shape[1]), False)
        PMask1[start:, shape] = True
        PMask2[: start + self.overlap, ...] = True
        
        part1[PMask1] = per_img[PIMask1]
        part2[PMask2] = per_img[PImask2]

        # combine both parts with raytracing formular
        result = part1 * part2
 
        # Resize back to original size and scale start/overlap
        scale = per_img.shape[0]/ result.shape[0]
        start = int(start * scale)
        self.overlap = int(self.overlap * scale)
        result = Resize(per_img.shape[:2])(result.permute(2, 0, 1)).permute(1, 2, 0)

        # set mask and image
        if sample["ood_mask"] != None:
            sample["ood_mask"][start : start + self.overlap, :] = 0
            sample.image[:,start : start + self.overlap, :] = result.permute(2, 0, 1)[:,start : start + self.overlap, :]

        sample.image = torch.clip(sample.image,0,1)
        # reverse rotation
        sample.inverse = -1
        sample = rotate(sample)
        del sample.inverse
        return sample


class FoldingAugmentation(OODAugmentation, SizeAugmentation, SamplableAugmentation):
    def __init__(
        self,
        overlap: int = 50,
        severity: SeverityMeasurement = None,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        keep_ignorred: bool=True,
    ) -> None:
        super().__init__()
        self.overlap = overlap
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(10, 100)]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred

    def _get_parameter_dict(self):
        return {"overlap": self.overlap}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"overlap": self.sample_intervals}, True
        )

    def _augment(self, sample: Sample) -> Tuple[Tensor, Tensor]:
        # rotate image to start folding from random side
        rotate = InRotateImg()
        sample = rotate(sample)

        # get image
        per_img = sample.image.permute(1, 2, 0)        
        # we loose as much tissue as we fold over
        start = self.overlap

        # create two parts of the image: part1 is the part that is folded over, part2 is the part that is folded
        part1 = torch.ones((per_img.shape[0] - start, per_img.shape[1], 3))
        part2 = torch.ones((start + self.overlap, per_img.shape[1], 3))

        shape = np.arange(per_img.shape[0])

        # create masks to add random noise at edges
        PIMask1 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PImask2 = torch.full((per_img.shape[0], per_img.shape[1]), False)
        PIMask1[start :, shape] = True
        PImask2[: start, ...] = True

        # to maintain format of the image
        PMask2 = torch.full((start + self.overlap, per_img.shape[1]), False)
        PMask2[: start, ...] = True

        part1 = per_img[PIMask1].reshape((per_img.shape[0] - start, per_img.shape[1], 3))
        part2[PMask2] = per_img[PImask2]

        # fold part2 over part1
        part2 = torch.flip(part2, [0])
        
        # fill empty space with white: number is arbtary but more realistic than [1,1,1]
        white = torch.tensor([0.9882, 0.9594, 0.9722])
        part2[:start,...] = torch.normal(torch.zeros((start, per_img.shape[1], 3))+ white*0.90, 0.01*sum(white))

        # Pad to original size        
        part2_ = torch.ones((per_img.shape[0], per_img.shape[1], 3))
        part2_[:start+self.overlap,...] = part2
        part1_ = torch.ones((per_img.shape[0], per_img.shape[1], 3))
        part1_[start:, ...] = part1

        # combine both parts with raytracing formular
        result = part1_ * part2_

        if sample["ood_mask"] != None:
            sample["ood_mask"][ : start + self.overlap, :] = 0
            sample.image = result.permute(2, 0, 1)

        sample.image = torch.clip(sample.image,0,1)
        # reverse rotation
        sample.inverse = -1
        sample = rotate(sample)
        del sample.inverse
        return sample
