from numpy import full
import numpy as np
from torchvision.transforms import GaussianBlur, CenterCrop, Resize
import torch

from copy import deepcopy


from .. import OODAugmentation, SamplableAugmentation
from ....data import Sample
from ....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from ..utils import full_image_ood
from ...augmentations.indistribution.rotate import InRotateImg


class SliceThicknessAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        sigma=3,
        kernel_size=(19, 19),
        offset=0.5,
        thickness_proportion = 0.5,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.offset = offset
        self.thickness_proportion = thickness_proportion if thickness_proportion <= 1.0 else 1.0
        #[sigma, proportion]
        if sample_intervals == None:
            self.sample_intervals ={"thickness_proportion":[(0.5, 1.0)], "offset":[(0.0,1.)]}
        else:
            self.sample_intervals = sample_intervals
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )

        self.keep_ignorred = keep_ignorred

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value
        self.augmentation = GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"thickness_proportion": self.sample_intervals["thickness_proportion"], "offset": self.sample_intervals["offset"]}, False
        )

    def _get_parameter_dict(self):
        return {"thickness_proportion":self.thickness_proportion, "offset": self.offset}

    def _augment(self, sample: Sample) -> Sample:
        # get parameters
        _,h,w = sample["image"].shape
        
        proportion = self.thickness_proportion
        id_set = int(np.random.uniform(low=0.0, high=1.0-self.offset)* w)
        self.offset = int(self.offset * w)
        # rotate image to start folding from random side
        rotate = InRotateImg()
        sample = rotate(sample)
        # take copy to simulate the increased thickness
        copied_img = deepcopy(sample)
        # blur as it will be always a bit out of focus
        copied_img["image"] = self.augmentation(copied_img["image"])
        # create mask for thickness
        mask = torch.full_like(copied_img["image"],0.0)
        mask[:,:,:self.offset] = 1.0
        mask[:,:,self.offset:(w-id_set)] = torch.linspace(1.,0.,w-self.offset-id_set).unsqueeze(0).unsqueeze(0).repeat(1,h,1) 
        mask[:,:,(w-id_set):] = 0.0
        tmp = mask + 1.0
        # apply thickness
        sample["image"] = (sample["image"] - (proportion) * (copied_img["image"]*(mask)))/tmp
        # create ood mask
        full_image_ood(sample, self.keep_ignorred)
        sample["ood_mask"][(proportion *(mask)).mean(0) <= 0.1] = 1
        # clip image...
        sample.image = torch.clip(sample.image,0,1)
        # reverse rotation
        sample.inverse = -1
        sample = rotate(sample)
        del sample.inverse
        return sample
