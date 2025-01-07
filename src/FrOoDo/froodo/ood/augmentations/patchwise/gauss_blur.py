from numpy import full
from torchvision.transforms import GaussianBlur
import torch

from copy import deepcopy

from ...augmentations import OODAugmentation, SamplableAugmentation, SampledAugmentation, PartialOODAugmentaion
from ....data import Sample
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ..utils import full_image_ood


class GaussianBlurAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        sigma=5,
        kernel_size=(19, 19),
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        if sample_intervals == None:
            self.sample_intervals = [(10, 50)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "sigma", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
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
            {"sigma": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"sigma": self.sigma}

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.augmentation(sample["image"])
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample

class PartialGaussianBlurAugmentation(PartialOODAugmentaion):
    def __init__(self, sigma=5, kernel_size=(19, 19), sample_intervals=None, severity=None, 
                 keep_ignorred=True, mode="linear", base_severity: SeverityMeasurement = None, 
                 probability=0.8):
        
        gaussian_blur_base = GaussianBlurAugmentation(
            sigma=sigma, 
            kernel_size=kernel_size, 
            sample_intervals=sample_intervals, 
            severity=severity, 
            keep_ignorred=keep_ignorred
        )
        
        self.sampled_augmentation = SampledAugmentation(gaussian_blur_base, probability)
        
        super().__init__(base_augmentation=self.sampled_augmentation, mode=mode, severity=base_severity)

    # def __call__(self, sample: Sample):
    #     return self.sampled_augmentation(sample)


