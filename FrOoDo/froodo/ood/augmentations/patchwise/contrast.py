import torch
from torchvision.transforms import ColorJitter

import random

from ...augmentations import (
    OODAugmentation,
    ProbabilityAugmentation,
    PickNComposite,
    SamplableAugmentation,
    SampledAugmentation,
    PartialOODAugmentaion,
)
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement
from ....data import Sample
from ..utils import full_image_ood


class ContrastAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        contrast=1.5,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.contrast = contrast
        if sample_intervals == None:
            self.sample_intervals = [[(0.01, 0.3)], [(1.7, 2)]]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "contrast",
                (self.sample_intervals[0][0][0], self.sample_intervals[-1][-1][1]),
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        if type(value) == tuple:
            self._contrast = value
        elif type(value) == float or type(value) == int:
            assert value > 0
            self._contrast = (value, value)
        self.jitter = ColorJitter(contrast=self.contrast)

    def _get_parameter_dict(self):
        return {"contrast": self.contrast[0]}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"contrast": random.choice(self.sample_intervals)}
        )

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.jitter(sample["image"])
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample


class PartialContrastAugmentation(PartialOODAugmentaion):
    def __init__(self, contrast=1.5, sample_intervals=None, severity=None, 
                 keep_ignorred=True, mode="linear", base_severity: SeverityMeasurement = None, 
                 probability=0.8):
        
        contrast_base = ContrastAugmentation(
            contrast=contrast, 
            sample_intervals=sample_intervals, 
            severity=severity, 
            keep_ignorred=keep_ignorred
        )
        
        self.sampled_augmentation = SampledAugmentation(contrast_base, probability)
        
        super().__init__(base_augmentation=self.sampled_augmentation, mode=mode, severity=base_severity)
