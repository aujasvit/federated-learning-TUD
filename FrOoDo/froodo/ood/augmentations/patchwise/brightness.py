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


class BrightnessAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        brightness=1.5,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        if sample_intervals == None:
            self.sample_intervals = [[(0.01, 0.3)], [(1.6, 4)]]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "brightness",
                (self.sample_intervals[0][0][0], self.sample_intervals[-1][-1][1]),
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    @property
    def brightness(self):
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        if type(value) == tuple:
            self._brightness = value
        elif type(value) == float or type(value) == int:
            assert value > 0
            self._brightness = (value, value)
        self.jitter = ColorJitter(brightness=self.brightness)

    def _get_parameter_dict(self):
        return {"brightness": self.brightness[0]}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"brightness": random.choice(self.sample_intervals)}
        )

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.jitter(sample["image"])
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample

class PartialBrightnessAugmentation(PartialOODAugmentaion):
    def __init__(self, brightness=1.5, sample_intervals=None, severity=None, 
                 keep_ignorred=True, mode="linear", base_severity: SeverityMeasurement = None, 
                 probability=0.8):
        
        brightness_base = BrightnessAugmentation(
            brightness=brightness, 
            sample_intervals=sample_intervals, 
            severity=severity, 
            keep_ignorred=keep_ignorred
        )
        
        self.sampled_augmentation = SampledAugmentation(brightness_base, probability)
        
        super().__init__(base_augmentation=self.sampled_augmentation, mode=mode, severity=base_severity)
