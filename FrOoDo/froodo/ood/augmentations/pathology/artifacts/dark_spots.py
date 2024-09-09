import random
from os import listdir
from os.path import join
from typing import Optional, List, Tuple


from ....augmentations import OODAugmentation, SamplableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class DarkSpotsAugmentation(
    ArtifactAugmentation, OODAugmentation, SamplableAugmentation
):
    def __init__(
        self,
        scale: float = 1,
        severity: SeverityMeasurement = None,
        path: str = join(data_folder, "dark_spots/small_spot.png"),
        mask_threshold: float = 0.5,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        keep_ignorred: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.path = path
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(0.1, 5)]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:
        img, mask = super().transparentOverlay(
            sample["image"],
            sample["ood_mask"],
            scale=self.scale,
            mask_threshold=self.mask_threshold,
            overlay_path=join(
                data_folder,
                f"dark_spots/{random.choice(listdir(join(data_folder,'dark_spots')))}",
            )
            if self.path == None
            else self.path,
            width_slack=(-0.1, -0.1),
            height_slack=(-0.1, -0.1),
            ignore_index=None
            if not self.keep_ignorred
            else sample.metadata["ignore_index"],
        )
        sample["image"] = img
        sample["ood_mask"] = mask
        return sample
