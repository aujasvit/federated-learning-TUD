from .. import OODAugmentation, SamplableAugmentation
from ...severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from ....data.metadata import *
from ..utils import *
from ....data.samples import Sample


class RealOODAugmentation3(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        scale: float = 1.0,
        severity: SeverityMeasurement = None,
        keep_ignorred: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )

        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": [(0.5, 1.5)]}
        )

    def _augment(self, sample: Sample) -> Sample:
        # print("RealOODAugmentation")
        # print(sample.metadata["image_path"])
        # raise NotImplementedError("Please Implement this method")
        return sample
