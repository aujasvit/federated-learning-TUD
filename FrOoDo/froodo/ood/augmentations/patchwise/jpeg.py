import numpy as np
import PIL.Image
import torch
from torchvision.transforms import ToTensor
from io import StringIO

import tempfile
import imageio

from ...augmentations import OODAugmentation, SamplableAugmentation, InCrop, InResize
from ....data import Sample
from ...severity import ParameterSeverityMeasurement, SeverityMeasurement


class JPEGAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        quality: float = 50,
        sample_intervals=None,
        severity: SeverityMeasurement = None,
        keep_ignorred=True,
    ) -> None:
        super().__init__()
        assert (
            0 <= quality <= 100
        ), "Expected quality to be in the interval [0, 100], " "got %.4f." % (quality,)
        self.quality = quality
        if sample_intervals == None:
            self.sample_intervals = [(0, 100)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "quality", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred
        self.transform = ToTensor()

    def _get_parameter_dict(self):
        return {"quality": self.quality}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"quality": self.sample_intervals}
        )

    def _augment(self, sample: Sample) -> Sample:

        image = sample.image.permute(1, 2, 0).numpy()

        has_no_channels = image.ndim == 2
        is_single_channel = image.ndim == 3 and image.shape[-1] == 1
        if is_single_channel:
            image = image[..., 0]

        image_pil = PIL.Image.fromarray((image * 255.0).astype(np.uint8))

        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".jpg") as f:
            image_pil.save(f, subsampling=0, quality=round(self.quality), format="JPEG")

            f.seek(0)
            pilmode = "RGB"
            if has_no_channels or is_single_channel:
                pilmode = "L"
            image = imageio.imread(f, pilmode=pilmode, format="jpeg")
        if is_single_channel:
            image = image[..., np.newaxis]
        sample.image = (
            torch.from_numpy(image).permute(2, 0, 1) / 255.0
        )  # self.transform(image)

        return sample
