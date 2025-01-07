import random
from os import listdir
from os.path import join

import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import GaussianBlur, Resize
from random import randrange
from copy import deepcopy
from typing import Optional, List, Tuple, Type

from ....augmentations import OODAugmentation, SamplableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.metadata import *
from ...utils import *
from ...utility import Erosion2d, Dilation2d
from .....data.samples import Sample
from ....augmentations.indistribution.rotate import InRotateImg



#op100, hard98
#op51, hard44 3 times!
class BubbleAugmentation(ArtifactAugmentation, OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        base_augmentation: Type[torch.nn.Module] = GaussianBlur(kernel_size=(19, 19),sigma=10),
        severity: SeverityMeasurement = None,
        mask_threshold: float =0.1,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        keep_ignorred: bool =True,
    ) -> None:
        super().__init__()
        self.base_augmentation = base_augmentation
        self.mask_threshold = mask_threshold
        self.overlay_size = (1053,811)
        self.overlay_h = None
        self.overlay_w = None
        self.in_out_blur = random.choice([True, False])
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = {"overlay_h": [(self.overlay_size[0]//3,self.overlay_size[0])], "overlay_w": [(self.overlay_size[1]//3,self.overlay_size[1])],"in_out_blur": [(0,1)]}
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred
    
    def _get_parameter_dict(self):
        return {"overlay_h": self.overlay_h,"overlay_w": self.overlay_w, "in_out_blur": self.in_out_blur}

    def _apply_sampling(self):
        self.in_out_blur = random.choice([True, False])
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"overlay_h": self.sample_intervals["overlay_h"], "overlay_w": self.sample_intervals["overlay_w"]}, True
        )

    def _augment(self, sample: Sample) -> Sample:
        # rotate image to start folding from random side
        rotate = InRotateImg()
        sample = rotate(sample)

        #set up transforms
        _, h, w = sample.image.shape
        toPIL = T.ToPILImage()
        toTensor = T.ToTensor()

        #get overlay (should have alpha value)
        # overlay_path=join(
        #         data_folder,
        #         # "froodo/ood/augmentations/pathology/artifacts/imgs/bubble",
        #         random.choice(
        #             listdir(join(data_folder,"bubble"))
        #         ))
        overlay_path=join(
                data_folder,
                f"bubble/{random.choice(listdir(join(data_folder,'bubble')))}",
            )
        overlay = Image.open(overlay_path)

        # random crop overlay
        x1 = randrange(0, self.overlay_size[0] - self.overlay_h)
        y1 = randrange(0, self.overlay_size[1] - self.overlay_w)
        overlay = overlay.crop((x1, y1, x1 + self.overlay_h, y1 + self.overlay_w))

        #resize overlay to image size and add to base image
        overlay = overlay.resize((h,w))
        img = toPIL(sample["image"])
        img.paste(overlay,(0,0),overlay)
        img = toTensor(img)

        #prepare further computation
        overlay = toTensor(overlay)
        copied_img = deepcopy(img)
        mask = deepcopy(sample["ood_mask"]).float()

        #coinflip on wether to apply the augmentation inside or outside of bubble
        if self.in_out_blur:
            img[:,overlay[3,:,:] > self.mask_threshold] = self.base_augmentation(copied_img)[:,overlay[3,:,:] > self.mask_threshold]
            mask[overlay[3,:,:] > self.mask_threshold] = 0.0 
        else:
            img[:,overlay[3,:,:] < self.mask_threshold] = self.base_augmentation(copied_img)[:,overlay[3,:,:] < self.mask_threshold]
            mask[overlay[3,:,:] < self.mask_threshold] = 0.0
        
        #set sample image
        sample["image"] = img        

        # Apply closing operation to mask to remove small gap between blur and bubble edge
        mask[overlay[3,:,:] > 0.63] = 0.0 
        with torch.no_grad():
            if h//21 > 1:
                d=Dilation2d(1, 1, h//21, soft_max=False)
                e=Erosion2d(1, 1, h//21, soft_max=False)
                t=d(e(mask.unsqueeze(0).unsqueeze(0)))
                sample["ood_mask"]=Resize((h,w))(t).squeeze(0).squeeze(0)
        sample["ood_mask"][mask==0.0] = 0.0

        # reverse rotation
        sample.inverse = -1
        sample = rotate(sample)
        del sample.inverse
        return sample