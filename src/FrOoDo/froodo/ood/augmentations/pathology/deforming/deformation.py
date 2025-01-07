
from xmlrpc.client import FastUnmarshaller
import elasticdeform.torch as ed
import numpy as np
import cv2

import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from torchvision.transforms import Resize, GaussianBlur, RandomAffine 
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F
from torchvision.io import read_image 
from torchmetrics.functional.image import image_gradients
import random
from typing import Optional, List, Tuple

from ....augmentations import OODAugmentation, SamplableAugmentation
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....data.datatypes import DistributionSampleType, OODReason
from .....data.metadata import *
from ...utils import *
from ...utility import Erosion2d, Dilation2d
from .....data.samples import Sample
from ....augmentations.indistribution.rotate import InRotateImg

# grad intensity = param for tweaking the intensity of the color changes, i.e. brightness
# color_thr = threshold from 0-1 for the whiteness underneath the color changes get applied. Is calulated in HLS color scheme
#0.4-0.8 for squeeze
#2.0/5.0 - 0.95
class DeformationAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        severity: SeverityMeasurement = None,
        grid_points: int = 20,
        grad_intensity: float = 0.4,
        mode: str = "stretch",
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        deformation_threshold: float = 1.0,
        keep_ignorred: bool=True,
    ) -> None:
        super().__init__()

        self.grid_points = grid_points
        self.grad_intensity = grad_intensity
        self.color_thr = 0.95 if mode =="stretch" else 0.8
        self.mode = mode
        self.deformation_threshold = deformation_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = {"grid_points": [(1,25)], "grad_intensity": [[(0.2, 1)], [(1,30)]]} if mode =="squeeze" else {"grid_points": [(2,6)], "grad_intensity":  [[(1, 6)], [(1,6)]]} #"grid_points": [(2,20)], "grad_intensity":  [[(0.2, 1)], [(1,6)]]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred

    def param_intervals(self):
        return self.sample_intervals
    
    def _get_parameter_dict(self):
        return {"grid_points": self.grid_points, "grad_intensity": self.grad_intensity}

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"grid_points": self.sample_intervals["grid_points"], "grad_intensity": random.choice(self.sample_intervals["grad_intensity"])}, False
        )

    def _augment(self, sample: Sample) -> Sample:
        with torch.no_grad():
            self.grid_points = int(self.grid_points)

            # rotate image to start folding from random side
            rotate = InRotateImg()
            sample = rotate(sample)

            # get image, create clone and resize for deformation
            _, h, w = sample.image.shape
            real = sample.image.clone()
            max_grad = int(h/self.grid_points)
            resize = Resize((h,w))

            # load random deformation image
            if self.mode == "stretch" or self.mode == "squeeze":
                path=join(
                    f"/data/imgs/{self.mode}",
                    # f"froodo/ood/augmentations/pathology/deforming/imgs/{self.mode}",
                    random.choice(
                        listdir(f"/data/imgs/{self.mode}")
                    ),
                )
            else:
                raise ValueError('Selected mode is invalid. Please select from "stretch" or "squeeze"')
            aug = read_image(path).int()

            #119 is grey in the test images, change with mean later maybe
            #substract mean, center around 0 and convert so size (h,w)
            aug -= 119
            aug = aug / 255 * max_grad
            aug = resize(aug)
            aug = aug[1:,:,:]
            aug[1,:,:] = 0

            # skip deformation for high intensities, when its not visible anyway
            if (self.grad_intensity < self.deformation_threshold):
                step_h = int(h/self.grid_points)
                step_w = int(w/self.grid_points)

                grid = aug[:,::step_h,::step_w]

                img = ed.deform_grid(sample["image"],grid, axis=(1,2))
            else:
                img = sample["image"]

            # prepare deformation intensity counters
            cum_grad_top = torch.sum(torch.abs(aug[0,:int(h/2),int(w/2)]))
            cum_grad_bot = torch.sum(torch.abs(aug[0,int(h/2):,int(w/2)]))
            cum_top, cum_bot, cum_sum_top, cum_sum_bot = 0, 0, 0, 0

            #convert into hls color space to detect lightness easier
            hls_img = cv2.cvtColor(img.permute(1,2,0).numpy(),cv2.COLOR_RGB2HLS)
            hls_img = torch.from_numpy(hls_img)

            #apply stain transformation to simulate cutting/compression
            for i in range(int(h/2)):
                cum_top += torch.abs(aug[0,i,int(w/2)]) / cum_grad_top
                cum_bot += torch.abs(aug[0,h-1-i,int(w/2)]) / cum_grad_bot

                #skip calculations where gradient isnt large enough to notice
                if((cum_top or cum_bot) < 0.1 or torch.isnan(cum_bot) or torch.isnan(cum_top)):
                    continue

                #cumulated level of deformation (to center) * the intensity param. use this? top solution kinda shitty for squeeze
                cum_sum_top += torch.abs(aug[0,i,int(w/2)]) / cum_grad_top * self.grad_intensity
                cum_sum_bot += torch.abs(aug[0,h-1-i,int(w/2)]) / cum_grad_bot * self.grad_intensity

                color_threshold_top = (hls_img[i,:,1]<self.color_thr).unsqueeze(0).repeat(3,1)
                color_threshold_bot = (hls_img[h-1-i,:,1]<self.color_thr).unsqueeze(0).repeat(3,1)

                if self.mode == "squeeze": # good example: grad_intensity=1.0,color_thr=0.8
                    #saturation
                    img[:,i,:] = F.adjust_saturation(img[:,i,:].unsqueeze(1),1+cum_sum_top).squeeze()
                    img[:,h-1-i,:] = F.adjust_saturation(img[:,h-1-i,:].unsqueeze(1),1+cum_sum_bot).squeeze()


                    #Brightness
                    brightnessTop = (1-cum_sum_top*0.3) if (1-cum_sum_top*0.3) > 0.3 else 0.3
                    brightnessBot = (1-cum_sum_bot*0.3) if (1-cum_sum_bot*0.3) > 0.3 else 0.3
                    img[:,i,:][color_threshold_top] = F.adjust_brightness(img[:,i,:][color_threshold_top].unsqueeze(1),brightnessTop).squeeze()
                    img[:,h-1-i,:][color_threshold_bot] = F.adjust_brightness(img[:,h-1-i,:][color_threshold_bot].unsqueeze(1),brightnessBot).squeeze()



                elif self.mode == "stretch": # good example: grad_intensity=2.0,color_thr=0.95
                    #saturation
                    intensityTop = (1-cum_sum_top*0.3) if (1-cum_sum_top*0.3) > 0.3 else 0.3
                    intensityBot = (1-cum_sum_bot*0.3) if (1-cum_sum_bot*0.3) > 0.3 else 0.3
                    img[:,i,:] = F.adjust_saturation(img[:,i,:].unsqueeze(1),intensityTop).squeeze()
                    img[:,h-1-i,:] = F.adjust_saturation(img[:,h-1-i,:].unsqueeze(1),intensityBot).squeeze()

                    #Brightness
                    img[:,i,:][color_threshold_top] = F.adjust_brightness(img[:,i,:][color_threshold_top].unsqueeze(1),1+cum_sum_top).squeeze()
                    img[:,h-1-i,:][color_threshold_bot] = F.adjust_brightness(img[:,h-1-i,:][color_threshold_bot].unsqueeze(1),1+cum_sum_bot).squeeze()
                        
            # calculate differences to OG image
            diff = torch.abs(torch.subtract(real,img))
            diff = diff.mean(dim=0)
            diff = GaussianBlur(15, sigma=(50,50))(diff.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            # init mask
            mask = torch.ones((h,w)).float()
            # print("diff",diff.mean(),diff.max(),diff.min())
            # create mask for ood detection (squeeze: 0.1, stretch: 0.15)
            if self.mode == "squeeze":
                if h > 50:
                    mask[diff>0.13] = 0.0
                    # mask for ultra low diff
                    if diff.mean() < 0.12:
                        mask[diff>0.07]  = 0.0
                else:
                    mask[diff>0.18] = 0.0
                    # mask for ultra low diff
                    if diff.mean() < 0.13:
                        mask[diff> 0.09]  = 0.0
                    if diff.mean() < 0.06:
                        mask[diff> 0.072]  = 0.0
                    if diff.mean() < 0.03:
                        mask[diff> 0.054]  = 0.0
            else: 
                if h > 50:       
                    mask[diff>0.15] = 0.0
                    # mask for ultra low diff
                    if diff.mean() < 0.05:
                        mask[diff>1e-6]  = 0.0
                else:
                    mask[diff>0.16] = 0.0
                    # mask for ultra low diff
                    if diff.mean() < 0.057 and diff.mean() > 1e-5:
                        # print(img.mean(0).shape, img.mean(0).max(), img.mean(0).min(), img.mean(0).mean())
                        mask[img.mean(0) > 0.94] = 0.0
                        # mask[diff>0.075]  = 0.0
            if (self.grad_intensity < self.deformation_threshold):
                aug = aug[0,:,:]
                aug -= aug.min()
                aug /= aug.max()
                # use gradients to detect largest stretched areas
                aug_x, aug_y = image_gradients(aug.unsqueeze(0).unsqueeze(0))
                mask[aug_x.squeeze(0).squeeze(0) < -1.0] = 0.0
                mask[aug_y.squeeze(0).squeeze(0) < -1.0] = 0.0
                # use closing of holes to better fit OOD areas
                with torch.no_grad():
                    if h//50 > 5:
                        d=Dilation2d(1, 1, h//50, soft_max=False)
                        e=Erosion2d(1, 1, h//50, soft_max=False)
                        mask=e(d(mask.unsqueeze(0).unsqueeze(0))).squeeze(0).squeeze(0)
            mask = resize(mask.unsqueeze(0)).squeeze(0)
            sample["ood_mask"][mask==0.0] = 0.0
            sample["image"] = img

            sample.image = torch.clip(sample.image,0,1)
            # reverse rotation
            sample.inverse = -1
            sample = rotate(sample)
            del sample.inverse
        return sample
    
class CompressionAugmentation(DeformationAugmentation):
    def __init__(
        self,
        severity: SeverityMeasurement = None,
        grid_points: int = 20,
        grad_intensity: float = 0.4,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        deformation_threshold: float = 1.0,
        keep_ignorred: bool=True,
    ) -> None:
        super().__init__(
            severity=severity,
            grid_points=grid_points,
            grad_intensity=grad_intensity,
            mode="squeeze",  # Set mode to "squeeze" by default
            sample_intervals=sample_intervals,
            deformation_threshold=deformation_threshold,
            keep_ignorred=keep_ignorred,
        )

class CutsAugmentation(DeformationAugmentation):
    def __init__(
        self,
        severity: SeverityMeasurement = None,
        grid_points: int = 20,
        grad_intensity: float = 0.4,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        deformation_threshold: float = 1.0,
        keep_ignorred: bool=True,
    ) -> None:
        super().__init__(
            severity=severity,
            grid_points=grid_points,
            grad_intensity=grad_intensity,
            mode="stretch",  # Set mode to "stretch" by default
            sample_intervals=sample_intervals,
            deformation_threshold=deformation_threshold,
            keep_ignorred=keep_ignorred,
        )

class RealOODAugmentation(OODAugmentation, SamplableAugmentation):
    def __init__(
        self,
        scale: float = 1.0,
        sample_intervals: Optional[List[Tuple[float, float]]] = None,
        severity: SeverityMeasurement = None,
        keep_ignorred: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = {"scale": [(0.5, 1.5)]} #"grid_points": [(2,20)], "grad_intensity":  [[(0.2, 2)], [(2,6)]]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred

    def param_intervals(self):
        return self.sample_intervals
    
    def _get_parameter_dict(self):
        return {"scale": self.scale}
    
    def _apply_sampling(self):
        super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": [(0.5, 1.5)]}, False
        )

    def _augment(self, sample: Sample) -> Sample:
        # print(sample["ood_mask"].unique())
        return sample
    
    def _set_metadata(self, sample: Sample) -> Sample:

        is_ood = True

        if hasattr(self, "severity_class"):
            severity = self._set_severity(sample)
            if severity.get_bin(ignore_true_bin=True) == -1:
                is_ood = False
                sample["metadata"][SampleMetadataCommonTypes.OOD_SEVERITY.name] = None
                sample["metadata"].type = DistributionSampleType.IN_DATA
            else:
                sample["metadata"][
                    SampleMetadataCommonTypes.OOD_SEVERITY.name
                ] = severity

        if is_ood:
            sample["metadata"].type = DistributionSampleType.OOD_DATA
            sample["metadata"][
                SampleMetadataCommonTypes.OOD_REASON.name
            ] = OODReason.AUGMENTATION_OOD
            sample["metadata"][SampleMetadataCommonTypes.OOD_AUGMENTATION.name] = type(
                self
            ).__name__

        return sample