import torch
import numpy as np

from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    DivisiblePadd
)

from monai.apps.detection.transforms.dictionary import (
    BoxToMaskd,
    ClipBoxToImaged,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandZoomBoxd,
)

def generate_train_transform(config):
    train_transform = Compose(
        [
            LoadImaged(keys=['image'], dtype=np.float32, reader="NumpyReader"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=['label'], dtype=torch.long),
            RandCropBoxByPosNegLabeld(
                image_keys=["image"],
                box_keys="box",
                label_keys='label',
                spatial_size=config['crop_size'],
                whole_box=True,
                num_samples=config['batch_size'],
                allow_smaller=True,
                pos=3,
                neg=1,
            ),
            RandZoomBoxd(
                    image_keys=["image"],
                    box_keys=["box"],
                    box_ref_image_keys=["image"],
                    prob=0.2,
                    min_zoom=0.7,
                    max_zoom=1.4,
                    padding_mode="constant",
                    keep_size=True,
            ),
            ClipBoxToImaged(
                box_keys="box",
                label_keys=["label"],
                box_ref_image_keys="image",
                remove_empty=True,
            ),
            RandFlipBoxd(
                image_keys=['image'],
                box_keys=["box"],
                box_ref_image_keys=['image'],
                prob=0.5,
                spatial_axis=0,
            ),
            BoxToMaskd(
                box_keys=['box'],
                label_keys=['label'],
                box_mask_keys=["box_mask"],
                box_ref_image_keys="image",
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=["image", "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=['box'],
                label_keys=['label'],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=['image'],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=['image'], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=['image'], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7, 1.5)),
        ]
    )
    return train_transform


def generate_val_transform():
    val_transform = Compose(
        [
            LoadImaged(keys=['image'], dtype=np.float32, reader="NumpyReader"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "box"], dtype=torch.float32),
            EnsureTyped(keys=['label'], dtype=torch.long),
            RandCropBoxByPosNegLabeld(
                image_keys=["image"],
                box_keys="box",
                label_keys='label',
                spatial_size=(128, 128, 128),
                whole_box=True,
                num_samples=1,
                pos=3,
                neg=1,
            ),
        ]
    )
    return val_transform

def generate_infer_transform():
    infer_transform = Compose(
        [
            LoadImaged(keys=['image'], dtype=np.float32, reader="NumpyReader"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            DivisiblePadd(keys=['image'],
                          k=(16, 16, 16),
                          mode="constant",
                          method="end")
        ]
    )
    return infer_transform