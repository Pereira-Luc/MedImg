import os
import yaml
import monai

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,  # <-- Use the dictionary-based transform
    ToTensord,
    CenterSpatialCropd
)
from monai.data import Dataset, DataLoader


def get_monai_dataloader(
        yaml_path: str,
        prefix: str,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 4
):
    """
    1) Reads a YAML file containing a list of patient folder names.
    2) Builds a list of data dictionaries for T1 + segmentation.
    3) Creates a MONAI Dataset/DataLoader.

    Args:
        yaml_path   (str): Path to the YAML file (e.g., 'train.yaml')
        prefix      (str): Root dataset directory (e.g., '/work/projects/ai_imaging_class/dataset')
        batch_size  (int): Batch size for DataLoader
        shuffle    (bool): Shuffle the dataset
        num_workers (int): Number of workers for data loading

    Returns:
        DataLoader: A MONAI DataLoader ready to use in your training/validation loop.
    """

    # Remove .yaml from the yaml_path
    dataset_type = yaml_path.replace('.yaml', '')

    # -----------------------
    # 1. Load patient folders from YAML
    # -----------------------
    with open(yaml_path, 'r') as f:
        patient_folders = yaml.safe_load(f)[dataset_type]

    # -----------------------
    # 2. Construct data dictionaries
    # -----------------------
    data_dicts = []
    for folder_name in patient_folders:
        # For T1 images + segmentation
        image_path = os.path.join(prefix, folder_name, f"{folder_name}_t1.nii.gz")
        label_path = os.path.join(prefix, folder_name, f"{folder_name}_seg.nii.gz")

        if os.path.exists(image_path) and os.path.exists(label_path):
            data_dicts.append({
                "image": image_path,
                "label": label_path
            })
        else:
            print(f"Warning: Missing T1 or seg file for {folder_name}")

    # -----------------------
    # 3. Define basic transforms
    #    (Includes EnsureDivisibled to avoid off-by-one dimension mismatch)
    # -----------------------
    base_transforms = monai.transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=2000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=[256, 256, 256],  # Larger size that's divisible by 16
            mode="constant"
        ),
        CenterSpatialCropd(
            keys=["image", "label"],
            roi_size=[240, 240, 240]  # Size divisible by 16
        ),
        ToTensord(keys=["image", "label"])
    ])

    # -----------------------
    # 4. Create Dataset and DataLoader
    # -----------------------
    dataset = Dataset(data=data_dicts, transform=base_transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader

