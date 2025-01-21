from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    SpatialPadd, ScaleIntensityRanged, MapLabelValued, ToTensord, Compose
)

from monai.data import DataLoader, Dataset
import yaml
import os

def get_dataloader(yaml_path, data_dir, batch_size=2, shuffle=True, num_workers=1):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    key = os.path.basename(yaml_path).replace('.yaml', '')
    patient_ids = data[key]

    data_dicts = [
        {
            "image": os.path.join(data_dir, pid, f"{pid}_t1.nii.gz"),
            "label": os.path.join(data_dir, pid, f"{pid}_seg.nii.gz")
        }
        for pid in patient_ids
    ]

    
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        SpatialPadd(keys=["image", "label"], spatial_size=(240, 240, 160)),  # Pad depth to 160
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2000, b_min=0.0, b_max=1.0, clip=True),
        MapLabelValued(keys=["label"], orig_labels=[0, 1, 2, 3, 4], target_labels=[0, 1, 2, 3, 3]),
        ToTensord(keys=["image", "label"])
    ])
    

    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

