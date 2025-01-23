import test
import util.data_processing as dp
import util.patiant_yaml_spliter as pys
import train
import test

"""
    Path to the BraTS 2021 dataset

    Data Structure:
        dataset
            - BraTS2021_00000
                - BraTS2021_00000_flair.nii.gz  // MRI FLAIR
                - BraTS2021_00000_t1.nii.gz     // MRI T1
                - BraTS2021_00000_t1ce.nii.gz   // MRI T1 with contrast enhancement
                - BraTS2021_00000_t2.nii.gz     // MRI T2
                - BraTS2021_00000_seg.nii.gz    // Segmentation
            - BraTS2021_XXXXX
                - BraTS2021_XXXXX_flair.nii.gz
                - BraTS2021_XXXXX_t1.nii.gz
                - BraTS2021_XXXXX_t1ce.nii.gz
                - BraTS2021_XXXXX_t2.nii.gz
                - BraTS2021_XXXXX_seg.nii.gz

"""
DATA_SET_PATH = "/work/projects/ai_imaging_class/dataset"
DATA_SET_PATH_PREFIX = "/work/projects/ai_imaging_class/dataset"


def print_hi(name):
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    # Check if YAML splits exist
    if not pys.check_yaml_files():
        pys.create_yaml_file(DATA_SET_PATH, 70, 15, 15)

    # Create DataLoaders
    train_loader = dp.get_monai_dataloader(
        yaml_path="train.yaml",
        prefix=DATA_SET_PATH_PREFIX,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    val_loader = dp.get_monai_dataloader(
        yaml_path="val.yaml",
        prefix=DATA_SET_PATH_PREFIX,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = dp.get_monai_dataloader(
        yaml_path="test.yaml",
        prefix=DATA_SET_PATH_PREFIX,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    for batch in train_loader:
        print(batch["image"].shape)  # [batch_size, channels, depth, height, width]
        print(batch["label"].shape)
        break

    # Step 2: Train the model
    print("Starting training...")
    train.main(train_loader, val_loader)
    
    # Step 3: Test the model
    print("Starting testing...")
    test.main(test_loader)
