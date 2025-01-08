import util.data_processing as dp
import util.patiant_yaml_spliter as pys

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
DATA_SET_PATH_PREFIX = '/work/projects/ai_imaging_class/dataset'


def print_hi(name):
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    DATA_SET_PATH = "/work/projects/ai_imaging_class/dataset"

    # Check if the train.yaml, val.yaml, and test.yaml files exist
    # If they do not exist, run pys.create_yaml_file to create them
    if not pys.check_yaml_files():
        pys.create_yaml_file(DATA_SET_PATH, 70, 15, 15)


    train_loader = dp.get_monai_dataloader(
        yaml_path="train.yaml",
        prefix=DATA_SET_PATH_PREFIX,
        batch_size=2,
        shuffle=True,
        num_workers=4
    )

    print(f"Number of batches in train_loader: {len(train_loader)}")
