import os
import random
import yaml


def get_all_Patients(data_dir):
    all_folders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    random.shuffle(all_folders)
    return all_folders


"""
    This function creates one yaml file for each train, val, and test set.
    
    Args:
        data_dir: Path to the data directory
        train_amount: The amount of data in %
        val_amount: The amount of data in %
        test_amount: The amount of data in %
        
    Returns:
        None
        
    structure of the yaml file:
        - train: 
            - BraTS2021_00000
            - BraTS2021_00001
            - ...
        - val:
            - BraTS2021_00002
            - BraTS2021_00003
            - ...
        - test:
            - BraTS2021_00004
            - BraTS2021_00005
            - ...
            
    Output:
        - train.yaml
        - val.yaml
        - test.yaml
"""


def create_yaml_file(data_dir, train_amount, val_amount, test_amount):
    all_folders = get_all_Patients(data_dir)

    print(f"Total subjects: {len(all_folders)}")

    # Calculate the amount of data for each set
    train_amount = int(len(all_folders) * train_amount / 100)
    val_amount = int(len(all_folders) * val_amount / 100)
    test_amount = int(len(all_folders) * test_amount / 100)

    # Create dictionaries for each set
    train_dict = {'train': all_folders[:train_amount]}
    val_dict = {'val': all_folders[train_amount:train_amount + val_amount]}
    test_dict = {'test': all_folders[train_amount + val_amount:]}


    # Write the dictionaries to yaml files
    with open('train.yaml', 'w') as file:
        documents = yaml.dump(train_dict, file)

    with open('val.yaml', 'w') as file:
        documents = yaml.dump(val_dict, file)

    with open('test.yaml', 'w') as file:
        documents = yaml.dump(test_dict, file)

    print(f"train.yaml created with {train_amount} subjects")
    print(f"val.yaml created with {val_amount} subjects")
    print(f"test.yaml created with {test_amount} subjects")


def check_yaml_files():
    if not os.path.exists('train.yaml'):
        print("train.yaml does not exist")
        return False
    if not os.path.exists('val.yaml'):
        print("val.yaml does not exist")
        return False
    if not os.path.exists('test.yaml'):
        print("test.yaml does not exist")
        return False
    return True
