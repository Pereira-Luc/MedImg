import os
import random
import yaml

def create_split(data_dir, train_pct=70, val_pct=15, test_pct=15):
    all_patients = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    random.shuffle(all_patients)

    total = len(all_patients)
    train_split = int(train_pct / 100 * total)
    val_split = int(val_pct / 100 * total)

    splits = {
        'train': all_patients[:train_split],
        'val': all_patients[train_split:train_split + val_split],
        'test': all_patients[train_split + val_split:]
    }

    os.makedirs("splits", exist_ok=True)
    for split, data in splits.items():
        with open(f"splits/{split}.yaml", 'w') as file:
            yaml.dump({split: data}, file)

if __name__ == "__main__":
    create_split("/work/projects/ai_imaging_class/dataset")

