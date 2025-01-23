import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ToTensord
from dataloader import get_dataloader
from monai.networks.nets import UNet


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance"
    ).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Prepare the test dataloader
    test_loader = get_dataloader("splits/test.yaml", "/work/projects/ai_imaging_class/dataset")

    # Initialize metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Evaluation loop
    with torch.no_grad():
        test_score = 0.0
        for batch_idx, batch in enumerate(test_loader):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            outputs = model(inputs)

            # Calculate Dice score for this batch
            batch_scores = dice_metric(y_pred=outputs, y=labels)  # Get Dice scores per class
            batch_score = batch_scores.mean().item()  # Compute mean Dice score
            print(f"Batch {batch_idx + 1} Dice Score: {batch_score}")
            test_score += batch_score

        # Average test score
        test_score /= len(test_loader)
        print(f"Overall Test Dice Score: {test_score}")

if __name__ == "__main__":
    test()
