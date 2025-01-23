import torch
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete



def main(test_loader):
    # Define the U-Net model with same architecture as training
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to("cuda")

    # Load the trained model weights
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Define metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    post_trans = AsDiscrete(threshold=0.5)

    # Test loop
    test_dice_scores = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            inputs, labels = batch_data["image"].to("cuda"), batch_data["label"].to("cuda")
            outputs = model(inputs)
            
            # Apply post-processing
            outputs = post_trans(outputs)
            
            # Calculate Dice score
            dice_metric(y_pred=outputs, y=labels)
            
            # Optionally save predictions
            # You can add code here to save the predicted segmentations

        # Calculate mean Dice score
        mean_dice = dice_metric.aggregate()[0].item()
        dice_metric.reset()
        
        print(f"Test Dataset Dice Score: {mean_dice:.4f}")

    return mean_dice


