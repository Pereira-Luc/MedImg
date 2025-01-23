import torch
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from dataloader import get_dataloader
import monai
from tqdm import tqdm

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())


def main(train_loader, val_loader):
    # Define the U-Net model
    model = UNet(
        spatial_dims=3,
        in_channels=1,  # Number of MRI modalities
        out_channels=1,  # Single channel for segmentation
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to("cuda")

 
    # Define optimizer, loss, and metrics
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    post_trans = AsDiscrete(threshold=0.5)

    # Early stopping parameters
    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    max_epochs = 50
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_loss = 0

        # Training loop
        for batch_data in tqdm(train_loader, desc="Training"):
            inputs, labels = batch_data["image"].to("cuda"), batch_data["label"].to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                inputs, labels = batch_data["image"].to("cuda"), batch_data["label"].to("cuda")
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                # Calculate Dice score
                outputs = post_trans(outputs)
                dice_metric(y_pred=outputs, y=labels)

        val_loss /= len(val_loader)
        dice_score = dice_metric.aggregate()[0].item()
        dice_metric.reset()

        print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")
        print(f"Epoch {epoch + 1} Dice Score: {dice_score}")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
