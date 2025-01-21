import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from dataloader import get_dataloader
import monai


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance"
    ).to(device)


    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metric = DiceMetric(include_background=False, reduction="mean")

    train_loader = get_dataloader("splits/train.yaml", "/work/projects/ai_imaging_class/dataset")
    val_loader = get_dataloader("splits/val.yaml", "/work/projects/ai_imaging_class/dataset")

    best_metric = -1

    for epoch in range(50):
        print(f"\n=== Epoch {epoch + 1} ===")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            try:
                outputs = model(inputs)
            except Exception as e:
                print("Error during forward pass")
                print(f"Input shape: {inputs.shape}")
                print(f"Model configuration: {model}")
                raise e

            try:
                loss = loss_fn(outputs, labels)
                print(f"Loss: {loss.item()}")
            except Exception as e:
                print("Error during loss computation")
                print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                raise e

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_score = 0.0

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                print(f"Validation Batch {val_batch_idx + 1}: Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

                outputs = model(inputs)
                print(f"Validation output shape: {outputs.shape}")

                val_score += metric(y_pred=outputs, y=labels).item()

            val_score /= len(val_loader)
            print(f"Validation Score: {val_score}")

        if val_score > best_metric:
            print("New best model found!")
            torch.save(model.state_dict(), "best_model.pth")
            best_metric = val_score

if __name__ == "__main__":
    train()
