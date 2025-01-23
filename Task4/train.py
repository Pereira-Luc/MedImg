import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from dataloader import get_dataloader
import monai

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())



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
    metric = DiceMetric(include_background=False, get_not_nans=True, reduction="mean")

    train_loader = get_dataloader("splits/train.yaml", "/work/projects/ai_imaging_class/dataset", batch_size=1)
    val_loader = get_dataloader("splits/val.yaml", "/work/projects/ai_imaging_class/dataset", batch_size=1)

    best_metric = -1
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    for epoch in range(50):
        print(f"\n=== Epoch {epoch + 1} ===")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch["image"], batch["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            # One-hot encode labels
            post_label = AsDiscrete(to_onehot=4)
            labels = post_label(labels)
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

        post_pred = AsDiscrete(argmax=True)
        
        post_label = AsDiscrete(to_onehot=4)
        
        
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                # Forward pass
                outputs = model(inputs)
                # One-hot encode labels
                post_label = AsDiscrete(to_onehot=4)
                labels = post_label(labels)  # One-hot encode -> [num_classes, batch_size, D, H, W]
                # Post-process predictions
                post_pred = AsDiscrete(argmax=True)
                outputs = post_pred(outputs)
                # Verify shapes
                assert outputs.shape == labels.shape, f"Shape mismatch: Outputs {outputs.shape}, Labels {labels.shape}"
                # Compute metric and reduce to scalar
                metric_output = metric(y_pred=outputs, y=labels)
                # Debug: Check raw metric output
                print(f"  Metric output (raw): {metric_output}")
                metric_output = metric_output[~torch.isnan(metric_output)]  # Remove NaN values
                if metric_output.numel() > 0:  # If there are valid scores
                    val_score += metric_output.mean().item()
                else:
                    print("Skipping metric update due to all NaN scores.")
            if val_score > 0:
                    val_score /= len(val_loader)  # Compute average if there are valid scores
            print(f"Validation Score: {val_score}")
        
        

    
        if val_score > best_metric:
            print("New best model found!")
            torch.save(model.state_dict(), "best_model.pth")
            best_metric = val_score
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
    
        # Save intermediate model after every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
            print(f"Saved intermediate model at epoch {epoch + 1}.")
    
        if patience_counter >= patience:
            print("Early stopping triggered. No improvement in validation score.")
            break
    

if __name__ == "__main__":
    train()
