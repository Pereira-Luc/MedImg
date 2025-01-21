import torch
from monai.metrics import DiceMetric
from dataloader import get_dataloader
from monai.networks.nets import UNet

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3, in_channels=1, out_channels=4,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loader = get_dataloader("splits/test.yaml", "/path/to/dataset")
    metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        score = 0.0
        for batch in test_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            score += metric(y_pred=outputs, y=labels).item()
        print(f"Test Dice Score: {score / len(test_loader)}")

if __name__ == "__main__":
    test()

