"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.models import MLPPlanner
from homework.metrics import compute_lateral_error, compute_longitudinal_error
from homework.datasets.road_dataset import RoadDataset
from homework.datasets.road_transforms import EgoTrackProcessor
from homework.supertux_utils import save_model  # update path if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=2,
    lr=1e-3,
    batch_size=64,
    num_epoch=20,
):
    print(f"\n🛠️ Starting training: model={model_name}, lr={lr}, batch_size={batch_size}, epochs={num_epoch}")

    # === Data ===
    transform = EgoTrackProcessor()  # update if you add more transforms later
    train_ds = RoadDataset(split="train", transform=transform)
    val_ds = RoadDataset(split="val", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    # === Model ===
    model = MLPPlanner().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in train_loader:
            track_left = batch["track_left"].to(DEVICE)
            track_right = batch["track_right"].to(DEVICE)
            waypoints = batch["waypoints"].to(DEVICE)
            mask = batch["waypoints_mask"].to(DEVICE).unsqueeze(-1)  # (B, 3, 1)

            optimizer.zero_grad()
            preds = model(track_left=track_left, track_right=track_right)
            loss = loss_fn(preds[mask], waypoints[mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # === Validation ===
        model.eval()
        val_loss = 0
        total_lat_err = 0
        total_long_err = 0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(DEVICE)
                track_right = batch["track_right"].to(DEVICE)
                waypoints = batch["waypoints"].to(DEVICE)
                mask = batch["waypoints_mask"].to(DEVICE).unsqueeze(-1)

                preds = model(track_left=track_left, track_right=track_right)
                loss = loss_fn(preds[mask], waypoints[mask])
                val_loss += loss.item()

                total_lat_err += compute_lateral_error(preds, waypoints, mask).item()
                total_long_err += compute_longitudinal_error(preds, waypoints, mask).item()
                count += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / count
        avg_lat_err = total_lat_err / count
        avg_long_err = total_long_err / count

        print(
            f"📉 Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Lateral Err: {avg_lat_err:.4f} | Longitudinal Err: {avg_long_err:.4f} | "
            f"LR: {lr:.1e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, model_name)
            print(f"💾 Saved new best model to '{model_name}.pt'")

if __name__ == "__main__":
    train()
