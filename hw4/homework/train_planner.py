"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
import torch.optim as optim

from homework.models import MODEL_FACTORY, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=2,
    lr=1e-3,
    batch_size=64,
    num_epoch=20,
    model_kwargs=None,
):
    if model_kwargs is None:
        model_kwargs = {}

    print(f"Starting training: model={model_name}, lr={lr}, batch_size={batch_size}, epochs={num_epoch}")

    # === Load data ===
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # === Model ===
    model_class = MODEL_FACTORY[model_name]
    model = model_class(**model_kwargs).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if sum(p.numel() for p in model.parameters()) == 0:
        raise ValueError(f"{model_name} has no trainable parameters")

    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        train_metric = PlannerMetric()

        for batch in train_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            if "image" in inputs:
                preds = model(image=inputs["image"])
            else:
                preds = model(track_left=inputs["track_left"], track_right=inputs["track_right"])

            loss = loss_fn(preds[inputs["waypoints_mask"]], inputs["waypoints"][inputs["waypoints_mask"]])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_metric.add(preds, inputs["waypoints"], inputs["waypoints_mask"])

        train_stats = train_metric.compute()

        # === Validation ===
        model.eval()
        val_loss = 0
        val_metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}

                if "image" in inputs:
                    preds = model(image=inputs["image"])
                else:
                    preds = model(track_left=inputs["track_left"], track_right=inputs["track_right"])

                loss = loss_fn(preds[inputs["waypoints_mask"]], inputs["waypoints"][inputs["waypoints_mask"]])
                val_loss += loss.item()

                val_metric.add(preds, inputs["waypoints"], inputs["waypoints_mask"])

        val_stats = val_metric.compute()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Lateral: {train_stats['lateral_error']:.4f} | Train Longitudinal: {train_stats['longitudinal_error']:.4f} | "
            f"Val Lateral: {val_stats['lateral_error']:.4f} | Val Longitudinal: {val_stats['longitudinal_error']:.4f} | "
            f"LR: {lr:.1e}"
        )

    # Track best combined error that meets grading threshold
    if (
        val_stats["lateral_error"] < 0.6
        and val_stats["longitudinal_error"] < 0.2
    ):
        combined_error = val_stats["lateral_error"] + val_stats["longitudinal_error"]

        if combined_error < best_val_loss:  # repurpose best_val_loss to track best error sum
            best_val_loss = combined_error
            save_model(model)
            print(f"Saved model with lateral {val_stats['lateral_error']:.4f} and longitudinal {val_stats['longitudinal_error']:.4f}")


if __name__ == "__main__":
    train()

