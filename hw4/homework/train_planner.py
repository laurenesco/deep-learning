"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from homework.models import MODEL_FACTORY, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, dataloader, optimizer, loss_fn, train=True):
    metric = PlannerMetric()
    total_loss = 0.0
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            if train:
                optimizer.zero_grad()

            if "image" in inputs:
                preds = model(image=inputs["image"])
            else:
                preds = model(track_left=inputs["track_left"], track_right=inputs["track_right"])

            loss = loss_fn(preds[inputs["waypoints_mask"]], inputs["waypoints"][inputs["waypoints_mask"]])

            if train:
                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            metric.add(preds, inputs["waypoints"], inputs["waypoints_mask"])

    return total_loss / len(dataloader), metric.compute()

def train_transformer(model, train_loader, val_loader, num_epoch, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        train_loss, train_stats = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        val_loss, val_stats = run_epoch(model, val_loader, optimizer, loss_fn, train=False)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Lateral: {train_stats['lateral_error']:.4f} | Train Longitudinal: {train_stats['longitudinal_error']:.4f} | "
            f"Val Lateral: {val_stats['lateral_error']:.4f} | Val Longitudinal: {val_stats['longitudinal_error']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

        combined_error = val_stats["lateral_error"] + val_stats["longitudinal_error"]
        if combined_error < best_val_loss:
            best_val_loss = combined_error
            save_model(model)
            print(f"Saved new best model with combined error {combined_error:.4f} "
                  f"(Lateral: {val_stats['lateral_error']:.4f}, "
                  f"Longitudinal: {val_stats['longitudinal_error']:.4f})")

def train_cnn(model, train_loader, val_loader, num_epoch, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        train_loss, train_stats = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        val_loss, val_stats = run_epoch(model, val_loader, optimizer, loss_fn, train=False)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Lateral: {train_stats['lateral_error']:.4f} | Train Longitudinal: {train_stats['longitudinal_error']:.4f} | "
            f"Val Lateral: {val_stats['lateral_error']:.4f} | Val Longitudinal: {val_stats['longitudinal_error']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

        combined_error = val_stats["lateral_error"] + val_stats["longitudinal_error"]
        if combined_error < best_val_loss:
            best_val_loss = combined_error
            save_model(model)
            print(f"Saved new best model with combined error {combined_error:.4f} "
                  f"(Lateral: {val_stats['lateral_error']:.4f}, "
                  f"Longitudinal: {val_stats['longitudinal_error']:.4f})")

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

    model_class = MODEL_FACTORY[model_name]
    model = model_class(**model_kwargs).to(DEVICE)

    if model_name == "transformer_planner":
        train_transformer(model, train_loader, val_loader, num_epoch, lr)
    elif model_name == "cnn_planner":
        train_cnn(model, train_loader, val_loader, num_epoch, lr)
    else:
        # Fallback to default training logic for MLP
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_val_loss = float("inf")

        for epoch in range(num_epoch):
            model.train()
            total_loss = 0
            train_metric = PlannerMetric()

            for batch in train_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}

                optimizer.zero_grad()
                preds = model(track_left=inputs["track_left"], track_right=inputs["track_right"])
                loss = loss_fn(preds[inputs["waypoints_mask"]], inputs["waypoints"][inputs["waypoints_mask"]])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_metric.add(preds, inputs["waypoints"], inputs["waypoints_mask"])

            train_stats = train_metric.compute()

            model.eval()
            val_loss = 0
            val_metric = PlannerMetric()

            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
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

            combined_error = val_stats["lateral_error"] + val_stats["longitudinal_error"]
            if combined_error < best_val_loss:
                best_val_loss = combined_error
                save_model(model)
                print(f"Saved new best model with combined error {combined_error:.4f} "
                      f"(Lateral: {val_stats['lateral_error']:.4f}, "
                      f"Longitudinal: {val_stats['longitudinal_error']:.4f})")

if __name__ == "__main__":
    train()