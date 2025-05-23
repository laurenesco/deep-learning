from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.mlp = nn.Sequential(
            nn.Flatten(),                   # (B, 10, 2) + (B, 10, 2) → (B, 40)
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_waypoints * 2), # Output (B, 6) → reshaped to (B, 3, 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        x = torch.cat([track_left, track_right], dim=1)  # (B, 10, 2) → (B, 20, 2)
        out = self.mlp(x)                                # → (B, 6)
        return out.view(-1, self.n_waypoints, 2)         # → (B, 3, 2)

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,              # <-- default = 2
        dim_feedforward: int = 128,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.input_proj = nn.Linear(2, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)  # <-- 2 layers here

        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        B = track_left.size(0)
        src = torch.cat([track_left, track_right], dim=1)  # (B, 2 * n_track, 2)
        src = self.input_proj(src)                         # (B, 2 * n_track, d_model)
        memory = self.encoder(src)                         # (B, 2 * n_track, d_model)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, n_waypoints, d_model)
        tgt = self.decoder(queries, memory)                            # (B, n_waypoints, d_model)
        return self.output_proj(tgt)                                   # (B, n_waypoints, 2)


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_waypoints * 2),
        )

#        self.cnn = nn.Sequential(
#            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
#            nn.BatchNorm2d(16),
#            nn.ReLU(),
#            # extra bottleneck block
#            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            # expanded deeper layer
#            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
#            nn.Flatten(),                 # (B, 128)
#        )

#        self.mlp = nn.Sequential(
#            nn.Linear(128, 128),
#            nn.ReLU(),
#            nn.Dropout(p=0.5),
#            nn.Linear(128, 64),
#            nn.ReLU(),
#            nn.Dropout(p=0.3),
#            nn.Linear(64, n_waypoints * 2),
#        )

        # ── weight initialization ─────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.cnn(x)                  # (B, 64)
        out = self.mlp(x)               # (B, 6)
        return out.view(-1, self.n_waypoints, 2)  # (B, 3, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024

