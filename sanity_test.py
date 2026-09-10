import json
import torch
from forecasting_model import ForecastingModel, load_config

# Load metadata
with open(r"data\chronological\chronological_metadata.json", "r") as f:
    metadata = json.load(f)

latent_dim = metadata["lstm"]["latent_dimension"]
sequence_length = metadata["forecasting"]["history_length"]
horizon = metadata["forecasting"]["forecast_horizon"]
num_classes = metadata["dataset"]["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("NETSIGHT FORECASTING SANITY TEST")
print("=" * 60)
print(f"Device            : {device}")
print(f"Latent dimension  : {latent_dim}")
print(f"Sequence length   : {sequence_length}")
print(f"Forecast horizon  : {horizon}")
print(f"Classes            : {num_classes}")

# Load forecasting configuration
config = load_config("config.yaml")

# Small artificial batch
batch_size = 2

x = torch.randn(
    batch_size,
    sequence_length,
    latent_dim,
    device=device
)

# Correct constructor
model = ForecastingModel(
    latent_dim,
    num_classes,
    config
).to(device)

model.eval()

with torch.no_grad():
    output = model(
        x,
        horizon=horizon
    )

print()
print("FORWARD PASS: PASSED")
print(f"Output type       : {type(output)}")

if isinstance(output, tuple):
    print(f"Output elements   : {len(output)}")

    for i, item in enumerate(output):
        if torch.is_tensor(item):
            print(
                f"Output[{i}] shape : {tuple(item.shape)}"
            )

print("=" * 60)
print("SANITY TEST PASSED")
print("=" * 60)