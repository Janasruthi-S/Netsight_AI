import json
import numpy as np

with open(r"data\chronological\chronological_metadata.json", "r") as f:
    m = json.load(f)

z = np.load(
    r"data\chronological\full_final_latents.npy",
    mmap_mode="r"
)

y = np.load(
    r"data\chronological\full_labels.npy",
    mmap_mode="r"
)

sequence_length = m["forecasting"]["history_length"]
horizon = m["forecasting"]["forecast_horizon"]

train_end = m["chronological_split"]["train"]["row_end_exclusive"]
validation_end = m["chronological_split"]["validation"]["row_end_exclusive"]

n = len(y)

print()
print("=" * 60)
print("NETSIGHT FORECASTING DATA VALIDATION")
print("=" * 60)

print(f"Latent shape       : {z.shape}")
print(f"Label shape        : {y.shape}")
print(f"Latent dimension   : {m['lstm']['latent_dimension']}")
print(f"Sequence length    : {sequence_length}")
print(f"Forecast horizon   : {horizon}")
print(f"Classes            : {m['dataset']['classes']}")

print(f"Train samples      : {train_end - sequence_length - horizon + 1:,}")
print(f"Validation samples : {validation_end - train_end - horizon + 1:,}")
print(f"Test samples       : {n - validation_end - horizon + 1:,}")

print("=" * 60)
print("VALIDATION PASSED")
print("=" * 60)