import json
import numpy as np
import torch
import yaml

from forecasting_model import (
    ForecastPredictor,
    RLAdapter,
    load_forecasting_model
)


# ============================================================
# CONFIG
# ============================================================

with open(
    "config.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)


# ============================================================
# METADATA
# ============================================================

with open(
    "checkpoints/forecasting_metadata.json",
    "r"
) as f:

    metadata = json.load(f)


class_names = metadata[
    "classes"
]

horizon = metadata[
    "forecast_horizon"
]


# ============================================================
# LOAD MODEL
# ============================================================

model = load_forecasting_model(
    "checkpoints/best_forecasting_model.pt",
    config,
    class_names
)


predictor = ForecastPredictor(
    model,
    class_names
)


rl_adapter = RLAdapter(
    class_names
)


# ============================================================
# LOAD LSTM LATENT OUTPUT
# ============================================================

data = np.load(
    "live_latent_input.npy"
)


# Expected:
#
# [sequence_length, latent_dimension]
#
# OR
#
# [batch, sequence_length, latent_dimension]


# ============================================================
# FORECAST
# ============================================================

forecast = predictor.predict(
    latent_sequence=data,
    horizon=horizon
)


# First sample
prediction = forecast[0]


# ============================================================
# RL STATE
# ============================================================

rl_output = rl_adapter.convert(
    prediction
)


# ============================================================
# OUTPUT
# ============================================================

print(
    json.dumps(
        rl_output,
        indent=4
    )
)