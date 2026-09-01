"""
Network attack forecasting using:

    LSTM Encoder
          ↓
    Current latent state
          ↓
      World Model
          ↓
    Future latent state
          ↓
    Attack classifier
          ↓
    Future attack probability

Run from the project root:

    python forecasting/forecast.py

Or:

    python forecasting/forecast.py --sample 0 --steps 3
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import joblib

# ------------------------------------------------------------
# Parent project directory
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


import config

from model import LSTMEncoder

from world_model import WorldModel

import config_forecasting


# ============================================================
# LOAD LSTM
# ============================================================

def load_lstm(input_size):

    model = LSTMEncoder(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    ).to(config_forecasting.DEVICE)

    checkpoint = torch.load(
        config_forecasting.LSTM_CHECKPOINT_PATH,
        map_location=config_forecasting.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# LOAD WORLD MODEL
# ============================================================

def load_world_model():

    model = WorldModel(
        latent_size=config_forecasting.LATENT_SIZE,
        hidden_size=config_forecasting.WORLD_MODEL_HIDDEN_SIZE,
        num_layers=config_forecasting.WORLD_MODEL_LAYERS,
        dropout=config_forecasting.WORLD_MODEL_DROPOUT,
    ).to(config_forecasting.DEVICE)

    checkpoint = torch.load(
        config_forecasting.WORLD_MODEL_CHECKPOINT_PATH,
        map_location=config_forecasting.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

def load_label_encoder():

    return joblib.load(
        config_forecasting.LABEL_ENCODER_PATH
    )


# ============================================================
# GENERATE CURRENT LATENT STATE
# ============================================================

def get_current_latent(
    lstm,
    sequence
):
    """
    Converts the current traffic sequence into
    an LSTM latent representation.
    """

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    x = x.to(
        config_forecasting.DEVICE
    )

    with torch.no_grad():

        latent_sequence, class_logits, _, _ = (
            lstm(x)
        )

    return (
        latent_sequence,
        class_logits
    )


# ============================================================
# CLASSIFICATION FROM LATENT STATE
# ============================================================

def classify_latent(
    lstm,
    latent_state,
    label_encoder
):
    """
    Uses the LSTM classification head on a latent state.

    The classifier is:

        latent state
              ↓
        Linear → ReLU
              ↓
        Linear
              ↓
        attack class
    """

    with torch.no_grad():

        logits = lstm.classifier(
            latent_state
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        class_index = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        class_probability = float(
            probabilities[0, class_index].item()
        )

    try:

        class_name = label_encoder.inverse_transform(
            [class_index]
        )[0]

    except Exception:

        class_name = str(class_index)

    return (
        class_name,
        class_probability,
        probabilities
    )


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(score):

    if score < config_forecasting.LOW_RISK_THRESHOLD:
        return "LOW"

    if score < config_forecasting.MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    if score < config_forecasting.HIGH_RISK_THRESHOLD:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# FORECAST
# ============================================================

def generate_forecast(
    sequence,
    lstm,
    world_model,
    label_encoder,
    steps
):
    """
    Generate future network-state predictions.

    The important part:

        z_t
         ↓
       World Model
         ↓
        z_t+1
         ↓
       World Model
         ↓
        z_t+2
         ↓
       World Model
         ↓
        z_t+3

    This is called latent-state rollout.
    """

    # --------------------------------------------------------
    # Get current LSTM latent sequence
    # --------------------------------------------------------

    latent_sequence, current_logits = (
        get_current_latent(
            lstm,
            sequence
        )
    )

    # Current latent sequence:
    #
    # [1, SEQ_LEN, LATENT_SIZE]

    rolling_latent_sequence = (
        latent_sequence.clone()
    )

    results = []

    # --------------------------------------------------------
    # Forecast current state first
    # --------------------------------------------------------

    current_state = (
        rolling_latent_sequence[:, -1, :]
    )

    (
        current_class,
        current_class_probability,
        current_probabilities,
    ) = classify_latent(
        lstm,
        current_state,
        label_encoder
    )

    print()
    print("=" * 70)
    print("CURRENT NETWORK STATE")
    print("=" * 70)

    print(
        f"Current predicted class : {current_class}"
    )

    print(
        f"Current class confidence: "
        f"{current_class_probability:.4f}"
    )

    # --------------------------------------------------------
    # Future rollout
    # --------------------------------------------------------

    for step in range(1, steps + 1):

        with torch.no_grad():

            (
                predicted_state,
                predicted_delta,
                predicted_risk,
                _,
                _,
            ) = world_model(
                rolling_latent_sequence
            )

        # ----------------------------------------------------
        # Classify predicted future latent state
        # ----------------------------------------------------

        (
            predicted_class,
            class_probability,
            probabilities,
        ) = classify_latent(
            lstm,
            predicted_state,
            label_encoder
        )

        risk = float(
            predicted_risk[0, 0].item()
        )

        level = risk_level(
            risk
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {
            "future_step": step,
            "predicted_attack_class":
                predicted_class,
            "attack_probability":
                class_probability,
            "world_model_risk":
                risk,
            "risk_level":
                level,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()
        print(
            f"Future Step +{step}"
        )

        print(
            f"  Predicted attack : "
            f"{predicted_class}"
        )

        print(
            f"  Attack probability: "
            f"{class_probability:.4f}"
        )

        print(
            f"  World-model risk : "
            f"{risk:.4f}"
        )

        print(
            f"  Risk level       : "
            f"{level}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Feed predicted state back into the World Model.
        #
        # This creates:
        #
        # z_t → z_t+1 → z_t+2 → z_t+3
        #
        # rather than predicting every future step
        # independently.
        # ----------------------------------------------------

        rolling_latent_sequence = torch.cat(
            [
                rolling_latent_sequence[:, 1:, :],
                predicted_state.unsqueeze(1),
            ],
            dim=1
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "World Model based network attack forecasting"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Validation sample index"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=config_forecasting.FORECAST_STEPS,
        help="Number of future states to forecast"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load feature columns
    # --------------------------------------------------------

    with open(
        config_forecasting.FEATURE_COLUMNS_PATH
    ) as f:

        feature_columns = json.load(f)

    input_size = len(
        feature_columns
    )

    print(
        f"Number of features: {input_size}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print(
        "Loading trained LSTM..."
    )

    lstm = load_lstm(
        input_size
    )

    print(
        "Loading trained World Model..."
    )

    world_model = load_world_model()

    label_encoder = load_label_encoder()

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    X_val = np.load(
        config_forecasting.X_VAL_PATH
    )

    if args.sample < 0 or args.sample >= len(X_val):

        raise ValueError(
            f"Sample index must be between "
            f"0 and {len(X_val) - 1}"
        )

    sequence = X_val[
        args.sample
    ]

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    results = generate_forecast(
        sequence,
        lstm,
        world_model,
        label_encoder,
        args.steps
    )

    # --------------------------------------------------------
    # Save forecast
    # --------------------------------------------------------

    output_path = os.path.join(
        config_forecasting.FORECASTING_CHECKPOINT_DIR,
        "latest_forecast.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print(
        f"Forecast saved to: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()