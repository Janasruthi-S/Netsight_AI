"""
World Model for predictive network security.

The World Model learns how the latent network state changes over time.

Input:
    Current LSTM latent state sequence

Output:
    Predicted future LSTM latent state

Concept:

    z_t  →  World Model  →  z_(t+1)

The model is intentionally separated from the LSTM.
The LSTM creates the representation of the current network state.
The World Model learns the transition dynamics of that state.
"""

import torch
import torch.nn as nn


class WorldModel(nn.Module):

    def __init__(
        self,
        latent_size=256,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ----------------------------------------------------
        # State transition network
        # ----------------------------------------------------
        #
        # Input:
        #     latent sequence z_1 ... z_T
        #
        # Output:
        #     transition representation
        #
        self.transition_lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ----------------------------------------------------
        # Future latent-state predictor
        # ----------------------------------------------------

        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, latent_size),
        )

        # ----------------------------------------------------
        # Delta predictor
        # ----------------------------------------------------
        #
        # Learns how much the network state changes.
        #
        self.delta_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

        # ----------------------------------------------------
        # Risk prediction head
        # ----------------------------------------------------
        #
        # Produces a continuous risk value in [0,1].
        #
        self.risk_head = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent_sequence):

        # latent_sequence:
        #
        # [batch, sequence_length, latent_size]

        transition_sequence, (hidden_state, cell_state) = (
            self.transition_lstm(latent_sequence)
        )

        # Last temporal representation
        last_state = transition_sequence[:, -1, :]

        # Predict next latent state
        predicted_next_state = self.state_predictor(
            last_state
        )

        # Predict state change
        predicted_delta = self.delta_predictor(
            last_state
        )

        # Risk associated with predicted state
        predicted_risk = self.risk_head(
            predicted_next_state
        )

        return (
            predicted_next_state,
            predicted_delta,
            predicted_risk,
            hidden_state,
            cell_state,
        )


def world_model_loss(
    predicted_state,
    target_state,
    predicted_delta,
    actual_delta,
    latent_weight=1.0,
    delta_weight=0.25,
):
    """
    World Model training loss.

    Two objectives:

    1. Predict the next latent state.
    2. Predict how much the network state changes.
    """

    state_loss = nn.functional.mse_loss(
        predicted_state,
        target_state
    )

    delta_loss = nn.functional.mse_loss(
        predicted_delta,
        actual_delta
    )

    total_loss = (
        latent_weight * state_loss
        + delta_weight * delta_loss
    )

    return total_loss, state_loss, delta_loss