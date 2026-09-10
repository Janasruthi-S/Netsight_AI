import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class LSTMBridge:
    """
    Loads the trained LSTM encoder from the existing IDS checkpoint
    and converts preprocessed traffic sequences into latent sequences.

    This module does NOT perform preprocessing.
    Preprocessing remains owned by the LSTM pipeline.
    """

    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found:\n{self.checkpoint_path}"
            )

        self.device = torch.device(
            device if device else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print("=" * 70)
        print("LSTM BRIDGE")
        print("=" * 70)

        print(f"Checkpoint : {self.checkpoint_path}")
        print(f"Device     : {self.device}")

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if not isinstance(self.checkpoint, dict):
            raise ValueError("Checkpoint must contain a dictionary.")

        self._load_metadata()
        self._build_encoder()
        self._load_lstm_weights()

        self.model.eval()
        self.model.to(self.device)

        print("LSTM encoder loaded successfully.")
        print(f"Input size : {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"LSTM layers: {self.lstm_layers}")
        print(f"Classes    : {self.num_classes}")
        print("=" * 70)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def _load_metadata(self):

        required_fields = [
            "input_size",
            "hidden_size",
            "lstm_layers"
        ]

        missing = [
            field for field in required_fields
            if field not in self.checkpoint
        ]

        if missing:
            raise ValueError(
                f"Missing checkpoint metadata: {missing}"
            )

        self.input_size = int(self.checkpoint["input_size"])
        self.hidden_size = int(self.checkpoint["hidden_size"])
        self.lstm_layers = int(self.checkpoint["lstm_layers"])

        self.sequence_length = int(
            self.checkpoint.get("sequence_length", 0)
        )

        self.classes = self.checkpoint.get("classes", [])

        self.num_classes = int(
            self.checkpoint.get(
                "num_classes",
                len(self.classes)
            )
        )

        self.features = self.checkpoint.get("features", [])

    # ---------------------------------------------------------
    # BUILD LSTM
    # ---------------------------------------------------------

    def _build_encoder(self):

        self.model = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True
        )

    # ---------------------------------------------------------
    # AUTOMATICALLY FIND LSTM PARAMETERS
    # ---------------------------------------------------------

    def _find_lstm_prefix(self, state_dict):

        possible_prefixes = set()

        for key in state_dict.keys():

            if key.endswith("weight_ih_l0"):

                prefix = key[:-len("weight_ih_l0")]

                possible_prefixes.add(prefix)

        for prefix in possible_prefixes:

            required_key = prefix + "weight_hh_l0"

            if required_key in state_dict:
                return prefix

        raise ValueError(
            "Could not automatically locate LSTM parameters "
            "inside the checkpoint."
        )

    # ---------------------------------------------------------
    # LOAD LSTM WEIGHTS
    # ---------------------------------------------------------

    def _load_lstm_weights(self):

        state_dict = self.checkpoint.get("model_state_dict")

        if state_dict is None:
            raise ValueError(
                "Checkpoint does not contain 'model_state_dict'."
            )

        prefix = self._find_lstm_prefix(state_dict)

        print(f"Detected LSTM parameter prefix: '{prefix}'")

        lstm_state = {}

        for key, value in state_dict.items():

            if key.startswith(prefix):

                new_key = key[len(prefix):]

                lstm_state[new_key] = value

        missing, unexpected = self.model.load_state_dict(
            lstm_state,
            strict=False
        )

        if missing:
            raise RuntimeError(
                f"Missing LSTM parameters: {missing}"
            )

        if unexpected:
            print(
                f"Warning: unexpected parameters: {unexpected}"
            )

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    @torch.no_grad()
    def encode(self, sequences, batch_size=512):

        """
        Convert preprocessed sequences into LSTM latent sequences.

        Input:
            [N, T, F]

        Output:
            [N, T, D]

        N = number of samples
        T = temporal sequence length
        F = input features
        D = LSTM hidden dimension
        """

        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(
                sequences.astype(np.float32)
            )

        if not isinstance(sequences, torch.Tensor):
            raise TypeError(
                "sequences must be a NumPy array or PyTorch tensor."
            )

        if sequences.ndim != 3:
            raise ValueError(
                f"Expected [N,T,F], got {tuple(sequences.shape)}"
            )

        if sequences.shape[-1] != self.input_size:

            raise ValueError(
                "Feature dimension mismatch.\n"
                f"LSTM expects : {self.input_size}\n"
                f"Received      : {sequences.shape[-1]}"
            )

        outputs = []

        for start in range(0, len(sequences), batch_size):

            batch = sequences[
                start:start + batch_size
            ].to(self.device)

            latent, _ = self.model(batch)

            outputs.append(
                latent.cpu()
            )

        return torch.cat(outputs, dim=0)

    # ---------------------------------------------------------
    # FINAL LATENT STATE
    # ---------------------------------------------------------

    @torch.no_grad()
    def encode_final_state(
        self,
        sequences,
        batch_size=512
    ):

        """
        Returns only the final latent state.

        Input:
            [N,T,F]

        Output:
            [N,D]
        """

        latent_sequence = self.encode(
            sequences,
            batch_size=batch_size
        )

        return latent_sequence[:, -1, :]

    # ---------------------------------------------------------
    # SAVE LATENT DATA
    # ---------------------------------------------------------

    def save_latent_sequences(
        self,
        sequences,
        output_path,
        batch_size=512
    ):

        latent = self.encode(
            sequences,
            batch_size=batch_size
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_path,
            latent.numpy()
        )

        metadata_path = output_path.with_suffix(
            ".json"
        )

        metadata = {
            "source_checkpoint": str(
                self.checkpoint_path
            ),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "lstm_layers": self.lstm_layers,
            "sequence_length": self.sequence_length,
            "latent_dimension": self.hidden_size,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "features": self.features
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2
            )

        print(
            f"Latent sequences saved to:\n{output_path}"
        )

        print(
            f"Metadata saved to:\n{metadata_path}"
        )

        return latent