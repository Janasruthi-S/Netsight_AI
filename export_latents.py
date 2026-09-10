import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "dataset",
    "combined_network_dataset.csv"
)

MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "models"
)

FORECASTING_DIR = os.path.join(
    PROJECT_DIR,
    "forecasting"
)

FORECASTING_DATA_DIR = os.path.join(
    FORECASTING_DIR,
    "data"
)

os.makedirs(
    FORECASTING_DATA_DIR,
    exist_ok=True
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "best_world_model_ids.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.json"
)

CONFIG_PATH = os.path.join(
    FORECASTING_DIR,
    "config.yaml"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

# This MUST match the seed used by train_model.py.
SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("NETSIGHT AI - LSTM LATENT STATE EXPORT")
print("=" * 80)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 80)


# ============================================================
# FILE VALIDATION
# ============================================================

required_files = {
    "Dataset": DATA_PATH,
    "LSTM checkpoint": CHECKPOINT_PATH,
    "Scaler": SCALER_PATH,
    "Label encoder": LABEL_ENCODER_PATH,
    "Feature columns": FEATURE_COLUMNS_PATH,
    "Forecasting config": CONFIG_PATH
}

for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


# ============================================================
# LOAD FORECASTING CONFIGURATION
# ============================================================

print("\nLoading forecasting configuration...")

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = yaml.safe_load(file) or {}


# ------------------------------------------------------------
# Temporal configuration
# ------------------------------------------------------------

temporal_config = config.get(
    "temporal",
    {}
)

forecast_horizon = temporal_config.get(
    "forecast_horizon"
)

if forecast_horizon is None:

    raise ValueError(
        "forecast_horizon is missing from "
        "forecasting/config.yaml"
    )

forecast_horizon = int(
    forecast_horizon
)


# ------------------------------------------------------------
# Sequence length
# ------------------------------------------------------------

sequence_length = (
    temporal_config.get("sequence_length")
)

if sequence_length is None:

    # Some projects may name it seq_len.
    sequence_length = (
        temporal_config.get("seq_len")
    )

if sequence_length is None:

    raise ValueError(
        "sequence_length is missing from "
        "forecasting/config.yaml.\n"
        "Add either:\n"
        "  sequence_length: <value>\n"
        "or:\n"
        "  seq_len: <value>"
    )

sequence_length = int(
    sequence_length
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

training_config = config.get(
    "training",
    {}
)

batch_size = int(
    training_config.get(
        "batch_size",
        512
    )
)


# ------------------------------------------------------------
# System configuration
# ------------------------------------------------------------

system_config = config.get(
    "system",
    {}
)

num_workers = int(
    system_config.get(
        "num_workers",
        0
    )
)


print(
    f"Sequence length: {sequence_length}"
)

print(
    f"Forecast horizon: {forecast_horizon}"
)

print(
    f"Batch size: {batch_size}"
)

print(
    f"Workers: {num_workers}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

if "Label" not in df.columns:

    raise ValueError(
        "Label column not found in dataset."
    )

df = df.dropna(
    how="all"
).reset_index(
    drop=True
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

print("\nLoading preprocessing artifacts...")

scaler = joblib.load(
    SCALER_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

with open(
    FEATURE_COLUMNS_PATH,
    "r",
    encoding="utf-8"
) as file:

    feature_columns = json.load(file)


print(
    f"Features: {len(feature_columns)}"
)

print(
    f"Classes: {len(label_encoder.classes_)}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing features...")

X = df[
    feature_columns
].copy()

for column in feature_columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

labels = df[
    "Label"
].astype(str)

y = label_encoder.transform(
    labels
)


# ============================================================
# REPRODUCE ORIGINAL DATA SPLIT
# ============================================================

print("\nReproducing training split...")

indices = np.arange(
    len(X)
)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=y[temp_idx]
)


# Same ordering used by the training pipeline
train_idx = np.sort(
    train_idx
)

val_idx = np.sort(
    val_idx
)

test_idx = np.sort(
    test_idx
)


print(
    f"Train: {len(train_idx):,}"
)

print(
    f"Validation: {len(val_idx):,}"
)

print(
    f"Test: {len(test_idx):,}"
)


# ============================================================
# PREPROCESSING
# ============================================================

print("\nApplying saved preprocessing...")

# The original training pipeline fitted its imputation
# statistics on the training split.
train_medians = X.iloc[
    train_idx
].median()


def preprocess_split(indices):

    data = X.iloc[
        indices
    ].copy()

    data = data.fillna(
        train_medians
    )

    data = data.fillna(
        0
    )

    transformed = scaler.transform(
        data
    )

    return transformed.astype(
        np.float32
    )


train_data = preprocess_split(
    train_idx
)

val_data = preprocess_split(
    val_idx
)

test_data = preprocess_split(
    test_idx
)

train_labels = y[
    train_idx
]

val_labels = y[
    val_idx
]

test_labels = y[
    test_idx
]

print("Preprocessing complete.")


# ============================================================
# LSTM ENCODER
# ============================================================

class LSTMEncoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

    def forward(self, x):

        output, _ = self.lstm(
            x
        )

        return output


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading trained LSTM checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


# ============================================================
# FIND LSTM WEIGHTS
# ============================================================

if any(
    key.startswith("encoder.")
    for key in state_dict.keys()
):

    prefix = "encoder."

elif any(
    key.startswith("lstm.")
    for key in state_dict.keys()
):

    prefix = "lstm."

else:

    raise RuntimeError(
        "Could not locate the LSTM encoder "
        "weights in the checkpoint."
    )


# ============================================================
# EXTRACT ENCODER WEIGHTS
# ============================================================

encoder_state = {}

for key, value in state_dict.items():

    if key.startswith(prefix):

        checkpoint_key = key[
            len(prefix):
        ]

        # LSTMEncoder contains:
        # self.lstm
        #
        # Therefore the checkpoint weights need
        # the "lstm." prefix.

        encoder_state[
            f"lstm.{checkpoint_key}"
        ] = value


# ============================================================
# DETECT MODEL DIMENSIONS FROM CHECKPOINT
# ============================================================

weight_ih_l0 = encoder_state[
    "lstm.weight_ih_l0"
]

weight_hh_l0 = encoder_state[
    "lstm.weight_hh_l0"
]

input_size = (
    weight_ih_l0.shape[1]
)

hidden_size = (
    weight_hh_l0.shape[1]
)

num_layers = len([
    key
    for key in encoder_state.keys()
    if key.startswith(
        "lstm.weight_ih_l"
    )
])


# Dropout does not affect checkpoint tensor
# dimensions. The original model used 0.2.
dropout = 0.2


print(
    f"Input size: {input_size}"
)

print(
    f"Hidden size: {hidden_size}"
)

print(
    f"LSTM layers: {num_layers}"
)


# ============================================================
# CREATE ENCODER
# ============================================================

encoder = LSTMEncoder(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)

encoder.load_state_dict(
    encoder_state
)

encoder.to(
    DEVICE
)

encoder.eval()

latent_dimension = hidden_size

print(
    f"Latent dimension: {latent_dimension}"
)


# ============================================================
# LATENT WINDOW DATASET
# ============================================================

class LatentWindowDataset(Dataset):

    def __init__(
        self,
        data,
        labels,
        seq_len
    ):

        self.data = data
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):

        return (
            len(self.data)
            - self.seq_len
            + 1
        )

    def __getitem__(self, index):

        end = (
            index
            + self.seq_len
        )

        x = self.data[
            index:end
        ]

        return (
            torch.from_numpy(x),
            index
        )


# ============================================================
# EXPORT FUNCTION
# ============================================================

def export_split(
    split_name,
    data,
    labels
):

    print("\n" + "-" * 70)

    print(
        f"Exporting {split_name.upper()}..."
    )

    print("-" * 70)


    dataset = LatentWindowDataset(
        data=data,
        labels=labels,
        seq_len=sequence_length
    )

    num_windows = len(
        dataset
    )

    valid_samples = (
        num_windows
        - forecast_horizon
    )


    if valid_samples <= 0:

        raise ValueError(
            f"{split_name} does not contain "
            "enough samples for the "
            "requested forecast horizon."
        )


    print(
        f"Windows: {num_windows:,}"
    )

    print(
        f"Valid forecast samples: "
        f"{valid_samples:,}"
    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    historical_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_historical_latents.npy"
    )

    final_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_final_latents.npy"
    )

    future_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_future_latents.npy"
    )

    future_labels_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_future_labels.npy"
    )


    # ========================================================
    # MEMORY-MAPPED OUTPUTS
    # ========================================================

    historical_latents = (
        np.lib.format.open_memmap(
            historical_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                num_windows,
                sequence_length,
                latent_dimension
            )
        )
    )

    final_latents = (
        np.lib.format.open_memmap(
            final_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                num_windows,
                latent_dimension
            )
        )
    )

    future_latents = (
        np.lib.format.open_memmap(
            future_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                valid_samples,
                forecast_horizon,
                latent_dimension
            )
        )
    )

    future_labels = (
        np.lib.format.open_memmap(
            future_labels_path,
            mode="w+",
            dtype=np.int64,
            shape=(
                valid_samples,
                forecast_horizon
            )
        )
    )


    # ========================================================
    # DATA LOADER
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # EXTRACT LATENTS
    # ========================================================

    position = 0

    with torch.no_grad():

        for batch_X, _ in loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            outputs = encoder(
                batch_X
            )

            outputs = (
                outputs
                .detach()
                .cpu()
                .numpy()
            )

            current_batch_size = (
                outputs.shape[0]
            )


            historical_latents[
                position:
                position + current_batch_size
            ] = outputs


            final_latents[
                position:
                position + current_batch_size
            ] = outputs[
                :, -1, :
            ]


            position += (
                current_batch_size
            )


            print(
                f"\rProcessed "
                f"{position:,}/"
                f"{num_windows:,}",
                end=""
            )


    print()


    # ========================================================
    # BUILD FUTURE LATENT TARGETS
    # ========================================================

    print(
        "Building future latent alignment..."
    )


    for start in range(
        valid_samples
    ):

        for horizon_step in range(
            forecast_horizon
        ):

            source_index = (
                start
                + horizon_step
                + 1
            )


            future_latents[
                start,
                horizon_step
            ] = final_latents[
                source_index
            ]


            label_index = (
                start
                + sequence_length
                + horizon_step
            )


            future_labels[
                start,
                horizon_step
            ] = labels[
                label_index
            ]


    # ========================================================
    # FLUSH MEMORY-MAPPED FILES
    # ========================================================

    historical_latents.flush()
    final_latents.flush()
    future_latents.flush()
    future_labels.flush()


    # ========================================================
    # SPLIT METADATA
    # ========================================================

    metadata = {

        "split": split_name,

        "sequence_length":
            sequence_length,

        "forecast_horizon":
            forecast_horizon,

        "latent_dimension":
            latent_dimension,

        "input_dimension":
            input_size,

        "num_windows":
            num_windows,

        "valid_samples":
            valid_samples,

        "num_classes":
            len(
                label_encoder.classes_
            ),

        "class_names": [
            str(class_name)
            for class_name
            in label_encoder.classes_
        ]
    }


    metadata_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_metadata.json"
    )


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


    # ========================================================
    # RESULT
    # ========================================================

    print(
        f"{split_name.upper()} export complete."
    )

    print(
        f"Historical: {historical_path}"
    )

    print(
        f"Final:      {final_path}"
    )

    print(
        f"Future:     {future_path}"
    )

    print(
        f"Labels:     {future_labels_path}"
    )


# ============================================================
# EXPORT TRAIN / VALIDATION / TEST
# ============================================================

print("\nExporting latent states...")

export_split(
    "train",
    train_data,
    train_labels
)

export_split(
    "val",
    val_data,
    val_labels
)

export_split(
    "test",
    test_data,
    test_labels
)


# ============================================================
# GLOBAL METADATA
# ============================================================

global_metadata = {

    "source":
        "NetSight AI LSTM encoder",

    "sequence_length":
        sequence_length,

    "forecast_horizon":
        forecast_horizon,

    "latent_dimension":
        latent_dimension,

    "input_dimension":
        input_size,

    "num_classes":
        len(
            label_encoder.classes_
        ),

    "classes": [
        str(class_name)
        for class_name
        in label_encoder.classes_
    ],

    "splits": [
        "train",
        "val",
        "test"
    ],

    "forecasting_directory":
        FORECASTING_DIR
}


global_metadata_path = os.path.join(
    FORECASTING_DATA_DIR,
    "lstm_forecasting_metadata.json"
)


with open(
    global_metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        global_metadata,
        file,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)

print(
    "LSTM LATENT EXPORT COMPLETED SUCCESSFULLY"
)

print("=" * 80)

print(
    "\nForecasting data:"
)

print(
    FORECASTING_DATA_DIR
)

print(
    "\nNo LSTM retraining was performed."
)

print("=" * 80)