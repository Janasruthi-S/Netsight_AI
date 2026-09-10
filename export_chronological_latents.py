import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        ".."
    )
)

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

OUTPUT_DIR = os.path.join(
    FORECASTING_DIR,
    "data",
    "chronological"
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
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 256

NUM_WORKERS = 0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD YAML
# ============================================================

try:
    import yaml
except ImportError:
    raise ImportError(
        "\nPyYAML is required.\n"
        "Install it using:\n\n"
        "pip install pyyaml\n"
    )


if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"\nConfig file not found:\n{CONFIG_PATH}"
    )


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = yaml.safe_load(f) or {}


temporal_config = config.get(
    "temporal",
    {}
)


SEQ_LEN = int(
    temporal_config.get(
        "sequence_length",
        20
    )
)

FORECAST_HORIZON = int(
    temporal_config.get(
        "forecast_horizon",
        5
    )
)

INTERVAL = temporal_config.get(
    "interval",
    None
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DATA_PATH,
    CHECKPOINT_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_COLUMNS_PATH,
    CONFIG_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("NETSIGHT AI - CHRONOLOGICAL LSTM LATENT EXPORT")
print("=" * 75)

print(
    f"Device            : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU               : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Dataset            : {DATA_PATH}"
)

print(
    f"Checkpoint         : {CHECKPOINT_PATH}"
)

print(
    f"Sequence length    : {SEQ_LEN}"
)

print(
    f"Forecast horizon   : {FORECAST_HORIZON}"
)

print(
    f"Temporal interval  : {INTERVAL}"
)

print(
    f"Batch size         : {BATCH_SIZE}"
)

print(
    f"Output directory   : {OUTPUT_DIR}"
)

print("=" * 75)


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

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# LABEL
# ============================================================

if "Label" not in df.columns:

    raise ValueError(
        "Label column was not found."
    )


df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = (
    df
    .dropna(how="all")
    .reset_index(drop=True)
)


# ============================================================
# LOAD SAVED FEATURE LIST
# ============================================================

with open(
    FEATURE_COLUMNS_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_columns = json.load(f)


if not isinstance(
    feature_columns,
    list
):

    raise ValueError(
        "feature_columns.json must contain a list."
    )


missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features:\n"
        + "\n".join(missing_features)
    )


INPUT_SIZE = len(
    feature_columns
)


print(
    f"Numerical features : {INPUT_SIZE}"
)


# ============================================================
# CREATE FEATURES
# ============================================================

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


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


y = label_encoder.transform(
    df["Label"]
)

y = np.asarray(
    y,
    dtype=np.int64
)


NUM_CLASSES = len(
    label_encoder.classes_
)


CLASS_NAMES = [
    str(x)
    for x in label_encoder.classes_
]


print(
    f"Classes            : {NUM_CLASSES}"
)


# ============================================================
# RECREATE ORIGINAL LSTM TRAINING SPLIT
#
# THIS IS ONLY TO RECREATE THE TRAINING MEDIANS.
#
# NO TRAINING HAPPENS HERE.
# ============================================================

print(
    "\nRecreating original LSTM preprocessing split..."
)


indices = np.arange(
    len(X)
)


train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)


val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=y[temp_indices]
)


train_indices = np.sort(
    train_indices
)

val_indices = np.sort(
    val_indices
)

test_indices = np.sort(
    test_indices
)


print(
    f"Original LSTM train : {len(train_indices):,}"
)

print(
    f"Original LSTM val   : {len(val_indices):,}"
)

print(
    f"Original LSTM test  : {len(test_indices):,}"
)


# ============================================================
# RECREATE TRAIN MEDIANS
# ============================================================

print(
    "\nRecreating training medians..."
)


train_medians = X.iloc[
    train_indices
].median()


X = X.fillna(
    train_medians
)

X = X.fillna(
    0
)


# ============================================================
# LOAD EXISTING SCALER
# ============================================================

print(
    "Loading existing scaler..."
)


scaler = joblib.load(
    SCALER_PATH
)


if scaler.n_features_in_ != INPUT_SIZE:

    raise ValueError(
        "Scaler feature count does not match "
        "feature_columns.json."
    )


# ============================================================
# SCALE
# ============================================================

print(
    "Applying existing scaler..."
)


X_scaled = scaler.transform(
    X
)

X_scaled = np.asarray(
    X_scaled,
    dtype=np.float32,
    order="C"
)


print(
    f"Scaled data shape  : {X_scaled.shape}"
)


# ============================================================
# FREE DATAFRAME MEMORY
# ============================================================

del X


# ============================================================
# EXACT MODEL FROM train_model.py
# ============================================================

class FinalWorldModelIDS(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        lstm_layers,
        transformer_layers,
        transformer_heads,
        dropout,
        sequence_length
    ):

        super().__init__()


        # ----------------------------------------------------
        # LSTM ENCODER
        # ----------------------------------------------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(
                dropout
                if lstm_layers > 1
                else 0.0
            )
        )


        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                hidden_size
            )
        )


        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )


        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=(
                    hidden_size
                    * 4
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )


        self.transformer = (
            nn.TransformerEncoder(
                transformer_layer,
                num_layers=transformer_layers,
                norm=nn.LayerNorm(
                    hidden_size
                )
            )
        )


        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        self.attention = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size // 2
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_size // 2,
                1
            )
        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                hidden_size
            ),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                num_classes
            )
        )


        # ----------------------------------------------------
        # WORLD MODEL INPUT
        # ----------------------------------------------------

        self.world_input = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # WORLD MODEL H/C
        # ----------------------------------------------------

        self.world_h = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.world_c = nn.Linear(
            hidden_size,
            hidden_size
        )


        # ----------------------------------------------------
        # WORLD DECODER
        # ----------------------------------------------------

        self.world_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )


        # ----------------------------------------------------
        # WORLD OUTPUT
        # ----------------------------------------------------

        self.world_output = nn.Sequential(

            nn.LayerNorm(
                hidden_size
            ),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_size,
                input_size
            )
        )


    def forward(self, x):

        encoder_output, _ = (
            self.encoder(x)
        )


        sequence_length = (
            encoder_output.size(1)
        )


        encoder_output = (
            encoder_output
            +
            self.position_embedding[
                :,
                :sequence_length,
                :
            ]
        )


        transformer_output = (
            self.transformer(
                encoder_output
            )
        )


        attention_scores = (
            self.attention(
                transformer_output
            )
        )


        attention_weights = (
            torch.softmax(
                attention_scores,
                dim=1
            )
        )


        context = torch.sum(
            transformer_output
            *
            attention_weights,
            dim=1
        )


        class_output = (
            self.classifier(
                context
            )
        )


        last_observation = (
            x[:, -1, :]
        )


        decoder_input = (
            self.world_input(
                last_observation
            )
            .unsqueeze(1)
        )


        h0 = torch.tanh(
            self.world_h(
                context
            )
        ).unsqueeze(0)


        c0 = torch.tanh(
            self.world_c(
                context
            )
        ).unsqueeze(0)


        world_output, _ = (
            self.world_decoder(
                decoder_input,
                (h0, c0)
            )
        )


        next_state = (
            self.world_output(
                world_output[:, -1, :]
            )
        )


        return (
            class_output,
            next_state,
            attention_weights
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\nLoading existing LSTM checkpoint..."
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)


if (
    not isinstance(
        checkpoint,
        dict
    )
    or
    "model_state_dict"
    not in checkpoint
):

    raise ValueError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )


state_dict = (
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# CHECKPOINT CONFIGURATION
# ============================================================

hidden_size = int(
    checkpoint.get(
        "hidden_size",
        128
    )
)

lstm_layers = int(
    checkpoint.get(
        "lstm_layers",
        2
    )
)

transformer_layers = int(
    checkpoint.get(
        "transformer_layers",
        2
    )
)

transformer_heads = int(
    checkpoint.get(
        "transformer_heads",
        8
    )
)

dropout = float(
    checkpoint.get(
        "dropout",
        0.20
    )
)

checkpoint_sequence_length = int(
    checkpoint.get(
        "sequence_length",
        SEQ_LEN
    )
)


if checkpoint_sequence_length != SEQ_LEN:

    raise ValueError(
        "\nSequence length mismatch.\n"
        f"Checkpoint : {checkpoint_sequence_length}\n"
        f"Config     : {SEQ_LEN}\n"
        "\nKeep the forecasting config sequence_length "
        "equal to the LSTM checkpoint sequence length."
    )


# ============================================================
# CREATE EXACT MODEL
# ============================================================

model = FinalWorldModelIDS(

    input_size=INPUT_SIZE,

    hidden_size=hidden_size,

    num_classes=NUM_CLASSES,

    lstm_layers=lstm_layers,

    transformer_layers=transformer_layers,

    transformer_heads=transformer_heads,

    dropout=dropout,

    sequence_length=SEQ_LEN
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        state_dict,
        strict=True
    )

except RuntimeError as error:

    print(
        "\nCHECKPOINT LOAD FAILED.\n"
    )

    print(
        error
    )

    raise


# ============================================================
# FREEZE EVERYTHING
# ============================================================

model = model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad = False


print(
    "\n✓ EXISTING MODEL LOADED SUCCESSFULLY"
)

print(
    f"Input dimension    : {INPUT_SIZE}"
)

print(
    f"Latent dimension   : {hidden_size}"
)

print(
    f"LSTM layers        : {lstm_layers}"
)

print(
    f"Transformer layers : {transformer_layers}"
)

print(
    f"Transformer heads  : {transformer_heads}"
)


# ============================================================
# SEQUENCE DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X_data,
        sequence_length
    ):

        self.X = X_data

        self.sequence_length = (
            sequence_length
        )


    def __len__(self):

        return (
            len(self.X)
            -
            self.sequence_length
            +
            1
        )


    def __getitem__(
        self,
        index
    ):

        sequence = self.X[
            index:
            index + self.sequence_length
        ]

        return torch.from_numpy(
            sequence
        )


# ============================================================
# EXTRACT LATENTS FROM FULL DATASET
#
# IMPORTANT:
#
# We run the FROZEN trained LSTM over the original row order.
#
# No optimizer.
# No backward().
# No model.train().
# No LSTM retraining.
# ============================================================

print("\n")
print("=" * 75)
print("EXTRACTING FROZEN LSTM LATENT STATES")
print("=" * 75)


sequence_dataset = SequenceDataset(
    X_scaled,
    SEQ_LEN
)


sequence_loader = DataLoader(
    sequence_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


NUMBER_OF_WINDOWS = len(
    sequence_dataset
)


print(
    f"Total LSTM windows : "
    f"{NUMBER_OF_WINDOWS:,}"
)


# ============================================================
# OUTPUT PATH
# ============================================================

FULL_LATENT_PATH = os.path.join(
    OUTPUT_DIR,
    "full_final_latents.npy"
)

FULL_LABEL_PATH = os.path.join(
    OUTPUT_DIR,
    "full_labels.npy"
)


# ============================================================
# CREATE MEMORY-MAPPED LATENT ARRAY
# ============================================================

latent_memmap = np.lib.format.open_memmap(

    FULL_LATENT_PATH,

    mode="w+",

    dtype=np.float32,

    shape=(
        NUMBER_OF_WINDOWS,
        hidden_size
    )
)


# ============================================================
# LATENT EXTRACTION
# ============================================================

write_position = 0

print("\nStarting extraction...\n")


with torch.inference_mode():

    for batch_number, batch_X in enumerate(
        sequence_loader,
        start=1
    ):

        batch_X = batch_X.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # EXACT SAME LSTM ENCODER
        # ----------------------------------------------------

        encoder_output, _ = (
            model.encoder(
                batch_X
            )
        )


        # ----------------------------------------------------
        # FINAL LSTM HIDDEN STATE
        #
        # This is the 128-dimensional latent state.
        # ----------------------------------------------------

        final_latent = (
            encoder_output[:, -1, :]
        )


        final_latent = (
            final_latent
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False
            )
        )


        current_batch = (
            final_latent.shape[0]
        )


        next_position = (
            write_position
            +
            current_batch
        )


        latent_memmap[
            write_position:
            next_position
        ] = final_latent


        write_position = (
            next_position
        )


        if (
            batch_number == 1
            or
            batch_number % 100 == 0
            or
            write_position == NUMBER_OF_WINDOWS
        ):

            print(
                f"Processed: "
                f"{write_position:,} / "
                f"{NUMBER_OF_WINDOWS:,}"
            )


latent_memmap.flush()

del latent_memmap


print(
    "\n✓ Latent extraction completed."
)


# ============================================================
# SAVE LABELS
# ============================================================

np.save(
    FULL_LABEL_PATH,
    y
)


print(
    f"Latent file:\n{FULL_LATENT_PATH}"
)

print(
    f"Label file:\n{FULL_LABEL_PATH}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
#
# ORIGINAL DATA:
#
# 0 ---------------- 70% ---------------- 85% -------- 100%
# |       TRAIN       |       VAL          |    TEST    |
#
# ============================================================

N = len(y)


train_end = int(
    N * TRAIN_RATIO
)

val_end = int(
    N *
    (
        TRAIN_RATIO
        +
        VAL_RATIO
    )
)


# ============================================================
# FORECASTING SAMPLE BOUNDARIES
#
# Anchor t:
#
# history:
#     t-19 ... t
#
# future:
#     t+1 ... t+5
#
# ============================================================

train_anchor_start = (
    SEQ_LEN - 1
)

train_anchor_end = (
    train_end
    -
    FORECAST_HORIZON
    -
    1
)


val_anchor_start = (
    train_end
)

val_anchor_end = (
    val_end
    -
    FORECAST_HORIZON
    -
    1
)


test_anchor_start = (
    val_end
)

test_anchor_end = (
    N
    -
    FORECAST_HORIZON
    -
    1
)


def range_count(
    start,
    end
):

    if end < start:

        return 0

    return (
        end
        -
        start
        +
        1
    )


train_samples = range_count(
    train_anchor_start,
    train_anchor_end
)

val_samples = range_count(
    val_anchor_start,
    val_anchor_end
)

test_samples = range_count(
    test_anchor_start,
    test_anchor_end
)


# ============================================================
# SAVE CHRONOLOGICAL METADATA
# ============================================================

metadata = {

    "dataset": {

        "rows": int(N),

        "features": int(
            INPUT_SIZE
        ),

        "classes": int(
            NUM_CLASSES
        ),

        "class_names": CLASS_NAMES
    },


    "lstm": {

        "checkpoint": CHECKPOINT_PATH,

        "input_dimension": int(
            INPUT_SIZE
        ),

        "latent_dimension": int(
            hidden_size
        ),

        "hidden_size": int(
            hidden_size
        ),

        "layers": int(
            lstm_layers
        ),

        "sequence_length": int(
            SEQ_LEN
        ),

        "frozen": True
    },


    "forecasting": {

        "forecast_horizon": int(
            FORECAST_HORIZON
        ),

        "temporal_interval": INTERVAL,

        "history_length": int(
            SEQ_LEN
        )
    },


    "chronological_split": {

        "method": (
            "original dataset row ordering"
        ),

        "train_ratio": TRAIN_RATIO,

        "validation_ratio": VAL_RATIO,

        "test_ratio": TEST_RATIO,


        "train": {

            "row_start": 0,

            "row_end_exclusive": int(
                train_end
            ),

            "anchor_start": int(
                train_anchor_start
            ),

            "anchor_end": int(
                train_anchor_end
            ),

            "samples": int(
                train_samples
            )
        },


        "validation": {

            "row_start": int(
                train_end
            ),

            "row_end_exclusive": int(
                val_end
            ),

            "anchor_start": int(
                val_anchor_start
            ),

            "anchor_end": int(
                val_anchor_end
            ),

            "samples": int(
                val_samples
            )
        },


        "test": {

            "row_start": int(
                val_end
            ),

            "row_end_exclusive": int(
                N
            ),

            "anchor_start": int(
                test_anchor_start
            ),

            "anchor_end": int(
                test_anchor_end
            ),

            "samples": int(
                test_samples
            )
        }
    },


    "files": {

        "latents": os.path.basename(
            FULL_LATENT_PATH
        ),

        "labels": os.path.basename(
            FULL_LABEL_PATH
        )
    },


    "scientific_limitation": (

        "The supplied dataset contains no timestamp column. "
        "Therefore original row ordering is used as the "
        "temporal ordering for this prototype. The forecast "
        "horizon represents future sequential records, not "
        "guaranteed real-world minutes. The LSTM checkpoint "
        "was previously trained using a random stratified "
        "split and is kept frozen here. Therefore this "
        "chronological forecasting evaluation is a prototype "
        "and should not be described as a completely "
        "leakage-free temporal experiment."
    )
}


METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "chronological_metadata.json"
)


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("CHRONOLOGICAL EXPORT COMPLETED")
print("=" * 75)

print(
    f"\nTotal dataset rows : {N:,}"
)

print(
    f"Features            : {INPUT_SIZE}"
)

print(
    f"Latent dimension    : {hidden_size}"
)

print(
    f"Sequence length     : {SEQ_LEN}"
)

print(
    f"Forecast horizon    : {FORECAST_HORIZON}"
)

print("\nForecasting samples:")

print(
    f"  Train             : "
    f"{train_samples:,}"
)

print(
    f"  Validation        : "
    f"{val_samples:,}"
)

print(
    f"  Test              : "
    f"{test_samples:,}"
)

print("\nGenerated files:")

print(
    f"  {FULL_LATENT_PATH}"
)

print(
    f"  {FULL_LABEL_PATH}"
)

print(
    f"  {METADATA_PATH}"
)

print("\n")
print("LSTM TRAINING: NOT PERFORMED")
print("LSTM CHECKPOINT: FROZEN")
print("BACKPROPAGATION: NOT PERFORMED")

print("\n")
print("=" * 75)
print("READY FOR FORECASTING MODULE")
print("=" * 75)