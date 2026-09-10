import os
import json
import time
import random
import warnings

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join("dataset", "combined_network_dataset.csv")

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================
# DATA CONFIGURATION
# ============================================================

SEQ_LEN = 20
PREDICTION_STEPS = 1


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 5


# ============================================================
# WORLD MODEL
# ============================================================

CLASSIFICATION_WEIGHT = 1.0
WORLD_MODEL_WEIGHT = 0.20


# ============================================================
# FOCAL LOSS
# ============================================================

FOCAL_GAMMA = 1.5
MAX_CLASS_WEIGHT = 6.0


# ============================================================
# MODEL CONFIGURATION
# ============================================================

LSTM_HIDDEN = 128
LSTM_LAYERS = 2
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_FF_MULTIPLIER = 4
DROPOUT = 0.20


# ============================================================
# GPU CONFIGURATION
# ============================================================

REQUIRE_CUDA = False

NUM_WORKERS_GPU =0


# ============================================================
# DEVICE
# ============================================================

cuda_available = torch.cuda.is_available()

if REQUIRE_CUDA and not cuda_available:
    raise RuntimeError(
        "\nCUDA GPU was not detected.\n"
        "Final training should be performed on an NVIDIA GPU.\n"
        "Install/use a CUDA-enabled PyTorch environment first."
    )

device = torch.device("cuda" if cuda_available else "cpu")


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


print("=" * 90)
print("FINAL NETWORK INTRUSION DETECTION SYSTEM")
print("LSTM + TRANSFORMER + LSTM WORLD MODEL")
print("=" * 90)

print("Device:", device)

if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    print("GPU Memory:", round(gpu_memory, 2), "GB")
    print("Mixed Precision: ENABLED")
else:
    print("GPU: NONE")
    print("Mixed Precision: DISABLED")
    print("WARNING: Training on CPU will be slow.")

print("=" * 90)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[1/12] Loading dataset...")

load_start = time.time()

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
    "Dataset loaded in:",
    round(time.time() - load_start, 2),
    "seconds"
)

print("Original rows:", f"{len(df):,}")
print("Original columns:", len(df.columns))


# ============================================================
# LABEL VALIDATION
# ============================================================

if "Label" not in df.columns:
    raise ValueError("Label column was not found.")

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

before = len(df)

df = df.dropna(
    how="all"
).reset_index(drop=True)

print(
    "Completely empty rows removed:",
    before - len(df)
)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_count = int(df.duplicated().sum())

print(
    "Duplicate rows detected:",
    f"{duplicate_count:,}"
)

print(
    "Duplicate rows retained for dataset completeness."
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_distribution = df["Label"].value_counts()

print(class_distribution.to_string())

print("\nNumber of classes:", len(class_distribution))


# ============================================================
# FEATURE DETECTION
# ============================================================

print("\n[2/12] Detecting numerical features...")

feature_columns = (
    df
    .drop(columns=["Label"])
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

print(
    "Number of numerical features:",
    len(feature_columns)
)

if len(feature_columns) < 10:
    raise ValueError("Too few numerical features detected.")

print("\nFeature list:")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i:03d}. {feature}")


# ============================================================
# CREATE X / Y
# ============================================================

X = df[feature_columns].copy()
y = df["Label"].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\nConverting features to numeric...")

for column in feature_columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

missing_count = int(X.isna().sum().sum())

print(
    "Missing / invalid values:",
    f"{missing_count:,}"
)


# ============================================================
# LABEL ENCODING
# ============================================================

print("\n[3/12] Encoding labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

print("Number of classes:", num_classes)

print("\nLabel mapping:")

for index, label in enumerate(label_encoder.classes_):
    print(f"{index:02d} -> {label}")


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

print("\n[4/12] Creating stratified dataset split...")

indices = np.arange(len(X))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y_encoded
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=y_encoded[temp_indices]
)

train_indices = np.sort(train_indices)
val_indices = np.sort(val_indices)
test_indices = np.sort(test_indices)

print("Training rows:", f"{len(train_indices):,}")
print("Validation rows:", f"{len(val_indices):,}")
print("Testing rows:", f"{len(test_indices):,}")


# ============================================================
# CREATE DATA SPLITS
# ============================================================

X_train = X.iloc[train_indices].copy()
X_val = X.iloc[val_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y_encoded[train_indices]
y_val = y_encoded[val_indices]
y_test = y_encoded[test_indices]


# ============================================================
# MISSING VALUE IMPUTATION
# ============================================================

print("\n[5/12] Handling missing values...")

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# FEATURE SCALING
# ============================================================

print("Standardizing features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = np.asarray(X_train, dtype=np.float32, order="C")
X_val = np.asarray(X_val, dtype=np.float32, order="C")
X_test = np.asarray(X_test, dtype=np.float32, order="C")


# ============================================================
# SAVE PREPROCESSING
# ============================================================

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

joblib.dump(
    label_encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

with open(
    os.path.join(MODEL_DIR, "feature_columns.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(feature_columns, f, indent=4)


# ============================================================
# DATASET DISTRIBUTIONS
# ============================================================

print("\nTraining distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nValidation distribution:")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTesting distribution:")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# WORLD MODEL DATASET
# ============================================================

class WorldModelSequenceDataset(Dataset):

    def __init__(self, X, y, sequence_length):

        self.X = X
        self.y = y
        self.sequence_length = sequence_length

        self.targets = np.asarray(
            self.y[
                sequence_length - 1:
                -1
            ],
            dtype=np.int64
        )

    def __len__(self):

        return max(
            0,
            len(self.X) - self.sequence_length
        )

    def __getitem__(self, index):

        start = index
        end = index + self.sequence_length

        sequence = self.X[start:end]

        classification_target = self.y[end - 1]

        next_state = self.X[end]

        return (
            torch.from_numpy(sequence),
            torch.tensor(
                classification_target,
                dtype=torch.long
            ),
            torch.from_numpy(next_state)
        )


# ============================================================
# CREATE DATASETS
# ============================================================

print("\n[6/12] Creating sequence datasets...")

train_dataset = WorldModelSequenceDataset(
    X_train,
    y_train,
    SEQ_LEN
)

val_dataset = WorldModelSequenceDataset(
    X_val,
    y_val,
    SEQ_LEN
)

test_dataset = WorldModelSequenceDataset(
    X_test,
    y_test,
    SEQ_LEN
)

print("Training sequences:", f"{len(train_dataset):,}")
print("Validation sequences:", f"{len(val_dataset):,}")
print("Testing sequences:", f"{len(test_dataset):,}")


# ============================================================
# IMBALANCE-AWARE SAMPLING
# ============================================================

print("\nPreparing imbalance-aware sampler...")

sequence_targets = train_dataset.targets

sequence_class_counts = np.bincount(
    sequence_targets,
    minlength=num_classes
)

print("\nSequence class counts:")

for i in range(num_classes):
    print(
        f"{i:02d} {label_encoder.classes_[i]}: "
        f"{sequence_class_counts[i]:,}"
    )

safe_counts = np.maximum(
    sequence_class_counts,
    1
)

sampling_class_weights = (
    1.0 / np.power(safe_counts, 0.75)
)

sample_weights = sampling_class_weights[sequence_targets]

sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

if cuda_available:
    num_workers = NUM_WORKERS_GPU
else:
    num_workers = 0

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": num_workers,
    "pin_memory": cuda_available,
}

if num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    **loader_kwargs
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **loader_kwargs
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **loader_kwargs
)

print("\nDataLoader workers:", num_workers)
print("Batch size:", BATCH_SIZE)


# ============================================================
# MODEL
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

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=(
                hidden_size * TRANSFORMER_FF_MULTIPLIER
            ),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_size)
        )

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

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                num_classes
            )
        )

        self.world_input = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size
            ),
            nn.GELU()
        )

        self.world_h = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.world_c = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.world_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.world_output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                input_size
            )
        )

    def forward(self, x):

        encoder_output, _ = self.encoder(x)

        sequence_length = encoder_output.size(1)

        encoder_output = (
            encoder_output
            +
            self.position_embedding[
                :, :sequence_length, :
            ]
        )

        transformer_output = self.transformer(
            encoder_output
        )

        attention_scores = self.attention(
            transformer_output
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            transformer_output
            * attention_weights,
            dim=1
        )

        class_output = self.classifier(context)

        last_observation = x[:, -1, :]

        decoder_input = (
            self.world_input(
                last_observation
            )
            .unsqueeze(1)
        )

        h0 = torch.tanh(
            self.world_h(context)
        ).unsqueeze(0)

        c0 = torch.tanh(
            self.world_c(context)
        ).unsqueeze(0)

        world_output, _ = self.world_decoder(
            decoder_input,
            (h0, c0)
        )

        next_state = self.world_output(
            world_output[:, -1, :]
        )

        return (
            class_output,
            next_state,
            attention_weights
        )


# ============================================================
# CREATE MODEL
# ============================================================

print("\n[7/12] Creating final model...")

input_size = len(feature_columns)

model = FinalWorldModelIDS(
    input_size=input_size,
    hidden_size=LSTM_HIDDEN,
    num_classes=num_classes,
    lstm_layers=LSTM_LAYERS,
    transformer_layers=TRANSFORMER_LAYERS,
    transformer_heads=TRANSFORMER_HEADS,
    dropout=DROPOUT,
    sequence_length=SEQ_LEN
)

model = model.to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Input features:", input_size)
print("Classes:", num_classes)
print("Total parameters:", f"{parameter_count:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")


# ============================================================
# CLASS-BALANCED FOCAL LOSS
# ============================================================

print("\n[8/12] Creating imbalance-aware loss...")


class ClassBalancedFocalLoss(nn.Module):

    def __init__(
        self,
        class_counts,
        gamma=1.5,
        max_weight=6.0
    ):

        super().__init__()

        counts = np.maximum(
            class_counts,
            1
        ).astype(np.float64)

        weights = (
            1.0 / np.log1p(counts)
        )

        weights = (
            weights / weights.mean()
        )

        weights = np.clip(
            weights,
            0.25,
            max_weight
        )

        weights = (
            weights / weights.mean()
        )

        self.register_buffer(
            "weights",
            torch.tensor(
                weights,
                dtype=torch.float32
            )
        )

        self.gamma = gamma

    def forward(self, logits, targets):

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        probabilities = torch.exp(-ce)

        focal_factor = (
            1.0 - probabilities
        ) ** self.gamma

        alpha = self.weights[targets]

        loss = (
            alpha
            * focal_factor
            * ce
        )

        return loss.mean()


classification_criterion = ClassBalancedFocalLoss(
    sequence_class_counts,
    gamma=FOCAL_GAMMA,
    max_weight=MAX_CLASS_WEIGHT
).to(device)


# ============================================================
# WORLD MODEL LOSS
# ============================================================

world_model_criterion = nn.SmoothL1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = device.type == "cuda"

if use_amp:
    amp_scaler = torch.amp.GradScaler("cuda")
    print("AMP: ENABLED")
else:
    amp_scaler = None
    print("AMP: DISABLED")


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_world_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_X, batch_y, batch_next in train_loader:

        batch_X = batch_X.to(
            device,
            non_blocking=True
        )

        batch_y = batch_y.to(
            device,
            non_blocking=True
        )

        batch_next = batch_next.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                class_output, next_prediction, _ = model(
                    batch_X
                )

                classification_loss = classification_criterion(
                    class_output,
                    batch_y
                )

                world_loss = world_model_criterion(
                    next_prediction,
                    batch_next
                )

                loss = (
                    CLASSIFICATION_WEIGHT
                    * classification_loss
                    +
                    WORLD_MODEL_WEIGHT
                    * world_loss
                )

            amp_scaler.scale(loss).backward()

            amp_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            amp_scaler.step(optimizer)
            amp_scaler.update()

        else:

            class_output, next_prediction, _ = model(
                batch_X
            )

            classification_loss = classification_criterion(
                class_output,
                batch_y
            )

            world_loss = world_model_criterion(
                next_prediction,
                batch_next
            )

            loss = (
                CLASSIFICATION_WEIGHT
                * classification_loss
                +
                WORLD_MODEL_WEIGHT
                * world_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

        current_batch_size = batch_X.size(0)

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_classification_loss += (
            classification_loss.item()
            * current_batch_size
        )

        total_world_loss += (
            world_loss.item()
            * current_batch_size
        )

        predictions = torch.argmax(
            class_output,
            dim=1
        )

        total_correct += (
            (predictions == batch_y)
            .sum()
            .item()
        )

        total_samples += current_batch_size

    return (
        total_loss / total_samples,
        total_classification_loss / total_samples,
        total_world_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_world_loss = 0.0

    targets = []
    predictions_all = []
    world_errors = []

    with torch.no_grad():

        for batch_X, batch_y, batch_next in loader:

            batch_X = batch_X.to(
                device,
                non_blocking=True
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True
            )

            batch_next = batch_next.to(
                device,
                non_blocking=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    class_output, next_prediction, _ = model(
                        batch_X
                    )

                    classification_loss = classification_criterion(
                        class_output,
                        batch_y
                    )

                    world_loss = world_model_criterion(
                        next_prediction,
                        batch_next
                    )

            else:

                class_output, next_prediction, _ = model(
                    batch_X
                )

                classification_loss = classification_criterion(
                    class_output,
                    batch_y
                )

                world_loss = world_model_criterion(
                    next_prediction,
                    batch_next
                )

            loss = (
                CLASSIFICATION_WEIGHT
                * classification_loss
                +
                WORLD_MODEL_WEIGHT
                * world_loss
            )

            current_batch_size = batch_X.size(0)

            total_loss += (
                loss.item()
                * current_batch_size
            )

            total_classification_loss += (
                classification_loss.item()
                * current_batch_size
            )

            total_world_loss += (
                world_loss.item()
                * current_batch_size
            )

            predictions = torch.argmax(
                class_output,
                dim=1
            )

            targets.extend(
                batch_y.cpu().numpy().tolist()
            )

            predictions_all.extend(
                predictions.cpu().numpy().tolist()
            )

            sample_errors = torch.mean(
                (
                    next_prediction
                    -
                    batch_next
                ) ** 2,
                dim=1
            )

            world_errors.extend(
                sample_errors.cpu().numpy().tolist()
            )

    targets = np.asarray(targets)
    predictions_all = np.asarray(predictions_all)
    world_errors = np.asarray(world_errors)

    accuracy = accuracy_score(
        targets,
        predictions_all
    )

    balanced_accuracy = balanced_accuracy_score(
        targets,
        predictions_all
    )

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions_all,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions_all,
            average="weighted",
            zero_division=0
        )
    )

    world_mae = float(
        np.mean(np.sqrt(world_errors))
    )

    world_rmse = float(
        np.sqrt(np.mean(world_errors))
    )

    return {
        "loss": total_loss / len(targets),
        "classification_loss": (
            total_classification_loss / len(targets)
        ),
        "world_loss": (
            total_world_loss / len(targets)
        ),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "world_mae": world_mae,
        "world_rmse": world_rmse,
        "targets": targets,
        "predictions": predictions_all,
        "world_errors": world_errors
    }


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 90)
print("STARTING FINAL WORLD MODEL TRAINING")
print("=" * 90)

best_macro_f1 = -1.0
best_balanced_accuracy = -1.0
epochs_without_improvement = 0
history = []
training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    (
        train_loss,
        train_classification_loss,
        train_world_loss,
        train_accuracy
    ) = train_epoch()

    val = evaluate(val_loader)

    scheduler.step(val["macro_f1"])

    epoch_time = time.time() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n")
    print("-" * 90)
    print(f"Epoch [{epoch}/{EPOCHS}]")
    print("-" * 90)

    print(f"Train Loss             : {train_loss:.6f}")
    print(f"Train Classification   : {train_classification_loss:.6f}")
    print(f"Train World Loss       : {train_world_loss:.6f}")
    print(f"Train Accuracy         : {train_accuracy:.4f}")
    print(f"Val Loss               : {val['loss']:.6f}")
    print(f"Val Classification     : {val['classification_loss']:.6f}")
    print(f"Val World Loss         : {val['world_loss']:.6f}")
    print(f"Val Accuracy           : {val['accuracy']:.4f}")
    print(f"Val Balanced Accuracy  : {val['balanced_accuracy']:.4f}")
    print(f"Val Macro Precision    : {val['macro_precision']:.4f}")
    print(f"Val Macro Recall       : {val['macro_recall']:.4f}")
    print(f"Val Macro F1           : {val['macro_f1']:.4f}")
    print(f"Val Weighted F1        : {val['weighted_f1']:.4f}")
    print(f"World MAE              : {val['world_mae']:.6f}")
    print(f"World RMSE             : {val['world_rmse']:.6f}")
    print(f"Learning Rate          : {current_lr:.8f}")
    print(f"Epoch Time             : {epoch_time / 60:.2f} minutes")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_classification_loss": train_classification_loss,
        "train_world_loss": train_world_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val["loss"],
        "val_classification_loss": val["classification_loss"],
        "val_world_loss": val["world_loss"],
        "val_accuracy": val["accuracy"],
        "val_balanced_accuracy": val["balanced_accuracy"],
        "val_macro_precision": val["macro_precision"],
        "val_macro_recall": val["macro_recall"],
        "val_macro_f1": val["macro_f1"],
        "val_weighted_f1": val["weighted_f1"],
        "val_world_mae": val["world_mae"],
        "val_world_rmse": val["world_rmse"],
        "learning_rate": current_lr,
        "epoch_time_seconds": epoch_time
    })

    if val["macro_f1"] > best_macro_f1:

        best_macro_f1 = val["macro_f1"]
        best_balanced_accuracy = val["balanced_accuracy"]
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": LSTM_HIDDEN,
            "lstm_layers": LSTM_LAYERS,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_heads": TRANSFORMER_HEADS,
            "dropout": DROPOUT,
            "num_classes": num_classes,
            "sequence_length": SEQ_LEN,
            "classification_weight": CLASSIFICATION_WEIGHT,
            "world_model_weight": WORLD_MODEL_WEIGHT,
            "focal_gamma": FOCAL_GAMMA,
            "classes": label_encoder.classes_.tolist(),
            "features": feature_columns,
            "class_weights": (
                classification_criterion
                .weights
                .detach()
                .cpu()
                .numpy()
                .tolist()
            ),
            "best_validation_macro_f1": float(best_macro_f1),
            "best_validation_balanced_accuracy": float(
                best_balanced_accuracy
            ),
            "validation_world_mae": float(val["world_mae"]),
            "validation_world_rmse": float(val["world_rmse"])
        }

        torch.save(
            checkpoint,
            os.path.join(
                MODEL_DIR,
                "best_world_model_ids.pth"
            )
        )

        print("\n✓ BEST MODEL SAVED")
        print(f"Best Macro F1: {best_macro_f1:.4f}")

    else:

        epochs_without_improvement += 1

        print("\nNo Macro F1 improvement.")
        print(
            "Patience:",
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    pd.DataFrame(history).to_csv(
        os.path.join(
            RESULT_DIR,
            "training_history.csv"
        ),
        index=False
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEARLY STOPPING TRIGGERED.")
        print("Validation Macro F1 stopped improving.")
        break


# ============================================================
# TRAINING FINISHED
# ============================================================

training_time_minutes = (
    time.time() - training_start
) / 60

print("\n")
print("=" * 90)
print("TRAINING FINISHED")
print("=" * 90)

print(
    "Best Validation Macro F1:",
    f"{best_macro_f1:.4f}"
)

print(
    "Best Validation Balanced Accuracy:",
    f"{best_balanced_accuracy:.4f}"
)

print(
    "Training Time:",
    f"{training_time_minutes:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n[10/12] Loading best model...")

best_model_path = os.path.join(
    MODEL_DIR,
    "best_world_model_ids.pth"
)

checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Best model loaded successfully.")


# ============================================================
# FINAL TEST
# ============================================================

print("\n[11/12] Evaluating final test dataset...")

test = evaluate(test_loader)


# ============================================================
# FINAL TEST RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FINAL TEST RESULTS")
print("=" * 90)

print(f"Test Accuracy            : {test['accuracy']:.4f}")
print(f"Test Balanced Accuracy   : {test['balanced_accuracy']:.4f}")
print(f"Test Macro Precision     : {test['macro_precision']:.4f}")
print(f"Test Macro Recall        : {test['macro_recall']:.4f}")
print(f"Test Macro F1            : {test['macro_f1']:.4f}")
print(f"Test Weighted F1         : {test['weighted_f1']:.4f}")
print(f"Test World MAE           : {test['world_mae']:.6f}")
print(f"Test World RMSE          : {test['world_rmse']:.6f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 90)
print("FINAL CLASSIFICATION REPORT")
print("=" * 90)

report = classification_report(
    test["targets"],
    test["predictions"],
    labels=np.arange(num_classes),
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test["targets"],
    test["predictions"],
    labels=np.arange(num_classes)
)

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = precision_recall_fscore_support(
    test["targets"],
    test["predictions"],
    labels=np.arange(num_classes),
    zero_division=0
)

per_class_df = pd.DataFrame({
    "class": label_encoder.classes_,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})

per_class_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "per_class_metrics.csv"
    ),
    index=False
)


# ============================================================
# WORLD MODEL ERROR
# ============================================================

world_error_df = pd.DataFrame({
    "true_class": label_encoder.inverse_transform(
        test["targets"]
    ),
    "world_prediction_mse": test["world_errors"]
})

world_error_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "world_model_errors.csv"
    ),
    index=False
)


# ============================================================
# WORLD MODEL ERROR BY CLASS
# ============================================================

world_error_by_class = (
    world_error_df
    .groupby("true_class")
    ["world_prediction_mse"]
    .agg([
        "count",
        "mean",
        "median",
        "std"
    ])
    .sort_values(
        "mean",
        ascending=False
    )
)

world_error_by_class.to_csv(
    os.path.join(
        RESULT_DIR,
        "world_model_error_by_class.csv"
    )
)

print("\n")
print("=" * 90)
print("WORLD MODEL ERROR BY CLASS")
print("=" * 90)
print(world_error_by_class)


# ============================================================
# WORLD MODEL ANOMALY THRESHOLD
# ============================================================

print("\n")
print("Calculating world-model anomaly threshold...")

val = evaluate(val_loader)

benign_matches = np.where(
    np.char.upper(
        label_encoder.inverse_transform(
            val["targets"]
        ).astype(str)
    ) == "BENIGN"
)[0]

if len(benign_matches) > 0:

    benign_errors = val["world_errors"][benign_matches]

    world_anomaly_threshold = float(
        np.percentile(
            benign_errors,
            99
        )
    )

else:

    world_anomaly_threshold = float(
        np.percentile(
            val["world_errors"],
            99
        )
    )

print(
    "World-model anomaly threshold:",
    f"{world_anomaly_threshold:.6f}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n[12/12] Saving final model...")

final_model_path = os.path.join(
    MODEL_DIR,
    "final_world_model_ids.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "transformer_layers": TRANSFORMER_LAYERS,
        "transformer_heads": TRANSFORMER_HEADS,
        "dropout": DROPOUT,
        "num_classes": num_classes,
        "sequence_length": SEQ_LEN,
        "classification_weight": CLASSIFICATION_WEIGHT,
        "world_model_weight": WORLD_MODEL_WEIGHT,
        "focal_gamma": FOCAL_GAMMA,
        "classes": label_encoder.classes_.tolist(),
        "features": feature_columns,
        "world_anomaly_threshold": world_anomaly_threshold,
        "test_accuracy": float(test["accuracy"]),
        "test_balanced_accuracy": float(test["balanced_accuracy"]),
        "test_macro_precision": float(test["macro_precision"]),
        "test_macro_recall": float(test["macro_recall"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "test_world_mae": float(test["world_mae"]),
        "test_world_rmse": float(test["world_rmse"])
    },
    final_model_path
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "architecture": "LSTM + Transformer + LSTM World Model",
    "total_rows": int(len(df)),
    "duplicate_rows_detected": duplicate_count,
    "total_features": int(len(feature_columns)),
    "number_of_classes": int(num_classes),
    "classes": label_encoder.classes_.tolist(),
    "sequence_length": SEQ_LEN,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "device": str(device),
    "world_model": True,
    "world_model_type": "LSTM next-state prediction",
    "classification_weight": CLASSIFICATION_WEIGHT,
    "world_model_weight": WORLD_MODEL_WEIGHT,
    "imbalance_method": (
        "Moderate weighted sampling + "
        "logarithmic class-balanced focal loss"
    ),
    "focal_gamma": FOCAL_GAMMA,
    "best_validation_macro_f1": float(best_macro_f1),
    "best_validation_balanced_accuracy": float(
        best_balanced_accuracy
    ),
    "test_accuracy": float(test["accuracy"]),
    "test_balanced_accuracy": float(
        test["balanced_accuracy"]
    ),
    "test_macro_precision": float(
        test["macro_precision"]
    ),
    "test_macro_recall": float(
        test["macro_recall"]
    ),
    "test_macro_f1": float(test["macro_f1"]),
    "test_weighted_f1": float(
        test["weighted_f1"]
    ),
    "test_world_mae": float(
        test["world_mae"]
    ),
    "test_world_rmse": float(
        test["world_rmse"]
    ),
    "world_anomaly_threshold": float(
        world_anomaly_threshold
    ),
    "training_time_minutes": float(
        training_time_minutes
    )
}

with open(
    os.path.join(
        RESULT_DIR,
        "training_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FORECASTING LATENT EXPORT
# ============================================================

print("\n")
print("=" * 90)
print("EXPORTING LSTM LATENT STATES FOR FORECASTING MODULE")
print("=" * 90)


# ------------------------------------------------------------
# Forecasting configuration
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FORECASTING_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "forecasting"
    )
)

FORECASTING_DATA_DIR = os.path.join(
    FORECASTING_DIR,
    "data"
)

os.makedirs(
    FORECASTING_DATA_DIR,
    exist_ok=True
)

print("Forecasting directory:")
print(FORECASTING_DIR)


CONFIG_PATH = os.path.join(
    FORECASTING_DIR,
    "config.yaml"
)

if not os.path.exists(CONFIG_PATH):

    raise FileNotFoundError(
        "\nForecasting config was not found:\n"
        f"{CONFIG_PATH}\n\n"
        "Create forecasting/config.yaml containing:\n\n"
        "temporal:\n"
        "  forecast_horizon: 5\n"
    )


try:

    import yaml

except ImportError:

    raise ImportError(
        "\nPyYAML is required for the forecasting configuration.\n"
        "Install it with:\n"
        "pip install pyyaml"
    )


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    forecasting_config = yaml.safe_load(f) or {}


temporal_config = forecasting_config.get(
    "temporal",
    {}
)

if "forecast_horizon" not in temporal_config:

    raise ValueError(
        "\nforecast_horizon is missing from "
        "forecasting/config.yaml.\n\n"
        "Add:\n"
        "temporal:\n"
        "  forecast_horizon: 5"
    )


FORECAST_HORIZON = int(
    temporal_config["forecast_horizon"]
)

if FORECAST_HORIZON < 1:

    raise ValueError(
        "forecast_horizon must be >= 1"
    )


EXPORT_BATCH_SIZE = int(
    forecasting_config
    .get("training", {})
    .get("batch_size", BATCH_SIZE)
)

EXPORT_NUM_WORKERS = int(
    forecasting_config
    .get("system", {})
    .get("num_workers", 0)
)

if EXPORT_BATCH_SIZE < 1:
    raise ValueError(
        "Forecasting export batch_size must be >= 1."
    )


print(
    "Sequence length:",
    SEQ_LEN
)

print(
    "Forecast horizon:",
    FORECAST_HORIZON
)

print(
    "Export batch size:",
    EXPORT_BATCH_SIZE
)

print(
    "Export device:",
    device
)


# ------------------------------------------------------------
# Latent extraction dataset
# ------------------------------------------------------------

class LatentWindowDataset(Dataset):

    """
    Produces every sliding LSTM input window.

    For a split containing N rows and a sequence length T,
    the number of windows is:

        N - T + 1

    No future information is included in a window.
    """

    def __init__(
        self,
        X_data,
        sequence_length
    ):

        self.X = X_data
        self.sequence_length = sequence_length

        if len(self.X) < self.sequence_length:

            raise ValueError(
                "Dataset split is smaller than "
                "the LSTM sequence length."
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

        window = self.X[
            index:
            index + self.sequence_length
        ]

        return torch.from_numpy(
            window
        )


# ------------------------------------------------------------
# Extract latent windows to memory-mapped .npy files
# ------------------------------------------------------------

def export_split_latents(
    X_data,
    y_data,
    split_name
):

    print("\n" + "-" * 90)
    print(
        f"EXPORTING {split_name.upper()} LATENTS"
    )
    print("-" * 90)

    dataset = LatentWindowDataset(
        X_data,
        SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=EXPORT_BATCH_SIZE,
        shuffle=False,
        num_workers=EXPORT_NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    number_of_windows = len(dataset)

    valid_samples = (
        number_of_windows
        -
        FORECAST_HORIZON
    )

    if valid_samples <= 0:

        raise ValueError(
            f"{split_name} split does not contain enough "
            f"rows for sequence_length={SEQ_LEN} and "
            f"forecast_horizon={FORECAST_HORIZON}."
        )


    # --------------------------------------------------------
    # Determine latent dimension from the actual model.
    # --------------------------------------------------------

    latent_dimension = int(
        model.encoder.hidden_size
    )


    # --------------------------------------------------------
    # Create disk-backed files.
    #
    # These are .npy files and do not require keeping the
    # entire latent dataset in RAM.
    # --------------------------------------------------------

    historical_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_historical_latents.npy"
    )

    final_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_final_latents.npy"
    )

    future_latent_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_future_latents.npy"
    )

    future_label_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_future_labels.npy"
    )


    historical_memmap = np.lib.format.open_memmap(
        historical_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            number_of_windows,
            SEQ_LEN,
            latent_dimension
        )
    )

    final_memmap = np.lib.format.open_memmap(
        final_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            number_of_windows,
            latent_dimension
        )
    )


    model.eval()

    write_position = 0

    export_start = time.time()


    with torch.no_grad():

        for batch_index, batch_X in enumerate(loader):

            batch_X = batch_X.to(
                device,
                non_blocking=True
            )


            # IMPORTANT:
            # Your existing LSTM is model.encoder.
            #

            lstm_output, _ = model.encoder(
                batch_X
            )


            batch_latent = (
                lstm_output
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False
                )
            )


            batch_final = batch_latent[:, -1, :]


            batch_size_actual = (
                batch_latent.shape[0]
            )

            next_position = (
                write_position
                +
                batch_size_actual
            )


            historical_memmap[
                write_position:next_position
            ] = batch_latent


            final_memmap[
                write_position:next_position
            ] = batch_final


            write_position = next_position


            if (
                batch_index == 0
                or
                (batch_index + 1) % 100 == 0
                or
                (batch_index + 1) == len(loader)
            ):

                print(
                    f"Processed windows: "
                    f"{write_position:,}/"
                    f"{number_of_windows:,}"
                )


    historical_memmap.flush()
    final_memmap.flush()

    del historical_memmap
    del final_memmap


    # --------------------------------------------------------
    # Construct only the valid forecasting portion.
    #
    # We use memory mapping so this step does not duplicate
    # the complete latent arrays in RAM.
    # --------------------------------------------------------

    historical_source = np.load(
        historical_path,
        mmap_mode="r"
    )

    final_source = np.load(
        final_path,
        mmap_mode="r"
    )


    # --------------------------------------------------------
    # Create future latent array on disk.
    # --------------------------------------------------------

    future_latent_memmap = np.lib.format.open_memmap(
        future_latent_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            valid_samples,
            FORECAST_HORIZON,
            latent_dimension
        )
    )


    # --------------------------------------------------------
    # Create future labels directly on disk.
    # --------------------------------------------------------

    future_label_memmap = np.lib.format.open_memmap(
        future_label_path,
        mode="w+",
        dtype=np.int64,
        shape=(
            valid_samples,
            FORECAST_HORIZON
        )
    )


    # --------------------------------------------------------
    # Fill future targets in chunks.
    # --------------------------------------------------------

    target_chunk_size = max(
        1,
        min(
            EXPORT_BATCH_SIZE,
            8192
        )
    )


    for start in range(
        0,
        valid_samples,
        target_chunk_size
    ):

        end = min(
            start + target_chunk_size,
            valid_samples
        )

        current_count = end - start


        for h in range(
            FORECAST_HORIZON
        ):

            # Future window ending h+1 steps after
            # the current anchor.

            source_start = (
                start + h + 1
            )

            source_end = (
                end + h + 1
            )

            future_latent_memmap[
                start:end,
                h,
                :
            ] = final_source[
                source_start:source_end
            ]


            # Current forecasting sample starts at row:
            #
            #     start + SEQ_LEN
            #
            # and advances one row for each sample.
            #

            label_start = (
                start
                +
                SEQ_LEN
                +
                h
            )

            label_end = (
                label_start
                +
                current_count
            )

            future_label_memmap[
                start:end,
                h
            ] = y_data[
                label_start:label_end
            ]


    future_latent_memmap.flush()
    future_label_memmap.flush()

    del future_latent_memmap
    del future_label_memmap

    del historical_source
    del final_source


    # --------------------------------------------------------
    # Save compact metadata for this split.
    # --------------------------------------------------------

    split_metadata = {
        "split": split_name,
        "original_rows": int(len(X_data)),
        "number_of_lstm_windows": int(number_of_windows),
        "forecast_samples": int(valid_samples),
        "sequence_length": int(SEQ_LEN),
        "forecast_horizon": int(FORECAST_HORIZON),
        "latent_dimension": int(latent_dimension),
        "input_dimension": int(X_data.shape[1]),
        "historical_latents": os.path.basename(
            historical_path
        ),
        "future_latents": os.path.basename(
            future_latent_path
        ),
        "future_labels": os.path.basename(
            future_label_path
        )
    }

    split_metadata_path = os.path.join(
        FORECASTING_DATA_DIR,
        f"{split_name}_metadata.json"
    )

    with open(
        split_metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_metadata,
            f,
            indent=4
        )


    elapsed = (
        time.time()
        -
        export_start
    )


    print("\nShapes:")

    print(
        "Historical latents:",
        (
            valid_samples,
            SEQ_LEN,
            latent_dimension
        )
    )

    print(
        "Future latents:",
        (
            valid_samples,
            FORECAST_HORIZON,
            latent_dimension
        )
    )

    print(
        "Future labels:",
        (
            valid_samples,
            FORECAST_HORIZON
        )
    )

    print(
        f"Export time: {elapsed / 60:.2f} minutes"
    )

    print(
        "Saved historical:",
        historical_path
    )

    print(
        "Saved future latents:",
        future_latent_path
    )

    print(
        "Saved future labels:",
        future_label_path
    )

    return split_metadata


# ------------------------------------------------------------
# Export train / validation / test independently.
# ------------------------------------------------------------

forecasting_splits = {
    "train": (
        X_train,
        y_train
    ),
    "val": (
        X_val,
        y_val
    ),
    "test": (
        X_test,
        y_test
    )
}


all_forecasting_metadata = {}

for split_name, (
    X_split,
    y_split
) in forecasting_splits.items():

    all_forecasting_metadata[
        split_name
    ] = export_split_latents(
        X_split,
        y_split,
        split_name
    )


# ------------------------------------------------------------
# Global forecasting metadata
# ------------------------------------------------------------

forecasting_metadata = {

    "source": "train_model.py",

    "architecture": (
        "LSTM encoder from FinalWorldModelIDS"
    ),

    "sequence_length": int(
        SEQ_LEN
    ),

    "forecast_horizon": int(
        FORECAST_HORIZON
    ),

    "input_dimension": int(
        input_size
    ),

    "latent_dimension": int(
        model.encoder.hidden_size
    ),

    "num_classes": int(
        num_classes
    ),

    "classes": [
        str(label)
        for label in label_encoder.classes_
    ],

    "feature_columns": [
        str(feature)
        for feature in feature_columns
    ],

    "splits": all_forecasting_metadata
}


metadata_path = os.path.join(
    FORECASTING_DATA_DIR,
    "lstm_forecasting_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        forecasting_metadata,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 90)
print("LSTM LATENT EXPORT COMPLETED")
print("=" * 90)

print(
    "\nForecasting data directory:"
)

print(
    FORECASTING_DATA_DIR
)

print("\nGenerated forecasting files:")

for filename in sorted(
    os.listdir(
        FORECASTING_DATA_DIR
    )
):

    print(
        "  -",
        filename
    )

print("\nForecasting metadata:")

print(
    json.dumps(
        forecasting_metadata,
        indent=4
    )
)

print("\n")
print("=" * 90)
print("FINAL IDS TRAINING + FORECASTING EXPORT COMPLETE")
print("=" * 90)
