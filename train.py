import os
import json
import yaml
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from forecasting_model import (
    ForecastingModel,
    calculate_class_weights,
    train_model,
    DEVICE,
)


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CONFIG_PATH = os.path.join(
    SCRIPT_DIR,
    "config.yaml",
)

DATA_DIR = os.path.join(
    SCRIPT_DIR,
    "data",
    "chronological",
)

LATENT_PATH = os.path.join(
    DATA_DIR,
    "full_final_latents.npy",
)

LABEL_PATH = os.path.join(
    DATA_DIR,
    "full_labels.npy",
)

METADATA_PATH = os.path.join(
    DATA_DIR,
    "chronological_metadata.json",
)


# ============================================================================
# CONFIG / FILE HELPERS
# ============================================================================

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================================
# STRATIFIED FORECAST DATASET
# ============================================================================

class ForecastDataset(Dataset):
    """
    Forecasting sample:

        Input:
            latent[t-history+1 ... t]

        Target:
            future latent[t+1 ... t+horizon]
            future labels[t+1 ... t+horizon]

    The dataset uses explicitly selected anchor indices.
    """

    def __init__(
        self,
        latents,
        labels,
        anchors,
        sequence_length,
        horizon,
    ):
        self.latents = latents
        self.labels = labels

        self.anchors = np.asarray(
            anchors,
            dtype=np.int64,
        )

        self.sequence_length = int(
            sequence_length
        )

        self.horizon = int(
            horizon
        )

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):

        anchor = int(
            self.anchors[index]
        )

        history_start = (
            anchor
            - self.sequence_length
            + 1
        )

        history_end = anchor + 1

        future_start = anchor + 1

        future_end = (
            anchor
            + 1
            + self.horizon
        )

        # IMPORTANT:
        # copy=True removes the read-only mmap warning.
        history = np.array(
            self.latents[
                history_start:history_end
            ],
            dtype=np.float32,
            copy=True,
        )

        future_states = np.array(
            self.latents[
                future_start:future_end
            ],
            dtype=np.float32,
            copy=True,
        )

        future_labels = np.array(
            self.labels[
                future_start:future_end
            ],
            dtype=np.int64,
            copy=True,
        )

        return (
            history,
            future_states,
            future_labels,
        )


# ============================================================================
# BENIGN
# ============================================================================

def get_benign_index(class_names):

    matches = [
        i
        for i, name in enumerate(class_names)
        if str(name).strip().lower() == "benign"
    ]

    if not matches:
        raise ValueError(
            "BENIGN class not found in class metadata."
        )

    return matches[0]


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_config(config):

    for key in [
        "transformer",
        "gru",
        "tcn",
    ]:
        if key not in config["model"]:
            raise KeyError(
                f"Missing model configuration: model.{key}"
            )

    if int(
        config["temporal"]["sequence_length"]
    ) < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    if int(
        config["temporal"]["forecast_horizon"]
    ) < 1:
        raise ValueError(
            "forecast_horizon must be >= 1."
        )


# ============================================================================
# STRATIFIED ANCHOR SELECTION
# ============================================================================

def build_stratified_splits(
    labels,
    sequence_length,
    horizon,
    num_classes,
    max_train_samples=400000,
    max_val_samples=60000,
    max_test_samples=60000,
    seed=42,
):
    """
    Creates forecasting anchors while preserving attack-class coverage.

    IMPORTANT:
    This is a prototype/evaluation split because the supplied merged
    dataset does not contain explicit timestamps.

    The target class used for stratification is the first future label.

    The complete horizon is still used during forecasting.
    """

    rng = np.random.default_rng(seed)

    num_rows = len(labels)

    first_anchor = (
        sequence_length - 1
    )

    last_anchor = (
        num_rows
        - horizon
        - 1
    )

    if last_anchor < first_anchor:
        raise ValueError(
            "Not enough rows for history + forecast horizon."
        )

    anchors = np.arange(
        first_anchor,
        last_anchor + 1,
        dtype=np.int64,
    )

    # First future label is used only for stratification.
    target_labels = np.asarray(
        labels[
            anchors + 1
        ],
        dtype=np.int64,
    )

    print()
    print("=" * 78)
    print("BUILDING STRATIFIED FORECASTING SPLIT")
    print("=" * 78)

    print(
        f"Available forecasting anchors : {len(anchors):,}"
    )

    # ------------------------------------------------------------------------
    # Stratified split:
    #
    # 70% train
    # 15% validation
    # 15% test
    #
    # This is intentionally used for the emergency prototype because the
    # merged dataset has no trustworthy timestamp column.
    # ------------------------------------------------------------------------

    train_parts = []
    val_parts = []
    test_parts = []

    for class_index in range(num_classes):

        class_positions = np.flatnonzero(
            target_labels == class_index
        )

        if len(class_positions) == 0:
            continue

        shuffled = class_positions.copy()

        rng.shuffle(
            shuffled
        )

        n = len(shuffled)

        n_train = int(
            n * 0.70
        )

        n_val = int(
            n * 0.15
        )

        # Make sure rare classes don't disappear
        # completely from training.
        if n >= 3:
            n_train = max(
                n_train,
                1,
            )

        if n >= 7:
            n_val = max(
                n_val,
                1,
            )

        n_test = (
            n
            - n_train
            - n_val
        )

        if n >= 3:
            n_test = max(
                n_test,
                1,
            )

        # Correct the total if necessary.
        while (
            n_train
            + n_val
            + n_test
            > n
        ):
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test -= 1

        while (
            n_train
            + n_val
            + n_test
            < n
        ):
            n_train += 1

        train_parts.append(
            shuffled[
                :n_train
            ]
        )

        val_parts.append(
            shuffled[
                n_train:n_train + n_val
            ]
        )

        test_parts.append(
            shuffled[
                n_train + n_val:
                n_train + n_val + n_test
            ]
        )

    train_positions = (
        np.concatenate(train_parts)
        if train_parts
        else np.array([], dtype=np.int64)
    )

    val_positions = (
        np.concatenate(val_parts)
        if val_parts
        else np.array([], dtype=np.int64)
    )

    test_positions = (
        np.concatenate(test_parts)
        if test_parts
        else np.array([], dtype=np.int64)
    )

    rng.shuffle(
        train_positions
    )

    rng.shuffle(
        val_positions
    )

    rng.shuffle(
        test_positions
    )

    # ------------------------------------------------------------------------
    # Limit size for RTX 3050 / emergency training.
    # ------------------------------------------------------------------------

    if len(train_positions) > max_train_samples:
        train_positions = rng.choice(
            train_positions,
            size=max_train_samples,
            replace=False,
        )

    if len(val_positions) > max_val_samples:
        val_positions = rng.choice(
            val_positions,
            size=max_val_samples,
            replace=False,
        )

    if len(test_positions) > max_test_samples:
        test_positions = rng.choice(
            test_positions,
            size=max_test_samples,
            replace=False,
        )

    rng.shuffle(
        train_positions
    )

    rng.shuffle(
        val_positions
    )

    rng.shuffle(
        test_positions
    )

    train_anchors = anchors[
        train_positions
    ]

    val_anchors = anchors[
        val_positions
    ]

    test_anchors = anchors[
        test_positions
    ]

    print()
    print(
        f"Train anchors      : {len(train_anchors):,}"
    )

    print(
        f"Validation anchors : {len(val_anchors):,}"
    )

    print(
        f"Test anchors       : {len(test_anchors):,}"
    )

    return (
        train_anchors,
        val_anchors,
        test_anchors,
    )


# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================

def print_class_distribution(
    labels,
    anchors,
    class_names,
    class_weights,
):

    num_classes = len(
        class_names
    )

    target_labels = np.asarray(
        labels[
            anchors + 1
        ],
        dtype=np.int64,
    )

    print()
    print("=" * 78)
    print("TRAINING CLASS DISTRIBUTION / WEIGHTS")
    print("=" * 78)

    for class_index in range(
        num_classes
    ):

        count = int(
            np.sum(
                target_labels
                == class_index
            )
        )

        weight = float(
            class_weights[
                class_index
            ].item()
        )

        print(
            f"{class_index:02d} | "
            f"{class_names[class_index]:35s} | "
            f"count={count:10,d} | "
            f"weight={weight:.4f}"
        )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model,
    loader,
    horizon,
    num_classes,
    benign_index,
):

    from forecasting_model import (
        _classification_metrics
    )

    all_true = [
        []
        for _ in range(horizon)
    ]

    all_pred = [
        []
        for _ in range(horizon)
    ]

    all_attack_probability = [
        []
        for _ in range(horizon)
    ]

    all_uncertainty = [
        []
        for _ in range(horizon)
    ]

    model.eval()

    with torch.no_grad():

        for (
            x,
            target_states,
            target_labels,
        ) in loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            target_states = target_states.to(
                DEVICE,
                non_blocking=True,
            )

            target_labels = target_labels.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(
                x,
                horizon,
            )

            probabilities = torch.softmax(
                outputs["attack_logits"],
                dim=-1,
            )

            predictions = (
                outputs["attack_logits"]
                .argmax(dim=-1)
            )

            attack_probability = (
                1.0
                - probabilities[
                    ...,
                    benign_index,
                ]
            )

            uncertainty = (
                outputs["uncertainty"]
                .squeeze(-1)
            )

            labels_np = (
                target_labels
                .cpu()
                .numpy()
            )

            predictions_np = (
                predictions
                .cpu()
                .numpy()
            )

            attack_probability_np = (
                attack_probability
                .cpu()
                .numpy()
            )

            uncertainty_np = (
                uncertainty
                .cpu()
                .numpy()
            )

            for step in range(
                horizon
            ):

                all_true[step].append(
                    labels_np[:, step]
                )

                all_pred[step].append(
                    predictions_np[:, step]
                )

                all_attack_probability[
                    step
                ].append(
                    attack_probability_np[
                        :, step
                    ]
                )

                all_uncertainty[
                    step
                ].append(
                    uncertainty_np[
                        :, step
                    ]
                )

    per_step = []

    for step in range(
        horizon
    ):

        y_true = np.concatenate(
            all_true[step]
        )

        y_pred = np.concatenate(
            all_pred[step]
        )

        metrics = _classification_metrics(
            y_true,
            y_pred,
            num_classes,
            benign_index,
        )

        metrics[
            "mean_attack_probability"
        ] = float(
            np.mean(
                np.concatenate(
                    all_attack_probability[
                        step
                    ]
                )
            )
        )

        metrics[
            "mean_uncertainty"
        ] = float(
            np.mean(
                np.concatenate(
                    all_uncertainty[
                        step
                    ]
                )
            )
        )

        per_step.append(
            metrics
        )

    metric_names = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "attack_type_accuracy",
        "attack_type_macro_f1",
        "mean_attack_probability",
        "mean_uncertainty",
    ]

    overall = {
        name: float(
            np.mean(
                [
                    step[name]
                    for step in per_step
                ]
            )
        )
        for name in metric_names
    }

    return (
        per_step,
        overall,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    config = load_yaml(
        CONFIG_PATH
    )

    validate_config(
        config
    )

    metadata = load_json(
        METADATA_PATH
    )

    sequence_length = int(
        metadata["forecasting"].get(
            "history_length",
            config["temporal"]["sequence_length"],
        )
    )

    horizon = int(
        metadata["forecasting"].get(
            "forecast_horizon",
            config["temporal"]["forecast_horizon"],
        )
    )

    latent_dimension = int(
        metadata["lstm"]["latent_dimension"]
    )

    num_classes = int(
        metadata["dataset"]["classes"]
    )

    class_names = list(
        metadata["dataset"]["class_names"]
    )

    benign_index = get_benign_index(
        class_names
    )

    # ------------------------------------------------------------------------
    # File checks
    # ------------------------------------------------------------------------

    for path, name in [
        (
            LATENT_PATH,
            "Latent",
        ),
        (
            LABEL_PATH,
            "Label",
        ),
        (
            METADATA_PATH,
            "Metadata",
        ),
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

    # ------------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("NETSIGHT AI - FORECASTING PROTOTYPE")
    print("=" * 78)

    print(
        f"Device             : {DEVICE}"
    )

    print(
        f"Latent file        : {LATENT_PATH}"
    )

    print(
        f"Label file         : {LABEL_PATH}"
    )

    print(
        f"Latent dimension   : {latent_dimension}"
    )

    print(
        f"Sequence length    : {sequence_length}"
    )

    print(
        f"Forecast horizon   : {horizon}"
    )

    print(
        f"Classes            : {num_classes}"
    )

    print(
        f"BENIGN index       : {benign_index}"
    )

    # ------------------------------------------------------------------------
    # Load mmap arrays
    # ------------------------------------------------------------------------

    print()
    print(
        "Loading memory-mapped latent data..."
    )

    latents = np.load(
        LATENT_PATH,
        mmap_mode="r",
    )

    labels = np.load(
        LABEL_PATH,
        mmap_mode="r",
    )

    if latents.ndim != 2:
        raise ValueError(
            f"Latent array must be 2-D. Got {latents.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Label array must be 1-D. Got {labels.shape}"
        )

    if latents.shape[1] != latent_dimension:
        raise ValueError(
            "Latent dimension mismatch."
        )

    # ------------------------------------------------------------------------
    # Align arrays
    # ------------------------------------------------------------------------

    if len(latents) != len(labels):

        print()
        print(
            "WARNING: latent and label row counts differ."
        )

        print(
            f"Latent rows: {len(latents):,}"
        )

        print(
            f"Label rows : {len(labels):,}"
        )

        usable_rows = min(
            len(latents),
            len(labels),
        )

        print(
            f"Using first {usable_rows:,} aligned rows."
        )

        latents = latents[
            :usable_rows
        ]

        labels = labels[
            :usable_rows
        ]

    num_rows = len(
        latents
    )

    print()
    print(
        f"Usable latent rows : {num_rows:,}"
    )

    # ------------------------------------------------------------------------
    # Emergency training limits
    # ------------------------------------------------------------------------

    max_train_samples = 400000
    max_val_samples = 60000
    max_test_samples = 60000

    # ------------------------------------------------------------------------
    # Build stratified splits
    # ------------------------------------------------------------------------

    (
        train_anchors,
        val_anchors,
        test_anchors,
    ) = build_stratified_splits(
        labels=labels,
        sequence_length=sequence_length,
        horizon=horizon,
        num_classes=num_classes,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        seed=42,
    )

    if (
        len(train_anchors) == 0
        or len(val_anchors) == 0
        or len(test_anchors) == 0
    ):
        raise ValueError(
            "One or more forecasting splits are empty."
        )

    # ------------------------------------------------------------------------
    # Dataset objects
    # ------------------------------------------------------------------------

    train_dataset = ForecastDataset(
        latents=latents,
        labels=labels,
        anchors=train_anchors,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    val_dataset = ForecastDataset(
        latents=latents,
        labels=labels,
        anchors=val_anchors,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    test_dataset = ForecastDataset(
        latents=latents,
        labels=labels,
        anchors=test_anchors,
        sequence_length=sequence_length,
        horizon=horizon,
    )

    # ------------------------------------------------------------------------
    # Check training classes
    # ------------------------------------------------------------------------

    train_target_labels = np.array(
        labels[
            train_anchors + 1
        ],
        dtype=np.int64,
        copy=True,
    )

    config["_num_classes"] = num_classes
    config["_horizon"] = horizon

    class_weights = calculate_class_weights(
        train_target_labels,
        config,
    )

    print_class_distribution(
        labels=labels,
        anchors=train_anchors,
        class_names=class_names,
        class_weights=class_weights,
    )

    # ------------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------------

    training_cfg = config[
        "training"
    ]

    batch_size = int(
        training_cfg[
            "batch_size"
        ]
    )

    num_workers = int(
        training_cfg.get(
            "num_workers",
            0,
        )
    )

    pin_memory = (
        DEVICE.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("BUILDING FORECASTING MODEL")
    print("=" * 78)

    model = ForecastingModel(
        latent_dim=latent_dimension,
        num_classes=num_classes,
        config=config,
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters         : {parameter_count:,}"
    )

    # ------------------------------------------------------------------------
    # Training scope
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING SCOPE")
    print("=" * 78)

    print(
        "LSTM training       : NOT PERFORMED"
    )

    print(
        "LSTM latents        : FROZEN INPUTS"
    )

    print(
        "Transformer         : TRAINED"
    )

    print(
        "GRU                 : TRAINED"
    )

    print(
        "TCN                 : TRAINED"
    )

    print(
        "Ensemble fusion     : TRAINED"
    )

    print(
        "Forecast head       : TRAINED"
    )

    print(
        "Backprop through LSTM: NOT PERFORMED"
    )

    print()
    print("=" * 78)
    print("PROTOTYPE SPLIT NOTE")
    print("=" * 78)

    print(
        "The merged dataset has no explicit timestamp column."
    )

    print(
        "For this emergency prototype, forecasting anchors are"
    )

    print(
        "stratified to preserve attack-class coverage."
    )

    print(
        "Forecast horizon represents future sequential records,"
    )

    print(
        "not guaranteed real-world minutes."
    )

    # ------------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------------

    checkpoint_cfg = config.setdefault(
        "checkpoint",
        {},
    )

    checkpoint_directory = checkpoint_cfg.get(
        "directory",
        "checkpoints",
    )

    if not os.path.isabs(
        checkpoint_directory
    ):
        checkpoint_directory = os.path.join(
            SCRIPT_DIR,
            checkpoint_directory,
        )

    checkpoint_cfg[
        "directory"
    ] = checkpoint_directory

    os.makedirs(
        checkpoint_directory,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config,
        benign_index=benign_index,
    )

    # ------------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------------

    best_filename = checkpoint_cfg.get(
        "best_filename",
        "best_forecasting_model.pt",
    )

    best_checkpoint_path = os.path.join(
        checkpoint_directory,
        best_filename,
    )

    if not os.path.exists(
        best_checkpoint_path
    ):
        raise FileNotFoundError(
            "Best forecasting checkpoint was not created:\n"
            f"{best_checkpoint_path}"
        )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    trained_model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    trained_model.eval()

    # ------------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL TEST EVALUATION")
    print("=" * 78)

    (
        test_per_step,
        test_overall,
    ) = evaluate_model(
        model=trained_model,
        loader=test_loader,
        horizon=horizon,
        num_classes=num_classes,
        benign_index=benign_index,
    )

    for index, metrics in enumerate(
        test_per_step,
        start=1,
    ):

        print()
        print(
            f"Forecast step {index}"
        )

        print(
            f"  Accuracy             : {metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro F1             : {metrics['macro_f1']:.4f}"
        )

        print(
            f"  Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Attack Precision     : {metrics['attack_precision']:.4f}"
        )

        print(
            f"  Attack Recall        : {metrics['attack_recall']:.4f}"
        )

        print(
            f"  Attack F1            : {metrics['attack_f1']:.4f}"
        )

        print(
            f"  Attack Type Accuracy : {metrics['attack_type_accuracy']:.4f}"
        )

        print(
            f"  Attack Type Macro F1 : {metrics['attack_type_macro_f1']:.4f}"
        )

    print()
    print("-" * 78)
    print("OVERALL TEST MEAN")
    print("-" * 78)

    for key, value in test_overall.items():

        print(
            f"{key:25s}: {value:.4f}"
        )

    # ------------------------------------------------------------------------
    # Save evaluation
    # ------------------------------------------------------------------------

    evaluation = {
        "project": "NetSight AI",
        "module": "Forecasting",
        "evaluation": "test",
        "split_method": "stratified prototype split",
        "device": str(DEVICE),
        "latent_dimension": latent_dimension,
        "sequence_length": sequence_length,
        "forecast_horizon": horizon,
        "num_classes": num_classes,
        "class_names": class_names,
        "benign_index": benign_index,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "per_step": test_per_step,
        "overall_mean": test_overall,
        "checkpoint": best_checkpoint_path,
    }

    evaluation_path = os.path.join(
        checkpoint_directory,
        "forecasting_evaluation.json",
    )

    with open(
        evaluation_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------------

    forecasting_metadata = {
        "project": {
            "name": "NetSight AI",
            "module": "Forecasting",
        },

        "data": {
            "latent_file": LATENT_PATH,
            "label_file": LABEL_PATH,
            "latent_rows_used": int(
                num_rows
            ),
            "latent_dimension": int(
                latent_dimension
            ),
        },

        "forecasting": {
            "sequence_length": int(
                sequence_length
            ),
            "forecast_horizon": int(
                horizon
            ),
            "num_classes": int(
                num_classes
            ),
            "class_names": class_names,
            "benign_index": int(
                benign_index
            ),
            "temporal_interval": None,
        },

        "split": {
            "method": (
                "stratified prototype split because "
                "the merged dataset has no explicit timestamp"
            ),
            "train_samples": int(
                len(train_dataset)
            ),
            "validation_samples": int(
                len(val_dataset)
            ),
            "test_samples": int(
                len(test_dataset)
            ),
        },

        "lstm": {
            "frozen": True,
            "training_performed": False,
            "backpropagation_performed": False,
        },

        "forecasting_models": [
            "Transformer World Model",
            "GRU Dynamics Model",
            "TCN Dynamics Model",
            "Ensemble Fusion",
        ],

        "checkpoint": {
            "path": best_checkpoint_path,
            "selection_metric": (
                "0.7 * validation_macro_f1 + "
                "0.3 * validation_attack_f1"
            ),
            "best_epoch": checkpoint.get(
                "best_epoch",
                None,
            ),
            "best_validation_macro_f1": checkpoint.get(
                "best_val_macro_f1",
                None,
            ),
            "best_validation_attack_f1": checkpoint.get(
                "best_val_attack_f1",
                None,
            ),
        },

        "scientific_note": (
            "The supplied merged dataset contains no explicit "
            "timestamp column. Therefore this emergency prototype "
            "uses stratified forecasting anchors to preserve "
            "attack-class coverage. The forecast horizon represents "
            "future sequential records rather than guaranteed "
            "real-world minutes. A production deployment should "
            "replace this with timestamped temporal windows."
        ),
    }

    metadata_output_path = os.path.join(
        checkpoint_directory,
        "forecasting_metadata.json",
    )

    with open(
        metadata_output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            forecasting_metadata,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FORECASTING TRAINING COMPLETE")
    print("=" * 78)

    print(
        f"Best checkpoint : {best_checkpoint_path}"
    )

    print(
        f"Evaluation      : {evaluation_path}"
    )

    print(
        f"Metadata        : {metadata_output_path}"
    )

    print()
    print(
        "LSTM TRAINING        : NOT PERFORMED"
    )

    print(
        "LSTM LATENTS         : FROZEN"
    )

    print(
        "FORECASTING MODELS   : TRAINED"
    )

    print(
        "BACKPROP THROUGH LSTM: NOT PERFORMED"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()