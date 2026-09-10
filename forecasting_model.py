import os
import json
from typing import Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForecastDataset(Dataset):
    """Tensor-backed forecasting dataset."""

    def __init__(self, latent_sequences, future_latents, future_labels):
        if len(latent_sequences) != len(future_latents) or len(latent_sequences) != len(future_labels):
            raise ValueError("Input, future latent, and future label counts must match.")

        self.x = torch.as_tensor(latent_sequences, dtype=torch.float32)
        self.y_state = torch.as_tensor(future_latents, dtype=torch.float32)
        self.y_attack = torch.as_tensor(future_labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y_state[index], self.y_attack[index]


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be a positive odd integer.")

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def _trim(self, x, target_length):
        return x[..., :target_length]

    def forward(self, x):
        residual = x
        target_length = x.size(-1)

        y = self._trim(self.conv1(x), target_length)
        y = self.dropout(F.gelu(y))

        y = self._trim(self.conv2(y), target_length)
        y = self.dropout(F.gelu(y))

        y = y + residual
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return y


class TransformerWorldModel(nn.Module):
    def __init__(self, latent_dim, heads, layers, ff_multiplier, dropout):
        super().__init__()

        if latent_dim % heads != 0:
            raise ValueError(
                f"Latent dimension {latent_dim} must be divisible by attention heads {heads}."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(latent_dim),
        )

        self.transition = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.transition(h[:, -1])


class GRUDynamics(nn.Module):
    def __init__(self, latent_dim, layers, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        self.transition = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return self.transition(output[:, -1])


class TCNDynamics(nn.Module):
    def __init__(self, latent_dim, levels, kernel_size, dropout):
        super().__init__()

        self.blocks = nn.ModuleList([
            TemporalBlock(
                channels=latent_dim,
                kernel_size=kernel_size,
                dilation=2 ** level,
                dropout=dropout,
            )
            for level in range(levels)
        ])

        self.transition = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        y = x.transpose(1, 2)

        for block in self.blocks:
            y = block(y)

        return self.transition(y[:, :, -1])


class ForecastEnsemble(nn.Module):
    def __init__(self, latent_dim, config):
        super().__init__()

        model_cfg = config["model"]
        dropout = float(model_cfg["dropout"])

        transformer_cfg = model_cfg["transformer"]
        gru_cfg = model_cfg["gru"]
        tcn_cfg = model_cfg["tcn"]

        self.transformer = TransformerWorldModel(
            latent_dim=latent_dim,
            heads=int(transformer_cfg["heads"]),
            layers=int(transformer_cfg["layers"]),
            ff_multiplier=int(transformer_cfg["ff_multiplier"]),
            dropout=dropout,
        )

        self.gru = GRUDynamics(
            latent_dim=latent_dim,
            layers=int(gru_cfg["layers"]),
            dropout=dropout,
        )

        self.tcn = TCNDynamics(
            latent_dim=latent_dim,
            levels=int(tcn_cfg["levels"]),
            kernel_size=int(tcn_cfg["kernel_size"]),
            dropout=dropout,
        )

        fusion_hidden = int(
            model_cfg.get("fusion_hidden_dim", latent_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, latent_dim),
        )

    def forward(self, x):
        transformer_state = self.transformer(x)
        gru_state = self.gru(x)
        tcn_state = self.tcn(x)

        model_states = torch.stack(
            [transformer_state, gru_state, tcn_state],
            dim=0,
        )

        combined = torch.cat(
            [transformer_state, gru_state, tcn_state],
            dim=-1,
        )

        fused_state = self.fusion(combined)

        # Ensemble disagreement is used as an uncertainty proxy.
        uncertainty = (
            model_states.var(dim=0, unbiased=False)
            .mean(dim=-1, keepdim=True)
        )

        return {
            "transformer": transformer_state,
            "gru": gru_state,
            "tcn": tcn_state,
            "fused": fused_state,
            "uncertainty": uncertainty,
        }


class AttackForecastHead(nn.Module):
    def __init__(self, latent_dim, num_classes, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, state):
        return self.classifier(self.network(state))


class ForecastingModel(nn.Module):
    """
    Forecasting module only.

    Input:
        Frozen LSTM latent sequence [B, history, latent_dim]

    Trained here:
        Transformer + GRU + TCN + ensemble fusion + attack forecast head

    Not trained here:
        Upstream LSTM
    """

    def __init__(
        self,
        latent_dim=None,
        num_classes=None,
        config=None,
        latent_dimension=None,
    ):
        super().__init__()

        if latent_dim is None:
            latent_dim = latent_dimension

        if latent_dim is None or num_classes is None or config is None:
            raise ValueError(
                "latent_dim, num_classes and config are required."
            )

        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)

        self.ensemble = ForecastEnsemble(
            self.latent_dim,
            config,
        )

        self.attack_head = AttackForecastHead(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            dropout=float(config["model"]["dropout"]),
        )

    def forward_one_step(self, sequence):
        result = self.ensemble(sequence)

        fused_state = result["fused"]

        attack_logits = self.attack_head(
            fused_state
        )

        return {
            "state": fused_state,
            "attack_logits": attack_logits,
            "uncertainty": result["uncertainty"],
        }

    def forward(self, sequence, horizon):
        horizon = int(horizon)

        if horizon < 1:
            raise ValueError("Forecast horizon must be >= 1.")

        if sequence.ndim != 3:
            raise ValueError(
                "Expected sequence shape [batch, history, latent_dim]."
            )

        if sequence.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"received {sequence.size(-1)}."
            )

        current_sequence = sequence

        predicted_states = []
        predicted_logits = []
        uncertainties = []

        for _ in range(horizon):
            result = self.forward_one_step(
                current_sequence
            )

            next_state = result["state"]

            predicted_states.append(next_state)
            predicted_logits.append(
                result["attack_logits"]
            )
            uncertainties.append(
                result["uncertainty"]
            )

            current_sequence = torch.cat(
                [
                    current_sequence[:, 1:],
                    next_state.unsqueeze(1),
                ],
                dim=1,
            )

        return {
            "states": torch.stack(
                predicted_states,
                dim=1,
            ),
            "attack_logits": torch.stack(
                predicted_logits,
                dim=1,
            ),
            "uncertainty": torch.stack(
                uncertainties,
                dim=1,
            ),
        }


def calculate_class_weights(labels, config=None):
    """
    Imbalance-aware class weighting.

    Default:
        sqrt inverse-frequency
        -> normalize mean to 1
        -> clip to configured limits

    This is deliberately softer than raw inverse-frequency weighting,
    which can become unstable for extremely rare classes.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    labels = labels[labels >= 0]

    if labels.size == 0:
        raise ValueError(
            "Cannot calculate class weights from empty labels."
        )

    if config is not None:
        num_classes = int(
            config.get(
                "_num_classes",
                labels.max() + 1,
            )
        )
        weighting_cfg = config.get(
            "loss",
            {}
        ).get(
            "class_weighting",
            {}
        )
    else:
        num_classes = int(
            labels.max() + 1
        )
        weighting_cfg = {}

    mode = str(
        weighting_cfg.get(
            "mode",
            "sqrt_inverse_frequency",
        )
    ).lower()

    max_weight = float(
        weighting_cfg.get(
            "max_weight",
            6.0,
        )
    )

    min_weight = float(
        weighting_cfg.get(
            "min_weight",
            0.25,
        )
    )

    counts = np.bincount(
        labels,
        minlength=num_classes,
    ).astype(np.float64)

    weights = np.ones(
        num_classes,
        dtype=np.float64,
    )

    present = counts > 0

    if mode == "inverse_frequency":
        weights[present] = 1.0 / counts[present]

    elif mode == "sqrt_inverse_frequency":
        weights[present] = 1.0 / np.sqrt(
            counts[present]
        )

    elif mode == "effective_number":
        beta = float(
            weighting_cfg.get(
                "beta",
                0.9999,
            )
        )

        beta = min(
            max(beta, 0.0),
            0.999999999,
        )

        weights[present] = (
            (1.0 - beta)
            /
            (
                1.0
                -
                np.power(
                    beta,
                    counts[present],
                )
            )
        )

    elif mode == "none":
        return torch.ones(
            num_classes,
            dtype=torch.float32,
        )

    else:
        raise ValueError(
            "Unsupported class weighting mode: "
            f"{mode}"
        )

    if present.any():
        weights[present] /= weights[present].mean()

    weights = np.clip(
        weights,
        min_weight,
        max_weight,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def calculate_loss(
    outputs,
    target_states,
    target_labels,
    class_weights,
    config,
    benign_index,
):
    predicted_states = outputs["states"]
    predicted_logits = outputs["attack_logits"]

    class_weights = class_weights.to(
        predicted_logits.device,
        dtype=torch.float32,
    )

    state_loss = F.smooth_l1_loss(
        predicted_states,
        target_states,
    )

    if target_states.size(1) > 1:
        target_delta = (
            target_states[:, 1:]
            -
            target_states[:, :-1]
        )

        predicted_delta = (
            predicted_states[:, 1:]
            -
            predicted_states[:, :-1]
        )

        delta_loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
        )
    else:
        delta_loss = torch.zeros(
            (),
            device=target_states.device,
        )

    flat_logits = predicted_logits.reshape(
        -1,
        predicted_logits.size(-1),
    )

    flat_labels = target_labels.reshape(
        -1
    )

    label_smoothing = float(
        config.get(
            "loss",
            {}
        ).get(
            "label_smoothing",
            0.02,
        )
    )

    attack_type_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    # Separate binary attack-vs-benign objective.
    #
    # IMPORTANT:
    # Do not use binary_cross_entropy() here because this training step
    # runs under CUDA autocast. Instead construct an attack logit from
    # the multiclass logits and use BCE-with-logits, which is autocast-safe.
    benign_logit = predicted_logits[
        ..., benign_index
    ]

    attack_indices = [
        index
        for index in range(
            predicted_logits.size(-1)
        )
        if index != benign_index
    ]

    attack_logits_only = predicted_logits[
        ...,
        attack_indices,
    ]

    attack_logit = (
        torch.logsumexp(
            attack_logits_only,
            dim=-1,
        )
        -
        benign_logit
    )

    attack_target = (
        target_labels != benign_index
    ).float()

    attack_occurrence_loss = F.binary_cross_entropy_with_logits(
        attack_logit,
        attack_target,
    )

    training_cfg = config["training"]

    state_weight = float(
        training_cfg.get(
            "state_loss_weight",
            training_cfg.get(
                "state",
                1.0,
            ),
        )
    )

    delta_weight = float(
        training_cfg.get(
            "delta_loss_weight",
            training_cfg.get(
                "delta",
                0.25,
            ),
        )
    )

    attack_type_weight = float(
        training_cfg.get(
            "attack_type_loss_weight",
            training_cfg.get(
                "attack",
                1.0,
            ),
        )
    )

    attack_occurrence_weight = float(
        training_cfg.get(
            "attack_occurrence_loss_weight",
            0.5,
        )
    )

    total_loss = (
        state_weight * state_loss
        +
        delta_weight * delta_loss
        +
        attack_type_weight * attack_type_loss
        +
        attack_occurrence_weight
        * attack_occurrence_loss
    )

    return {
        "total": total_loss,
        "state": state_loss,
        "delta": delta_loss,
        "attack_type": attack_type_loss,
        "attack_occurrence": attack_occurrence_loss,
    }


def _classification_metrics(
    y_true,
    y_pred,
    num_classes,
    benign_index,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    valid = (
        (y_true >= 0)
        &
        (y_true < num_classes)
        &
        (y_pred >= 0)
        &
        (y_pred < num_classes)
    )

    np.add.at(
        confusion,
        (
            y_true[valid],
            y_pred[valid],
        ),
        1,
    )

    total = int(
        confusion.sum()
    )

    accuracy = (
        float(
            np.trace(confusion)
            /
            total
        )
        if total
        else 0.0
    )

    tp = np.diag(
        confusion
    ).astype(np.float64)

    support = confusion.sum(
        axis=1
    ).astype(np.float64)

    predicted = confusion.sum(
        axis=0
    ).astype(np.float64)

    precision = np.divide(
        tp,
        predicted,
        out=np.zeros_like(tp),
        where=predicted > 0,
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros_like(tp),
        where=support > 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    active = (
        (support + predicted) > 0
    )

    macro_f1 = (
        float(f1[active].mean())
        if active.any()
        else 0.0
    )

    balanced_accuracy = (
        float(
            recall[support > 0].mean()
        )
        if np.any(support > 0)
        else 0.0
    )

    true_attack = (
        y_true != benign_index
    )

    pred_attack = (
        y_pred != benign_index
    )

    tp_attack = int(
        np.sum(
            true_attack
            &
            pred_attack
        )
    )

    fp_attack = int(
        np.sum(
            ~true_attack
            &
            pred_attack
        )
    )

    fn_attack = int(
        np.sum(
            true_attack
            &
            ~pred_attack
        )
    )

    attack_precision = (
        tp_attack
        /
        (tp_attack + fp_attack)
        if tp_attack + fp_attack
        else 0.0
    )

    attack_recall = (
        tp_attack
        /
        (tp_attack + fn_attack)
        if tp_attack + fn_attack
        else 0.0
    )

    attack_f1 = (
        2.0
        *
        attack_precision
        *
        attack_recall
        /
        (
            attack_precision
            +
            attack_recall
        )
        if attack_precision + attack_recall
        else 0.0
    )

    attack_mask = true_attack

    attack_type_accuracy = (
        float(
            np.mean(
                y_pred[attack_mask]
                ==
                y_true[attack_mask]
            )
        )
        if np.any(attack_mask)
        else 0.0
    )

    attack_class_f1 = []

    for cls in range(num_classes):
        if cls == benign_index:
            continue

        class_true = (
            y_true == cls
        )

        class_pred = (
            y_pred == cls
        )

        tp_cls = int(
            np.sum(
                class_true
                &
                class_pred
            )
        )

        fp_cls = int(
            np.sum(
                ~class_true
                &
                class_pred
                &
                true_attack
            )
        )

        fn_cls = int(
            np.sum(
                class_true
                &
                ~class_pred
            )
        )

        p = (
            tp_cls
            /
            (tp_cls + fp_cls)
            if tp_cls + fp_cls
            else 0.0
        )

        r = (
            tp_cls
            /
            (tp_cls + fn_cls)
            if tp_cls + fn_cls
            else 0.0
        )

        f = (
            2.0 * p * r / (p + r)
            if p + r
            else 0.0
        )

        if (
            np.sum(class_true) > 0
            or
            np.sum(class_pred) > 0
        ):
            attack_class_f1.append(f)

    attack_type_macro_f1 = (
        float(
            np.mean(
                attack_class_f1
            )
        )
        if attack_class_f1
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "attack_type_accuracy": attack_type_accuracy,
        "attack_type_macro_f1": attack_type_macro_f1,
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": recall.tolist(),
    }


def _run_validation(
    model,
    loader,
    class_weights,
    config,
    benign_index,
):
    model.eval()

    totals = {
        "total": 0.0,
        "state": 0.0,
        "delta": 0.0,
        "attack_type": 0.0,
        "attack_occurrence": 0.0,
    }

    horizon = int(
        config["_horizon"]
    )

    all_true = [
        []
        for _ in range(horizon)
    ]

    all_pred = [
        []
        for _ in range(horizon)
    ]

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
                target_states.size(1),
            )

            losses = calculate_loss(
                outputs,
                target_states,
                target_labels,
                class_weights,
                config,
                benign_index,
            )

            for key in totals:
                totals[key] += float(
                    losses[key].item()
                )

            predictions = (
                outputs["attack_logits"]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

            labels_np = (
                target_labels
                .detach()
                .cpu()
                .numpy()
            )

            for step in range(
                target_labels.size(1)
            ):
                all_true[step].append(
                    labels_np[:, step]
                )

                all_pred[step].append(
                    predictions[:, step]
                )

    n_batches = max(
        len(loader),
        1,
    )

    avg_losses = {
        key: value / n_batches
        for key, value in totals.items()
    }

    per_step = []

    for step in range(horizon):
        true_step = (
            np.concatenate(
                all_true[step]
            )
            if all_true[step]
            else np.empty(
                0,
                dtype=np.int64,
            )
        )

        pred_step = (
            np.concatenate(
                all_pred[step]
            )
            if all_pred[step]
            else np.empty(
                0,
                dtype=np.int64,
            )
        )

        per_step.append(
            _classification_metrics(
                true_step,
                pred_step,
                int(config["_num_classes"]),
                benign_index,
            )
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
    ]

    mean_metrics = {
        name: float(
            np.mean(
                [
                    item[name]
                    for item in per_step
                ]
            )
        )
        for name in metric_names
    }

    return (
        avg_losses,
        per_step,
        mean_metrics,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    config,
    benign_index=0,
):
    """
    Train forecasting models only.

    Checkpoint selection is based on a validation score that prioritizes:
        70% multiclass macro-F1
        30% attack occurrence F1

    This prevents the dominant BENIGN class from determining the
    best checkpoint through accuracy or raw loss alone.
    """

    model.to(DEVICE)

    training_cfg = config["training"]

    epochs = int(
        training_cfg["epochs"]
    )

    patience = int(
        training_cfg.get(
            "early_stopping_patience",
            training_cfg.get(
                "patience",
                5,
            ),
        )
    )

    learning_rate = float(
        training_cfg["learning_rate"]
    )

    weight_decay = float(
        training_cfg["weight_decay"]
    )

    gradient_clip = float(
        training_cfg.get(
            "gradient_clip",
            1.0,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(
            training_cfg.get(
                "scheduler_factor",
                0.5,
            )
        ),
        patience=int(
            training_cfg.get(
                "scheduler_patience",
                2,
            )
        ),
        min_lr=float(
            training_cfg.get(
                "min_learning_rate",
                1e-6,
            )
        ),
    )

    use_amp = (
        bool(
            training_cfg.get(
                "mixed_precision",
                True,
            )
        )
        and
        DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_cfg = config.get(
        "checkpoint",
        {}
    )

    checkpoint_dir = checkpoint_cfg.get(
        "directory",
        "checkpoints",
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    best_path = os.path.join(
        checkpoint_dir,
        checkpoint_cfg.get(
            "best_filename",
            "best_forecasting_model.pt",
        ),
    )

    best_score = -float("inf")
    best_macro_f1 = 0.0
    best_attack_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    class_weights = class_weights.to(
        DEVICE,
        dtype=torch.float32,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        train_totals = {
            "total": 0.0,
            "state": 0.0,
            "delta": 0.0,
            "attack_type": 0.0,
            "attack_occurrence": 0.0,
        }

        for (
            x,
            target_states,
            target_labels,
        ) in train_loader:

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

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(
                    x,
                    target_states.size(1),
                )

                losses = calculate_loss(
                    outputs,
                    target_states,
                    target_labels,
                    class_weights,
                    config,
                    benign_index,
                )

            scaler.scale(
                losses["total"]
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            for key in train_totals:
                train_totals[key] += float(
                    losses[key].detach().item()
                )

        n_train = max(
            len(train_loader),
            1,
        )

        train_avg = {
            key: value / n_train
            for key, value in train_totals.items()
        }

        (
            val_avg,
            val_per_step,
            val_mean,
        ) = _run_validation(
            model,
            val_loader,
            class_weights,
            config,
            benign_index,
        )

        selection_score = (
            0.7 * val_mean["macro_f1"]
            +
            0.3 * val_mean["attack_f1"]
        )

        scheduler.step(
            selection_score
        )

        current_lr = optimizer.param_groups[0]["lr"]

        min_improvement = float(
            training_cfg.get(
                "min_improvement",
                1e-4,
            )
        )

        improved = (
            selection_score
            >
            best_score + min_improvement
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_avg,
            "validation_loss": val_avg,
            "validation_metrics": val_mean,
            "validation_per_step": val_per_step,
            "selection_score": float(
                selection_score
            ),
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_avg['total']:.6f} | "
            f"Val {val_avg['total']:.6f} | "
            f"Macro-F1 {val_mean['macro_f1']:.4f} | "
            f"Attack F1 {val_mean['attack_f1']:.4f} | "
            f"Balanced Acc {val_mean['balanced_accuracy']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        if improved:
            best_score = selection_score
            best_macro_f1 = val_mean["macro_f1"]
            best_attack_f1 = val_mean["attack_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_dim": model.latent_dim,
                    "num_classes": model.num_classes,
                    "best_selection_score": float(best_score),
                    "best_val_macro_f1": float(best_macro_f1),
                    "best_val_attack_f1": float(best_attack_f1),
                    "best_epoch": int(best_epoch),
                    "benign_index": int(benign_index),
                    "config": config,
                    "class_weights": class_weights.detach().cpu(),
                    "training_history": history,
                },
                best_path,
            )

            print(
                f"  ✓ Best model saved: {best_path}"
            )

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    "Early stopping: validation "
                    "selection score did not improve."
                )
                break

    print(
        f"Best epoch: {best_epoch} | "
        f"Val Macro-F1: {best_macro_f1:.4f} | "
        f"Val Attack F1: {best_attack_f1:.4f}"
    )

    return model


class ForecastPredictor:
    def __init__(
        self,
        model,
        class_names,
        benign_index=None,
    ):
        self.model = model
        self.class_names = list(
            class_names
        )

        if benign_index is None:
            matches = [
                i
                for i, name in enumerate(
                    self.class_names
                )
                if str(name).strip().lower() == "benign"
            ]

            if not matches:
                raise ValueError(
                    "BENIGN class was not found in class metadata."
                )

            benign_index = matches[0]

        self.benign_index = int(
            benign_index
        )

        self.model.to(
            DEVICE
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        latent_sequence,
        horizon,
    ):
        tensor = torch.as_tensor(
            latent_sequence,
            dtype=torch.float32,
        )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 3:
            raise ValueError(
                "Expected latent sequence shape "
                "[batch, sequence, latent_dim]."
            )

        if tensor.size(-1) != self.model.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: "
                f"received {tensor.size(-1)}, "
                f"expected {self.model.latent_dim}."
            )

        outputs = self.model(
            tensor.to(DEVICE),
            int(horizon),
        )

        probabilities = torch.softmax(
            outputs["attack_logits"],
            dim=-1,
        )

        attack_indices = [
            i
            for i in range(
                len(self.class_names)
            )
            if i != self.benign_index
        ]

        results = []

        for batch_index in range(
            tensor.size(0)
        ):
            future = []

            for step in range(
                int(horizon)
            ):
                probs = probabilities[
                    batch_index,
                    step,
                ]

                attack_probability = (
                    1.0
                    -
                    probs[
                        self.benign_index
                    ]
                )

                attack_probs = probs[
                    attack_indices
                ]

                best_attack_position = int(
                    torch.argmax(
                        attack_probs
                    )
                )

                predicted_attack_index = (
                    attack_indices[
                        best_attack_position
                    ]
                )

                future.append(
                    {
                        "step": step + 1,
                        "attack_probability": float(
                            attack_probability.cpu()
                        ),
                        "predicted_attack_type": (
                            self.class_names[
                                predicted_attack_index
                            ]
                        ),
                        "predicted_class_probability": float(
                            probs[
                                predicted_attack_index
                            ].cpu()
                        ),
                        "benign_probability": float(
                            probs[
                                self.benign_index
                            ].cpu()
                        ),
                        "attack_type_probability": {
                            name: float(
                                probs[index].cpu()
                            )
                            for index, name in enumerate(
                                self.class_names
                            )
                        },
                        "uncertainty": float(
                            outputs["uncertainty"][
                                batch_index,
                                step,
                            ].flatten()[0].cpu()
                        ),
                    }
                )

            results.append(
                future
            )

        return results


class RLAdapter:
    """Convert forecast output to a flat RL observation."""

    def __init__(self, class_names):
        self.class_names = list(
            class_names
        )

    def convert(self, forecast_result):
        rl_state = []

        for prediction in forecast_result:
            rl_state.append(
                float(
                    prediction[
                        "attack_probability"
                    ]
                )
            )

            for class_name in self.class_names:
                rl_state.append(
                    float(
                        prediction[
                            "attack_type_probability"
                        ][class_name]
                    )
                )

            rl_state.append(
                float(
                    prediction[
                        "uncertainty"
                    ]
                )
            )

        return {
            "forecast": forecast_result,
            "rl_state": rl_state,
            "rl_state_dimension": len(
                rl_state
            ),
            "class_names": self.class_names,
        }


def load_config(path="config.yaml"):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_metadata(
    path,
    latent_dim,
    sequence_length,
    class_names,
    horizon=None,
):
    metadata = {
        "latent_dim": int(latent_dim),
        "sequence_length": int(sequence_length),
        "num_classes": len(
            class_names
        ),
        "classes": list(
            class_names
        ),
    }

    if horizon is not None:
        metadata["forecast_horizon"] = int(
            horizon
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


def load_forecasting_model(
    checkpoint_path,
    config,
    class_names=None,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = ForecastingModel(
        latent_dim=int(
            checkpoint["latent_dim"]
        ),
        num_classes=int(
            checkpoint["num_classes"]
        ),
        config=config,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(
        DEVICE
    )
    model.eval()

    return model
