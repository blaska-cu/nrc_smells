"""
baseline_conditioned_sensor_model.py

Model:
    current measurement -> shared encoder -> z_current
    baseline measurement -> shared encoder -> z_baseline

    [z_current,
     z_baseline,
     z_current - z_baseline,
     z_current * z_baseline]
                    |
                    v
               fusion MLP
                    |
                    v
              sensor-invariant embedding
                    |
                classifier


Training objective:

    L = L_classification
        + pair_loss_weight * L_temporal_cross_sensor

Cross-sensor loss:

    - different sensors
    - same label
    - same presentation/exposure, if presentation_col is supplied
    - temporal weighting:

          w(dt) = exp(-dt / tau)

    - optional hard maximum temporal separation


Expected DataFrame columns:

    sensor_index
    label
    cycle_start_timestamp

    log_resistance_0 ... log_resistance_9
    log_vapor_pressure_0 ... log_vapor_pressure_9

Optional explicit baseline columns:

    baseline_log_resistance_0 ... baseline_log_resistance_9
    baseline_log_vapor_pressure_0 ... baseline_log_vapor_pressure_9

Optional presentation/exposure column:

    presentation_id

Dependencies:

    pip install torch pandas numpy scikit-learn
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =====================================================================
# Configuration
# =====================================================================

NUM_HEATER_STEPS = 10

CURRENT_FEATURE_COLUMNS = (
    [f"log_resistance_{k}" for k in range(NUM_HEATER_STEPS)]
    + [f"log_vapor_pressure_{k}" for k in range(NUM_HEATER_STEPS)]
)

BASELINE_FEATURE_COLUMNS = (
    [
        f"baseline_log_resistance_{k}"
        for k in range(NUM_HEATER_STEPS)
    ]
    + [
        f"baseline_log_vapor_pressure_{k}"
        for k in range(NUM_HEATER_STEPS)
    ]
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Model
# =====================================================================

class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.05,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class BaselineConditionedEncoder(nn.Module):
    """
    Shared/Siamese encoder for current and baseline observations.

        z_current  = E(current)
        z_baseline = E(baseline)

    Fusion:

        [z_current,
         z_baseline,
         z_current - z_baseline,
         z_current * z_baseline]

    -> embedding
    -> classifier
    """

    def __init__(
        self,
        input_dim=20,
        encoder_hidden_dim=64,
        encoder_dim=32,
        fusion_hidden_dim=64,
        embedding_dim=16,
        num_classes=2,
        dropout=0.05,
    ):
        super().__init__()

        # Shared weights between current and baseline.
        self.encoder = MLPBlock(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_dim,
            dropout=dropout,
        )

        fusion_input_dim = 3 * encoder_dim

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_input_dim,
                fusion_hidden_dim,
            ),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_hidden_dim,
                embedding_dim,
            ),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def encode(
        self,
        current,
        baseline,
    ):
        z_current = self.encoder(current)
        z_baseline = self.encoder(baseline)

        z_delta = (
            z_current
            - z_baseline
        )

        z_product = (
            z_current
            * z_baseline
        )

        fused = torch.cat(
            [
                z_current,
                z_baseline,
                z_delta,
                #z_product,
            ],
            dim=-1,
        )

        embedding = self.fusion(fused)

        return embedding

    def forward(
        self,
        current,
        baseline,
    ):
        embedding = self.encode(
            current,
            baseline,
        )

        logits = self.classifier(
            embedding
        )

        return {
            "embedding": embedding,
            "logits": logits,
        }


# =====================================================================
# Dataset
# =====================================================================

class SensorDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df,
        baseline,
        label_encoder,
    ):
        self.current = torch.tensor(
            df[CURRENT_FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        self.baseline = torch.tensor(
            baseline[BASELINE_FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )

        self.labels = torch.tensor(
            label_encoder.transform(
                df["label"]
            ),
            dtype=torch.long,
        )

        self.sensor_indexs = torch.tensor(
            df["sensor_index"],
            dtype=torch.long,
        )

        # Store timestamps as seconds.
        timestamp_seconds = (
            df['cycle_start_timestamp'].astype("int64")
            .to_numpy()
            / 1e9
        )

        self.timestamps = torch.tensor(
            timestamp_seconds,
            dtype=torch.float32,
        )

        # Optional presentation/exposure identifier.
        if "presentation_id" in df.columns:
            self.presentation_ids = torch.tensor(
                LabelEncoder().fit_transform(
                    df["presentation_id"].astype(str)
                ),
                dtype=torch.long,
            )
        else:
            self.presentation_ids = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "current": self.current[idx],
            "baseline": self.baseline[self.sensor_indexs[idx]],
            "label": self.labels[idx],
            "sensor_index": self.sensor_indexs[idx],
            "timestamp": self.timestamps[idx],
        }

        if self.presentation_ids is not None:
            item["presentation_id"] = (
                self.presentation_ids[idx]
            )

        return item


# =====================================================================
# Stratified batch sampler
# =====================================================================

class StratifiedBatchSampler(
    torch.utils.data.Sampler
):
    """
    Approximately label-balanced batches.

    This deliberately does NOT attempt to construct cross-sensor
    pairs. Cross-sensor relationships are discovered dynamically
    inside each batch.

    If you later find that batches don't contain enough useful
    cross-sensor comparisons, this is the component to modify.
    """

    def __init__(
        self,
        labels,
        batch_size=128,
        shuffle=True,
    ):
        self.labels = np.asarray(
            labels.cpu()
        )

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classes = np.unique(
            self.labels
        )

        self.indices_by_class = {
            c: np.where(
                self.labels == c
            )[0].tolist()
            for c in self.classes
        }

    def __iter__(self):

        pools = {
            c: indices.copy()
            for c, indices
            in self.indices_by_class.items()
        }

        if self.shuffle:
            for indices in pools.values():
                random.shuffle(indices)

        batches = []

        # Round-robin over classes.
        current_batch = []

        while any(
            len(indices) > 0
            for indices in pools.values()
        ):

            available_classes = [
                c
                for c in self.classes
                if len(pools[c]) > 0
            ]

            if len(available_classes) == 0:
                break

            # Prefer a class not already dominating the batch.
            for c in available_classes:

                if len(current_batch) >= self.batch_size:
                    batches.append(
                        current_batch
                    )
                    current_batch = []

                if len(pools[c]) > 0:
                    current_batch.append(
                        pools[c].pop()
                    )

                if len(current_batch) >= self.batch_size:
                    batches.append(
                        current_batch
                    )
                    current_batch = []

        if current_batch:
            batches.append(
                current_batch
            )

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return int(
            np.ceil(
                len(self.labels)
                / self.batch_size
            )
        )


# =====================================================================
# Temporal cross-sensor loss
# =====================================================================

def temporal_cross_sensor_loss(
    embeddings,
    labels,
    sensor_indexs,
    timestamps,
    presentation_ids=None,
    tau_seconds=20.0,
    max_delta_seconds=60.0,
):
    """
    Encourage embeddings from different sensors to agree when they
    represent the same environmental state.

    Positive criteria:

        sensor_i != sensor_j

        label_i == label_j

        optionally:
            presentation_i == presentation_j

    Pair strength:

        w(dt) = exp(-dt / tau)

    Optional hard cutoff:

        dt <= max_delta_seconds

    Parameters
    ----------
    embeddings:
        [B, D]

    labels:
        [B]

    sensor_indexs:
        [B]

    timestamps:
        [B], seconds

    presentation_ids:
        [B] or None

    tau_seconds:
        Characteristic temporal similarity scale.

    max_delta_seconds:
        Ignore pairs farther apart than this.
        Set to None for no cutoff.
    """

    # Normalize so the model cannot trivially reduce the loss
    # by shrinking all embeddings toward zero.
    z = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    # Squared Euclidean distance.
    distances = torch.cdist(
        z,
        z,
        p=2,
    ).pow(2)

    # Absolute time difference.
    dt = (
        timestamps[:, None]
        - timestamps[None, :]
    ).abs()

    # Different physical sensors.
    valid = (
        sensor_indexs[:, None]
        != sensor_indexs[None, :]
    )

    # Same class/presentation.
    valid &= (
        labels[:, None]
        == labels[None, :]
    )

    # If presentation IDs are available, only compare samples
    # belonging to the same exposure/presentation.
    if presentation_ids is not None:
        valid &= (
            presentation_ids[:, None]
            == presentation_ids[None, :]
        )

    # Remove diagonal.
    valid &= ~torch.eye(
        len(embeddings),
        dtype=torch.bool,
        device=embeddings.device,
    )

    # Optional hard temporal cutoff.
    if max_delta_seconds is not None:
        valid &= (
            dt <= max_delta_seconds
        )

    # Soft temporal weight.
    weights = torch.exp(
        -dt / tau_seconds
    )

    weights = (
        weights
        * valid.float()
    )

    if weights.sum() == 0:
        return embeddings.sum() * 0.0

    return (
        distances * weights
    ).sum() / weights.sum()


# =====================================================================
# Training
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pair_loss_weight=0.1,
    tau_seconds=20.0,
    max_delta_seconds=60.0,
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_pair_loss = 0.0

    total_correct = 0
    total_samples = 0

    for batch in loader:

        current = batch[
            "current"
        ].to(device)

        baseline = batch[
            "baseline"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        sensor_indexs = batch[
            "sensor_index"
        ].to(device)

        timestamps = batch[
            "timestamp"
        ].to(device)

        if "presentation_id" in batch:
            presentation_ids = batch[
                "presentation_id"
            ].to(device)
        else:
            presentation_ids = None

        optimizer.zero_grad()

        output = model(
            current=current,
            baseline=baseline,
        )

        logits = output[
            "logits"
        ]

        embeddings = output[
            "embedding"
        ]

        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------

        classification_loss = (
            F.cross_entropy(
                logits,
                labels,
            )
        )

        # ---------------------------------------------------------
        # Temporal cross-sensor alignment
        # ---------------------------------------------------------

        pair_loss = (
            temporal_cross_sensor_loss(
                embeddings=embeddings,
                labels=labels,
                sensor_indexs=sensor_indexs,
                timestamps=timestamps,
                presentation_ids=(
                    presentation_ids
                ),
                tau_seconds=tau_seconds,
                max_delta_seconds=(
                    max_delta_seconds
                ),
            )
        )

        loss = (
            classification_loss
            + pair_loss_weight * pair_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = len(labels)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_cls_loss += (
            classification_loss.item()
            * batch_size
        )

        total_pair_loss += (
            pair_loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return {
        "loss":
            total_loss / total_samples,

        "classification_loss":
            total_cls_loss / total_samples,

        "pair_loss":
            total_pair_loss / total_samples,

        "accuracy":
            total_correct / total_samples,
    }


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    embeddings = []
    labels = []
    sensor_indexs = []
    timestamps = []

    for batch in loader:

        current = batch[
            "current"
        ].to(device)

        baseline = batch[
            "baseline"
        ].to(device)

        y = batch[
            "label"
        ].to(device)

        output = model(
            current=current,
            baseline=baseline,
        )

        logits = output[
            "logits"
        ]

        loss = F.cross_entropy(
            logits,
            y,
        )

        batch_size = len(y)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == y
        ).sum().item()

        total_samples += batch_size

        embeddings.append(
            output[
                "embedding"
            ].cpu()
        )

        labels.append(
            y.cpu()
        )

        sensor_indexs.append(
            batch[
                "sensor_index"
            ].cpu()
        )

        timestamps.append(
            batch[
                "timestamp"
            ].cpu()
        )

    return {
        "loss":
            total_loss / total_samples,

        "accuracy":
            total_correct / total_samples,

        "embeddings":
            torch.cat(embeddings),

        "labels":
            torch.cat(labels),

        "sensor_indexs":
            torch.cat(sensor_indexs),

        "timestamps":
            torch.cat(timestamps),
    }


# =====================================================================
# Validation of input data
# =====================================================================

def validate_dataframe(df):

    required = [
        "sensor_index",
        "label",
        "cycle_start_timestamp",
        *CURRENT_FEATURE_COLUMNS,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"    {c}"
                for c in missing
            )
        )

    if df["cycle_start_timestamp"].isna().any():
        raise ValueError(
            "cycle_start_timestamp contains NaNs."
        )

    if df["sensor_index"].isna().any():
        raise ValueError(
            "sensor_index contains NaNs."
        )

    if df["label"].isna().any():
        raise ValueError(
            "label contains NaNs."
        )


# =====================================================================
# Example training script
# =====================================================================
def compute_baseline(df):
    """Compute each sensor's baseline as the mean of its current features."""
    result = df.copy()
    baseline = (
        result.groupby("sensor_index", sort=False)[CURRENT_FEATURE_COLUMNS]
        .transform("mean")
    )
    baseline.columns = BASELINE_FEATURE_COLUMNS
    result[BASELINE_FEATURE_COLUMNS] = baseline
    return result


def load_dataframe():
    from load_data import log_vapor_pressure, pivot_bmedata
    nut_df = pd.read_csv('NRC_BME690/bme_690_all_nuts.csv')
    spice_df = pd.read_csv('NRC_BME690/bme_690_all_spices.csv')
    herb_df = pd.read_csv('NRC_BME690/bme_690_all_herbs.csv')

    # raw_df = load_bmerawdata(f'NRC_BME690/bme_690_data_14.bmerawdata')
    # raw_df['source_file'] = 'raw'
    # raw_df = raw_df[raw_df['timestamp_since_poweron'].between(0.2E6, 1.6E6)]
    sensor_df = pd.concat([nut_df, spice_df, herb_df], ignore_index=True)

    sensor_df['label_merged'] = sensor_df['label'].str.rstrip(
        '0123456789')  # merge labels like peanuts2 -> peanuts
    sensor_df['log_vapor_pressure'] = log_vapor_pressure(
        sensor_df['temperature'], sensor_df['relative_humidity'])
    sensor_df['log_resistance'] = np.log10(
        sensor_df['resistance_gassensor'] + 1)  # Add 1 to avoid log(0)
    feature_matrix = pivot_bmedata(sensor_df, features=[
                                   'log_resistance', 'relative_humidity', 'temperature', 'log_vapor_pressure', 'resistance_gassensor'])
    return feature_matrix


if __name__ == "__main__":

    set_seed(1337)

    df = load_dataframe()
    baseline = compute_baseline(df)

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    BATCH_SIZE = 128

    LEARNING_RATE = 1e-3

    WEIGHT_DECAY = 1e-4

    NUM_EPOCHS = 100

    # Strength of cross-sensor alignment.
    PAIR_LOSS_WEIGHT = 0.1

    # Temporal similarity scale.
    #
    # A pair 20 seconds apart has weight exp(-1) ~= 0.37.
    #
    # Tune this based on your convergence dynamics.
    TAU_SECONDS = 20.0

    # Ignore very distant pairs.
    MAX_PAIR_DELTA_SECONDS = 60.0

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_dataframe(df)

    # -------------------------------------------------------------
    # Parse timestamp
    # -------------------------------------------------------------

    df = df.copy()

    df["cycle_start_timestamp"] = (
        pd.to_datetime(
            df["cycle_start_timestamp"],
            unit='us'
        )
    )

    # -------------------------------------------------------------
    # Train/validation split
    #
    # IMPORTANT:
    #
    # For a real sensor-transfer experiment, replace this with
    # leave-one-sensor-out splitting.
    # -------------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Label encoding
    # -------------------------------------------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        train_df["label"]
    )

    # -------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------

    train_dataset = SensorDataset(
        train_df,
        baseline,
        label_encoder,
    )

    val_dataset = SensorDataset(
        val_df,
        baseline,
        label_encoder,
    )

    # -------------------------------------------------------------
    # Stratified batches
    # -------------------------------------------------------------

    train_batch_sampler = (
        StratifiedBatchSampler(
            labels=train_dataset.labels,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
        )
    )

    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = (
        BaselineConditionedEncoder(
            input_dim=20,
            encoder_hidden_dim=64,
            encoder_dim=32,
            fusion_hidden_dim=64,
            embedding_dim=16,
            num_classes=(
                len(
                    label_encoder.classes_
                )
            ),
            dropout=0.05,
        )
        .to(DEVICE)
    )

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    best_model_state = None
    best_val_accuracy = -np.inf

    for epoch in range(
        NUM_EPOCHS
    ):

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=DEVICE,

                pair_loss_weight=(
                    PAIR_LOSS_WEIGHT
                ),

                tau_seconds=(
                    TAU_SECONDS
                ),

                max_delta_seconds=(
                    MAX_PAIR_DELTA_SECONDS
                ),
            )
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=DEVICE,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train loss="
            f"{train_metrics['loss']:.4f} | "
            f"train cls="
            f"{train_metrics['classification_loss']:.4f} | "
            f"train pair="
            f"{train_metrics['pair_loss']:.4f} | "
            f"train acc="
            f"{train_metrics['accuracy']:.4f} | "
            f"val loss="
            f"{val_metrics['loss']:.4f} | "
            f"val acc="
            f"{val_metrics['accuracy']:.4f}"
        )

        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

    # -------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    print()
    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.4f}",
    )

    # -------------------------------------------------------------
    # Extract embeddings
    # -------------------------------------------------------------

    final_val = evaluate(
        model=model,
        loader=val_loader,
        device=DEVICE,
    )

    embeddings = (
        final_val["embeddings"]
        .numpy()
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )
