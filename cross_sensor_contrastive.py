"""
baseline_conditioned_sensor_model.py

Model:
    Parallel sequences representing 10 heating steps.
    Each token fuses:
      - current compensated resistance
      - current vapor pressure
      - absolute baseline resistance
      - placeholder for target temperature
      - placeholder for time offset

    Token_i -> Linear Projection -> Transformer Encoder

    [CLS] token is extracted as the sensor-invariant embedding.

Expected DataFrame columns:
    sensor_index, label, cycle_start_timestamp
    log_resistance_0 ... log_resistance_9
    log_vapor_pressure_0 ... log_vapor_pressure_9

Dependencies:
    pip install torch pandas numpy scikit-learn
"""

import copy
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# Configuration
# =====================================================================

NUM_HEATER_STEPS = 10

COMPENSATED_RESISTANCE_COLUMNS = [f"resistance_compensated_{k}" for k in range(NUM_HEATER_STEPS)]
RESISTANCE_COLUMNS = [f"log_resistance_{k}" for k in range(NUM_HEATER_STEPS)]
VAPOR_COLUMNS = [f"log_vapor_pressure_{k}" for k in range(NUM_HEATER_STEPS)]

CURRENT_FEATURE_COLUMNS = RESISTANCE_COLUMNS + VAPOR_COLUMNS
BASELINE_FEATURE_COLUMNS = RESISTANCE_COLUMNS + VAPOR_COLUMNS

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
# Explicit cross-sensor contrastive representation learning
# =====================================================================

class SensorInvariantEncoder(nn.Module):
    """
    Small encoder for:
      [logR_i - baseline_logR_i, logVP_i] for i=0..9

    No heating-profile information is supplied.
    The embedding is trained to bring simultaneous same-label measurements
    from different sensors together.
    """
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_relative_features(df, baseline):
    """
    Baseline-relative representation:
      resistance: logR_i - sensor_baseline_logR_i
      vapor:      log vapor pressure_i

    Baseline is computed from TRAINING data only.
    """
    r = df[RESISTANCE_COLUMNS].to_numpy(np.float32)
    vp = df[VAPOR_COLUMNS].to_numpy(np.float32)

    # baseline is indexed by sensor_index
    b = baseline.loc[df["sensor_index"], RESISTANCE_COLUMNS].to_numpy(np.float32)

    rel_r = r - b
    # rel_r = df[COMPENSATED_RESISTANCE_COLUMNS].to_numpy(np.float32)
    return np.concatenate([rel_r, vp], axis=1).astype(np.float32)


class CrossSensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, labels, sensor_ids, timestamps):
        self.x = torch.tensor(x)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sensor_ids = torch.tensor(sensor_ids, dtype=torch.long)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            "label": self.labels[i],
            "sensor_id": self.sensor_ids[i],
            "timestamp": self.timestamps[i],
        }


def explicit_cross_sensor_supcon_loss(
    embeddings,
    labels,
    sensor_ids,
    timestamps,
    timestamp_tolerance_seconds=2.0,
    temperature=0.1,
):
    """
    Supervised contrastive loss with an important restriction:

    Positive = same label + DIFFERENT sensor + close timestamp.

    Thus the model cannot satisfy the main objective by clustering samples
    from the same sensor. If your paired acquisitions have timestamps that
    differ by more than this, increase the tolerance.
    """
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature

    n = len(labels)
    device = embeddings.device
    eye = torch.eye(n, dtype=torch.bool, device=device)

    dt = (timestamps[:, None] - timestamps[None, :]).abs()

    positive = (
        (labels[:, None] == labels[None, :])
        & (sensor_ids[:, None] != sensor_ids[None, :])
        & (dt <= timestamp_tolerance_seconds)
        & ~eye
    )

    # Every other sample is a negative, except self.
    valid = ~eye

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * valid.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    positive_count = positive.sum(dim=1)
    valid_anchor = positive_count > 0

    if not valid_anchor.any():
        return embeddings.sum() * 0.0, 0

    loss = -(
        log_prob * positive.float()
    ).sum(dim=1) / positive_count.clamp_min(1)

    return loss[valid_anchor].mean(), int(valid_anchor.sum())


def build_cross_sensor_pairs(df, tolerance_seconds=2.0):
    """
    Diagnostic: report how many cross-sensor positive relationships exist.

    Assumes cycle_start_timestamp identifies simultaneous/near-simultaneous
    acquisitions. If your true pairing is represented by another column,
    replace this matching logic with that acquisition/sample ID.
    """
    tmp = df[["sensor_index", "label", "cycle_start_timestamp"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["cycle_start_timestamp"])
    tmp = tmp.sort_values("timestamp")

    positives = 0
    for _, g in tmp.groupby("label"):
        sensors = g["sensor_index"].to_numpy()
        ts = g["timestamp"].astype("int64").to_numpy() / 1e9

        # Count cross-sensor matches within tolerance.
        for i in range(len(g)):
            close = np.abs(ts - ts[i]) <= tolerance_seconds
            positives += np.sum(close & (sensors != sensors[i]))

    return positives // 2


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    zs, ys, ss = [], [], []

    for batch in loader:
        z = model(batch["x"].to(device))
        zs.append(F.normalize(z, dim=1).cpu().numpy())
        ys.append(batch["label"].numpy())
        ss.append(batch["sensor_id"].numpy())

    return np.concatenate(zs), np.concatenate(ys), np.concatenate(ss)
# =====================================================================
# Example training script
# =====================================================================
def compute_baseline(df):
    """Compute each sensor's baseline as the mean of its current features."""
    return df.groupby("sensor_index", sort=False)[
        BASELINE_FEATURE_COLUMNS
    ].transform("mean").reset_index()

def get_heatprofiles():
    profile_301 = np.array([
        [100, 2],
        [100, 41],
        [200, 2],
        [200, 14],
        [200, 14],
        [200, 14],
        [320, 2],
        [320, 14],
        [320, 14],
        [320, 14],
    ], dtype=np.float32)

    profile_354 = np.array([
        [320, 5],
        [100, 2],
        [100, 10],
        [100, 30],
        [200, 5],
        [200, 5],
        [200, 5],
        [320, 5],
        [320, 5],
        [320, 5],
    ], dtype=np.float32)

    profile_411 = np.array([
        [100, 43],
        [320, 2],
        [170, 43],
        [320, 2],
        [240, 2],
        [240, 20],
        [240, 21],
        [320, 2],
        [320, 20],
        [320, 21],
    ], dtype=np.float32)

    profile_501 = np.array([
        [210, 24],
        [265, 2],
        [265, 22],
        [320, 2],
        [320, 22],
        [265, 24],
        [210, 24],
        [155, 24],
        [100, 24],
        [155, 24],
    ], dtype=np.float32)

    profiles = [
        profile_301,
        profile_354,
        profile_411,
        profile_501,
    ]

    heatprofile_array = []
    for profile in profiles:
        durations = np.array([step[1] for step in profile], dtype=np.float32)
        times = np.cumsum(durations) * 140.0 / 1000.0
        temps = np.array([step[0] for step in profile], dtype=np.float32)
        temps = (temps - temps.min()) / (temps.max() - temps.min())
        step_array = np.stack([
            temps,
            times,
        ], axis=-1)
        heatprofile_array.append(step_array)

    return np.stack(heatprofile_array, axis=0)


def normalize_by_group_wide(df):
    from sklearn.linear_model import LinearRegression
    def get_residuals(sensor_group):
        group = sensor_group.copy()

        resistance_cols = [c for c in group.columns if c.startswith('log_resistance')]
        steps = [c.split('_')[-1] for c in resistance_cols]

        for step in steps:
            x = group[f'log_vapor_pressure_{step}'].to_numpy().reshape(-1,1)
            y = group[f'log_resistance_{step}']

            lr = LinearRegression()
            lr.fit(x, y)
            retval = y - lr.predict(x)

            group[f'resistance_compensated_{step}'] = retval

        return group

    return (
        df.groupby("sensor_index")
        .apply(get_residuals, include_groups=False)
        .reset_index(level="sensor_index")
    )

def load_dataframe():
    from load_data import log_vapor_pressure, pivot_bmedata

    nut_df = pd.read_csv("NRC_BME690/bme_690_all_nuts.csv")
    spice_df = pd.read_csv("NRC_BME690/bme_690_all_spices.csv")
    herb_df = pd.read_csv("NRC_BME690/bme_690_all_herbs.csv")

    sensor_df = pd.concat([nut_df, spice_df, herb_df], ignore_index=True)

    sensor_df["label_merged"] = sensor_df["label"].str.rstrip(
        "0123456789"
    )
    sensor_df["log_vapor_pressure"] = log_vapor_pressure(
        sensor_df["temperature"], sensor_df["relative_humidity"]
    )
    sensor_df["log_resistance"] = np.log10(
        sensor_df["resistance_gassensor"] + 1
    )
    feature_matrix = pivot_bmedata(
        sensor_df,
        features=[
            "log_resistance",
            "relative_humidity",
            "temperature",
            "log_vapor_pressure",
            "resistance_gassensor",
        ],
    )

    profiles = [0, 1, 2, 3, 0, 1, 2, 3]
    feature_matrix['heating_profile_id'] = feature_matrix['sensor_index'].apply(
        lambda x: profiles[x])
    feature_matrix = normalize_by_group_wide(feature_matrix)
    return feature_matrix



# =====================================================================
# Training / evaluation
# =====================================================================

if __name__ == "__main__":
    set_seed(1337)

    df = load_dataframe()

    # -------------------------------------------------------------
    # Leave-one-sensor-out split
    # -------------------------------------------------------------
    val_sensor = 4

    train_df = df[df["sensor_index"] != val_sensor].copy().reset_index(drop=True)
    val_df = df[df["sensor_index"] == val_sensor].copy().reset_index(drop=True)

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["label"])
    val_labels = label_encoder.transform(val_df["label"])

    # -------------------------------------------------------------
    # IMPORTANT: compute baselines from TRAINING sensors only.
    # -------------------------------------------------------------
    train_baseline = (
        train_df.groupby("sensor_index")[RESISTANCE_COLUMNS]
        .mean()
    )

    # For the held-out sensor, estimate its baseline from its unlabeled
    # measurements. This is allowed because baseline calibration does not
    # use labels. If you want a completely cold-start experiment, replace
    # this with a baseline obtained from a calibration segment only.
    val_baseline = val_df.groupby("sensor_index")[RESISTANCE_COLUMNS].mean()

    baseline_all = pd.concat([train_baseline, val_baseline])

    X_train = make_relative_features(train_df, baseline_all)
    X_val = make_relative_features(val_df, baseline_all)

    train_t = (
        pd.to_datetime(train_df["cycle_start_timestamp"])
        .astype("int64").to_numpy() / 1e9
    )
    val_t = (
        pd.to_datetime(val_df["cycle_start_timestamp"])
        .astype("int64").to_numpy() / 1e9
    )

    train_dataset = CrossSensorDataset(
        X_train,
        train_labels,
        train_df["sensor_index"].to_numpy(),
        train_t,
    )
    val_dataset = CrossSensorDataset(
        X_val,
        val_labels,
        val_df["sensor_index"].to_numpy(),
        val_t,
    )

    # Large batches are useful because positives are discovered within
    # the batch. We deliberately do NOT use the old label-only sampler.
    BATCH_SIZE=256
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SensorInvariantEncoder(embedding_dim=32).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    NUM_EPOCHS = 150
    best_loss = np.inf
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_anchors = 0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            s = batch["sensor_id"].to(DEVICE)
            t = batch["timestamp"].to(DEVICE)

            optimizer.zero_grad()

            z = model(x)

            loss, anchors = explicit_cross_sensor_supcon_loss(
                z,
                y,
                s,
                t,
                timestamp_tolerance_seconds=2.0,
                temperature=0.1,
            )

            if anchors == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_anchors += anchors
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"contrastive loss={mean_loss:.4f} | "
                f"anchors={n_anchors}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------------------------------------------------------------
    # Extract embeddings
    # -------------------------------------------------------------
    Z_train, Y_train, S_train = extract_embeddings(
        model, train_loader, DEVICE
    )
    Z_val, Y_val, S_val = extract_embeddings(
        model, val_loader, DEVICE
    )

    # -------------------------------------------------------------
    # Linear evaluation: exactly the downstream test we care about.
    # Train only a linear SVM on the learned representation.
    # -------------------------------------------------------------
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    clf = LinearSVC(C=1.0, dual="auto", max_iter=10000)
    clf.fit(Z_train, Y_train)

    train_pred = clf.predict(Z_train)
    val_pred = clf.predict(Z_val)

    print()
    print("Representation evaluation")
    print("-------------------------")
    print("Train linear SVM accuracy:", accuracy_score(Y_train, train_pred))
    print("Held-out sensor accuracy:", accuracy_score(Y_val, val_pred))
    print("Embedding shape:", Z_train.shape)
