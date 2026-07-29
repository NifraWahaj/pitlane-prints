"""
contrastive_encoder.py
-----------------------
Trains the same 1D-CNN architecture as cnn_encoder.py, but with a
Supervised Contrastive (SupCon) loss instead of cross-entropy.

Cross-entropy trains the encoder indirectly — the embedding is a
byproduct of learning to classify. SupCon trains it directly: pull
same-driver laps together and push different-driver laps apart in
embedding space, with no classifier head involved at all. This
typically produces embeddings that generalize better with small
datasets, since the objective matches what we actually want (a
clustered embedding space) rather than a proxy (classification logits).

Uses the same held-out-circuit split as cnn_encoder.py so the two
approaches are directly comparable on cross-circuit generalization.

Usage:
    python src/models/contrastive_encoder.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from cnn_encoder import (
    load_config, load_raw_telemetry, build_sequences,
    TELEMETRY_CHANNELS, SEQ_LEN,
)


# ─────────────────────────────────────────────
# Model — same conv backbone as DriverEncoder, but the forward pass
# returns an L2-normalized embedding, no classifier head.
# ─────────────────────────────────────────────

class ContrastiveEncoder(nn.Module):
    def __init__(self, n_channels: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        features  = self.encoder(x)
        embedding = self.projector(features)
        return F.normalize(embedding, dim=1)


# ─────────────────────────────────────────────
# Supervised Contrastive Loss (Khosla et al., 2020)
# For each anchor, positives = all other samples in the batch with the
# same label. Pulls all same-driver embeddings together, pushes all
# different-driver embeddings apart — richer signal than triplet loss,
# which only looks at one positive/negative pair at a time.
# ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        same_label = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(len(labels), device=device)
        positive_mask = same_label - self_mask  # exclude self as its own positive

        exp_sim = torch.exp(sim) * (1 - self_mask)  # exclude self from denominator
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        n_positives = positive_mask.sum(dim=1)
        valid = n_positives > 0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / n_positives[valid]

        return -mean_log_prob_pos.mean()


# ─────────────────────────────────────────────
# Class-balanced batch sampler — SupCon needs multiple positives per
# anchor in every batch, so plain random shuffling isn't reliable
# with only 6 classes. This guarantees every batch has ~equal
# representation from each driver.
# ─────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, batch_size: int, samples_per_class: int = 8):
        self.labels = labels
        self.classes = np.unique(labels)
        self.samples_per_class = samples_per_class
        self.batch_size = len(self.classes) * samples_per_class
        self.class_indices = {c: np.where(labels == c)[0] for c in self.classes}
        self.n_batches = len(labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                idx = np.random.choice(self.class_indices[c], self.samples_per_class, replace=False)
                batch.extend(idx.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class TelemetryDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    races         = config["races"]
    drivers       = config["drivers"]
    embedding_dim = config["model"]["embedding_dim"]
    seed          = config["model"]["random_seed"]
    models_dir    = os.path.join("outputs", "models")
    features_dir  = config["data"]["features_dir"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[device] Using: {device}")

    print("\n[data] Loading telemetry across all configured races...")
    df = load_raw_telemetry(processed_dir, races, drivers)
    sequences, labels_str, lap_meta = build_sequences(df, SEQ_LEN)

    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    n_classes  = len(le.classes_)
    n_channels = sequences.shape[1]
    race_ids   = np.array([m[1] for m in lap_meta])

    idx = np.arange(len(sequences))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(idx, labels, groups=race_ids))
    print(f"[split] Held-out races: {sorted(set(race_ids[val_idx]))}")
    print(f"[split] Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    train_ds = TelemetryDataset(sequences[train_idx], labels[train_idx])
    val_ds   = TelemetryDataset(sequences[val_idx],   labels[val_idx])

    sampler = BalancedBatchSampler(labels[train_idx], batch_size=n_classes * 8, samples_per_class=8)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ContrastiveEncoder(n_channels=n_channels, embedding_dim=embedding_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    criterion = SupConLoss(temperature=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    EPOCHS = 80
    EARLY_STOP_PATIENCE = 15
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    def get_embeddings(loader):
        model.eval()
        embs, ys = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                emb = model(X_batch.to(device))
                embs.append(emb.cpu().numpy())
                ys.append(y_batch.numpy())
        return np.concatenate(embs), np.concatenate(ys)

    def evaluate_val_acc():
        """kNN probe: fit on train embeddings, score on held-out-circuit val embeddings."""
        train_emb, train_y = get_embeddings(DataLoader(train_ds, batch_size=64, shuffle=False))
        val_emb, val_y     = get_embeddings(val_loader)
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(train_emb, train_y)
        pred = knn.predict(val_emb)
        return accuracy_score(val_y, pred)

    print("\n[train] Starting SupCon training...")
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            emb = model(X_batch)
            loss = criterion(emb, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total += len(y_batch)
        train_loss = total_loss / total
        scheduler.step(train_loss)

        val_acc = evaluate_val_acc()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_knn_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, "all_races_contrastive_encoder.pt"))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train loss {train_loss:.4f} | val kNN acc {val_acc*100:.1f}%")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    print(f"\n[best] Val kNN accuracy: {best_val_acc*100:.1f}%  at epoch {best_epoch}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(features_dir, "all_races_contrastive_history.csv"), index=False)

    # Reload best checkpoint and extract embeddings for ALL laps
    model.load_state_dict(torch.load(
        os.path.join(models_dir, "all_races_contrastive_encoder.pt"), map_location=device))
    full_loader = DataLoader(TelemetryDataset(sequences, labels), batch_size=64, shuffle=False)
    all_emb, all_y = get_embeddings(full_loader)

    sil = silhouette_score(all_emb, all_y, metric="cosine")
    print(f"[embed] Silhouette (cosine, all laps): {sil:.4f}")

    embed_df = pd.DataFrame(all_emb, columns=[f"dim_{i}" for i in range(embedding_dim)])
    embed_df["Driver"]    = labels_str
    embed_df["Race"]      = [m[1] for m in lap_meta]
    embed_df["LapNumber"] = [m[2] for m in lap_meta]
    embed_df.to_csv(os.path.join(features_dir, "all_races_contrastive_embeddings.csv"), index=False)
    print(f"[saved] Embeddings → {features_dir}/all_races_contrastive_embeddings.csv")

    import joblib
    joblib.dump(le, os.path.join(models_dir, "all_races_contrastive_label_encoder.pkl"))

    print(f"\n[summary] SupCon cross-circuit kNN accuracy: {best_val_acc*100:.1f}%")
    print(f"[summary] Cross-entropy CNN cross-circuit val accuracy (from cnn_encoder.py): 94.2%")
    print(f"[summary] XGBoost cross-circuit accuracy (from baseline.py): 61.2%")


if __name__ == "__main__":
    main()
