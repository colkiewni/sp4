"""
Training script. Runs on GPU (Colab T4 / Kaggle).
Trains play model (policy + value) and bid model from self-play data.
"""

from __future__ import annotations
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nn import PlayModel, BidModel, export_onnx
from features import FEATURE_DIM, BID_FEATURE_DIM


class PlayDataset(Dataset):
    def __init__(self, features, policies, values):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.policies[idx], self.values[idx]


class BidDataset(Dataset):
    def __init__(self, features, targets, values):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.values = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.values[idx]


def load_selfplay_data(data_dir: str, max_files: int = None):
    """Load all .npz files from selfplay output directory."""
    files = sorted(glob.glob(os.path.join(data_dir, 'game_*.npz')))
    if max_files:
        files = files[:max_files]

    play_f, play_p, play_v = [], [], []
    bid_f, bid_t, bid_v = [], [], []

    for f in files:
        data = np.load(f)
        if 'play_features' in data:
            play_f.append(data['play_features'])
            play_p.append(data['play_policies'])
            play_v.append(data['play_values'])
        if 'bid_features' in data:
            bid_f.append(data['bid_features'])
            bid_t.append(data['bid_targets'])
            bid_v.append(data['bid_values'])

    result = {}
    if play_f:
        result['play'] = (
            np.concatenate(play_f),
            np.concatenate(play_p),
            np.concatenate(play_v)
        )
        print(f"Play samples: {len(result['play'][0])}")
    if bid_f:
        result['bid'] = (
            np.concatenate(bid_f),
            np.concatenate(bid_t),
            np.concatenate(bid_v)
        )
        print(f"Bid samples: {len(result['bid'][0])}")

    return result


def train_play_model(
    data: tuple,
    model: PlayModel = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = None,
    save_dir: str = 'models'
) -> PlayModel:
    """Train the play model (policy + value)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training play model on {device}")

    os.makedirs(save_dir, exist_ok=True)
    features, policies, values = data

    n = len(features)
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = PlayDataset(features[train_idx], policies[train_idx], values[train_idx])
    val_ds = PlayDataset(features[val_idx], policies[val_idx], values[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device == 'cuda'))

    if model is None:
        model = PlayModel()
    model = model.to(device)
    print(f"Play model params: {model.param_count():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for feat, pol, val in train_dl:
            feat, pol, val = feat.to(device), pol.to(device), val.to(device)
            policy_logits, value_pred = model(feat)

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(pol * log_probs, dim=1).mean()
            value_loss = F.mse_loss(value_pred, val)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(feat)
            train_count += len(feat)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_pol_sum = 0.0
        val_val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for feat, pol, val in val_dl:
                feat, pol, val = feat.to(device), pol.to(device), val.to(device)
                policy_logits, value_pred = model(feat)

                log_probs = F.log_softmax(policy_logits, dim=1)
                pol_loss = -torch.sum(pol * log_probs, dim=1).mean()
                val_loss = F.mse_loss(value_pred, val)

                val_pol_sum += pol_loss.item() * len(feat)
                val_val_sum += val_loss.item() * len(feat)
                val_count += len(feat)

        train_avg = train_loss_sum / train_count
        val_pol_avg = val_pol_sum / val_count
        val_val_avg = val_val_sum / val_count
        val_avg = val_pol_avg + val_val_avg

        elapsed = time.time() - t_start
        eta_min = elapsed / (epoch + 1) * (epochs - epoch - 1) / 60
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_avg:.4f} | "
              f"Val policy: {val_pol_avg:.4f} | "
              f"Val value: {val_val_avg:.4f} | "
              f"Val total: {val_avg:.4f} | "
              f"{elapsed:.0f}s elapsed | ETA: {eta_min:.1f}min")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), os.path.join(save_dir, 'play_model_best.pt'))
            print(f"  ↳ Saved best model (val={val_avg:.4f})")

    # Export ONNX from the best checkpoint, not the final-epoch weights —
    # val loss can tick back up after the best epoch, and everything
    # downstream (evaluate.py, selfplay, play) uses the .onnx.
    best_pt = os.path.join(save_dir, 'play_model_best.pt')
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location='cpu'))
    model.cpu().eval()
    export_onnx(model, os.path.join(save_dir, 'play_model.onnx'), FEATURE_DIM)
    print(f"Exported ONNX to {save_dir}/play_model.onnx")

    return model


def train_bid_model(
    data: tuple,
    model: BidModel = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = None,
    save_dir: str = 'models'
) -> BidModel:
    """Train the bid model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training bid model on {device}")

    os.makedirs(save_dir, exist_ok=True)
    features, targets, values = data

    n = len(features)
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = BidDataset(features[train_idx], targets[train_idx], values[train_idx])
    val_ds = BidDataset(features[val_idx], targets[val_idx], values[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=(device == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device == 'cuda'))

    if model is None:
        model = BidModel()
    model = model.to(device)
    print(f"Bid model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for feat, tgt, val in train_dl:
            feat, tgt, val = feat.to(device), tgt.to(device), val.to(device)
            logits = model(feat)
            loss = F.cross_entropy(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(feat)
            train_count += len(feat)

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for feat, tgt, val in val_dl:
                feat, tgt = feat.to(device), tgt.to(device)
                logits = model(feat)
                loss = F.cross_entropy(logits, tgt)
                preds = logits.argmax(dim=1)

                val_loss_sum += loss.item() * len(feat)
                val_correct += (preds == tgt).sum().item()
                val_count += len(feat)

        train_avg = train_loss_sum / train_count
        val_avg = val_loss_sum / val_count
        val_acc = val_correct / val_count

        elapsed = time.time() - t_start
        eta_min = elapsed / (epoch + 1) * (epochs - epoch - 1) / 60
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_avg:.4f} | "
              f"Val: {val_avg:.4f} | "
              f"Val acc: {val_acc:.3f} | "
              f"{elapsed:.0f}s elapsed | ETA: {eta_min:.1f}min")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), os.path.join(save_dir, 'bid_model_best.pt'))
            print(f"  ↳ Saved best model (val={val_avg:.4f}, acc={val_acc:.3f})")

    # Same as play model: export the best checkpoint, not final-epoch weights
    best_pt = os.path.join(save_dir, 'bid_model_best.pt')
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location='cpu'))
    model.cpu().eval()
    export_onnx(model, os.path.join(save_dir, 'bid_model.onnx'), BID_FEATURE_DIM)
    print(f"Exported ONNX to {save_dir}/bid_model.onnx")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Spades NN models')
    parser.add_argument('--data', type=str, default='selfplay_data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default='models')
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    data = load_selfplay_data(args.data, max_files=args.max_files)

    if 'play' in data:
        train_play_model(data['play'], epochs=args.epochs,
                         batch_size=args.batch, lr=args.lr, save_dir=args.save)

    if 'bid' in data:
        train_bid_model(data['bid'], epochs=args.epochs,
                        batch_size=args.batch, lr=args.lr, save_dir=args.save)
