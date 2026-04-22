"""sklearn-compatible PyTorch wrappers for V2 NN models.

All wrappers expect an already-preprocessed **dense float32 array** at fit/predict
time. The standard ``ColumnTransformer`` from ``readmit_bench.features.pipeline``
returns a sparse matrix; the V2 training driver densifies it once before
splitting into train / internal-val for early stopping.

Each wrapper exposes the minimal sklearn API (``fit`` + ``predict_proba``) so
the existing ``compute_metrics`` / persistence logic in
``readmit_bench.models.baselines`` can be reused without changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def _seed_torch(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class _MLPNet(nn.Module):
    def __init__(
        self, n_features: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPClassifier:
    """Plain PyTorch MLP wrapped in an sklearn-style estimator."""

    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    epochs: int = 8
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 2
    val_frac: float = 0.1
    device: torch.device = DEVICE

    def __post_init__(self) -> None:
        self.model_: nn.Module | None = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        _seed_torch()
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n = len(X)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        n_val = max(1024, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        ds_tr = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
        ds_va = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        self.n_features_ = X.shape[1]
        self.model_ = _MLPNet(self.n_features_, self.hidden, self.dropout).to(self.device)
        pos = float(y[tr_idx].sum())
        neg = float(len(tr_idx) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            tr_loss = 0.0
            n_seen = 0
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()
                tr_loss += float(loss.detach()) * len(xb)
                n_seen += len(xb)
            tr_loss /= max(n_seen, 1)

            self.model_.eval()
            va_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model_(xb)
                    va_loss += float(loss_fn(logits, yb)) * len(xb)
                    n_va += len(xb)
            va_loss /= max(n_va, 1)
            logger.info(
                "MLP epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch + 1,
                self.epochs,
                tr_loss,
                va_loss,
            )

            if va_loss + 1e-6 < best_val:
                best_val = va_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info("MLP early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPClassifier is not fitted")
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.model_.eval()
        out = np.empty(len(X), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i : i + bs]).to(self.device)
                p = torch.sigmoid(self.model_(xb)).cpu().numpy()
                out[i : i + bs] = p
        return np.column_stack([1.0 - out, out])


# ---------------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------------


@dataclass
class TabNetWrapper:
    """Thin wrapper over ``pytorch-tabnet`` to fit our sklearn-style harness."""

    n_d: int = 16
    n_a: int = 16
    n_steps: int = 3
    gamma: float = 1.3
    lambda_sparse: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 4096
    virtual_batch_size: int = 512
    val_frac: float = 0.1

    def __post_init__(self) -> None:
        self.clf_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabNetWrapper:
        from pytorch_tabnet.tab_model import TabNetClassifier

        _seed_torch()
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(X)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        n_val = max(1024, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        self.clf_ = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            seed=SEED,
            verbose=0,
            device_name=str(DEVICE),
        )
        self.clf_.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("TabNetWrapper is not fitted")
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.clf_.predict_proba(X)


# ---------------------------------------------------------------------------
# FT-Transformer
# ---------------------------------------------------------------------------


@dataclass
class FTTransformerClassifier:
    """FT-Transformer (Gorishniy et al., 2021) treating all preprocessed columns as numerical.

    The preprocessor already one-hot-encodes low-card cats and target-encodes
    high-card cats, so the input matrix is fully numeric — we feed it to the
    "numerical-only" path of ``rtdl_revisiting_models.FTTransformer``.
    """

    d_block: int = 96
    n_blocks: int = 3
    attention_n_heads: int = 4
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    epochs: int = 6
    batch_size: int = 1024
    lr: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 2
    val_frac: float = 0.1
    device: torch.device = DEVICE

    def __post_init__(self) -> None:
        self.model_ = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FTTransformerClassifier:
        from rtdl_revisiting_models import FTTransformer

        _seed_torch()
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(X)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        n_val = max(1024, int(n * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        self.n_features_ = X.shape[1]

        # FTTransformer.make_default needs n_blocks for d_block schedule.
        self.model_ = FTTransformer(
            n_cont_features=self.n_features_,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        ).to(self.device)

        ds_tr = TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx]))
        ds_va = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        pos = float(y[tr_idx].sum())
        neg = float(len(tr_idx) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            tr_loss = 0.0
            n_seen = 0
            for xb, yb in dl_tr:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad(set_to_none=True)
                logits = self.model_(xb, None).squeeze(-1)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()
                tr_loss += float(loss.detach()) * len(xb)
                n_seen += len(xb)
            tr_loss /= max(n_seen, 1)

            self.model_.eval()
            va_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model_(xb, None).squeeze(-1)
                    va_loss += float(loss_fn(logits, yb)) * len(xb)
                    n_va += len(xb)
            va_loss /= max(n_va, 1)
            logger.info(
                "FT-T epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch + 1,
                self.epochs,
                tr_loss,
                va_loss,
            )

            if va_loss + 1e-6 < best_val:
                best_val = va_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info("FT-T early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("FTTransformerClassifier is not fitted")
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.model_.eval()
        out = np.empty(len(X), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i : i + bs]).to(self.device)
                p = torch.sigmoid(self.model_(xb, None).squeeze(-1)).cpu().numpy()
                out[i : i + bs] = p
        return np.column_stack([1.0 - out, out])
