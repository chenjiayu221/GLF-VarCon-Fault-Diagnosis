"""Shared code for the released checkpoints.

Data checks, preprocessing, model definitions and the two-phase training loop.
Imported by the three evaluation notebooks (01_proposed/evaluate_checkpoint.ipynb,
03_ablation/evaluate_checkpoints.ipynb, 04_sensitivity/evaluate_sensitivity.ipynb);
the training notebooks are self-contained and do not import this file.

Configuration keys use the paper symbols: "alpha" is the momentum of the
centroid update and "beta" the weight of the cross-view consistency KL.
The logs in released_results/ predate this rename and still print them
under their former names, "centroid_momentum" and "alpha_cons".
"""
import os, json, time, random, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# device_count() is checked as well: with CUDA_VISIBLE_DEVICES="" on a machine that has a driver,
# torch.cuda.is_available() can stay True while no device is visible.
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ configuration
CFG = {
    "seed": 163,
    "batch_size": 2048,
    "num_workers": 8,     # the feature masks are drawn inside DataLoader workers, so this value is part of the protocol
    "tf32": True,         # torch.set_float32_matmul_precision("high")
    "proposed": {
        "pretrain_epochs": 100,
        "finetune_epochs": 50,
        "lr_pretrain": 5e-4,
        "lr_finetune": 1e-3,
        "feature_keep_ratio": 0.9,
        "tau_1": 0.07,
        "epsilon": 0.05,
        "alpha": 0.9,          # paper alpha: momentum of the centroid update
        "beta": 0.25,          # paper beta: weight of the cross-view consistency KL
        "freeze_encoder": True,
    },
}


# ------------------------------------------------------------------ data
META_COLUMNS = ["sample_id", "label", "window_number", "source_folder", "source_sample"]
CLASS_NAMES = ["Aging", "Fault", "Noise", "Normal"]   # label ids 0..3 = sorted class names

# the 21 feature columns, in the order the model expects
FEATURE_COLUMNS = [
    "original_peak2peak", "original_rms", "original_kurtosis",
    "imf1_std", "imf1_rms", "imf1_fft_mean_amp",
    "imf2_std", "imf2_kurtosis", "imf2_fft_mean_amp",
    "imf3_std", "imf3_rms", "imf3_fft_mean_amp",
    "imf4_std", "imf4_rms", "imf4_fft_var_amp",
    "imf5_std", "imf5_rms", "imf5_fft_var_amp",
    "imf6_std", "imf6_peak2peak", "imf6_rms",
]

EXPECTED = {
    "selected_features_train.csv": dict(rows=197001, bytes=100295445,
        md5="d62ea1238c7db99a00e7b13506dce04f",
        sha256="9554cfa1b13ce62702fb76a0478a3e15ac839a35c75b965e0d45851865793574"),
    "selected_features_val.csv": dict(rows=21000, bytes=10648172,
        md5="9be06d4be1101dec20fa0da9e0f7bfb5",
        sha256="73b6fd41b5da2706adbd9a2799dc49df4a2087a772f84e998d60244abf279588"),
    "selected_features_test.csv": dict(rows=62002, bytes=30406470,
        md5="2b2ea2831376f973ad5a461950ac14e5",
        sha256="09abcbb5737fc2fd554049c31588f0404649f791025435aa4573b5256a5c8fbf"),
}
EXPECTED_LABEL_COUNTS = {
    "train": {0: 49251, 1: 49250, 2: 49250, 3: 49250},
    "val":   {0: 5250, 1: 5250, 2: 5250, 3: 5250},
    "test":  {0: 15501, 1: 15500, 2: 15501, 3: 15500},
}
# sha256 of the float64 feature matrix: any changed value (even by 1e-9) changes the digest
EXPECTED_DIGEST = {
    "train": "0ec1b643cb02dbb2ce4549e2895ebf56a1714892682a9c76428a6d2203283006",
    "val":   "46019b1b5678e6e4edfb1fdf02b6e0b63d26dec7b80440fe5ab9535df07801e0",
    "test":  "e1df325e31b0fb7447c3c39573755cb6cd954732e7c3b97f4e8f0130113b3ace",
}

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def locate_data(DATA_DIR):
    DATA_DIR = Path(DATA_DIR)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            "data directory not found: " + str(DATA_DIR)
            + "\nexpected files:\n"
            + "\n".join("  " + n + "  rows=" + str(v["rows"])
                        + "  sha256=" + v["sha256"][:16] + "..." for n, v in EXPECTED.items()))
    missing = [n for n in EXPECTED if not (DATA_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            str(DATA_DIR) + " missing files: " + str(missing) + "\nexpected sha256:\n"
            + "\n".join("  " + n + "  " + EXPECTED[n]["sha256"] for n in missing))
    return DATA_DIR

def content_digest(df):
    """sha256 of the float64 feature matrix (C order)."""
    arr = np.ascontiguousarray(df[FEATURE_COLUMNS].values.astype(np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def check_split(df, split):
    """Check row count, column names/order, class counts and content digest of one split."""
    exp = EXPECTED_LABEL_COUNTS[split]
    if len(df) != sum(exp.values()):
        raise ValueError(split + ": rows " + str(len(df)) + " != expected " + str(sum(exp.values())))
    cols = list(df.columns)
    want = META_COLUMNS + FEATURE_COLUMNS
    if cols != want:
        raise ValueError(split + ": column names/order mismatch.\nfirst 8 columns: " + str(cols[:8])
                         + "\nexpected: " + str(want[:8]))
    got = {CLASS_NAMES.index(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
    if got != exp:
        raise ValueError(split + ": class counts " + str(got) + " != expected " + str(exp))
    dig = content_digest(df)
    if dig != EXPECTED_DIGEST[split]:
        raise ValueError(split + ": feature digest mismatch\n got " + dig + "\n expected " + EXPECTED_DIGEST[split])
    return True

def load_and_verify(data_dir, verbose=True):
    d = locate_data(data_dir)
    frames, report = {}, []
    for name, exp in EXPECTED.items():
        p = d / name
        got_bytes, got_md5, got_sha = p.stat().st_size, md5_of(p), sha256_of(p)
        if got_bytes != exp["bytes"]:
            raise ValueError(name + ": size " + str(got_bytes) + " != expected " + str(exp["bytes"]))
        if got_md5 != exp["md5"]:
            raise ValueError(name + ": md5 mismatch\n got " + got_md5 + "\n expected " + exp["md5"])
        if got_sha != exp["sha256"]:
            raise ValueError(name + ": sha256 mismatch\n got " + got_sha + "\n expected " + exp["sha256"])
        split = name.replace("selected_features_", "").replace(".csv", "")
        df = pd.read_csv(p)
        check_split(df, split)
        frames[split] = df
        report.append(dict(file=name, split=split, rows=len(df), bytes=got_bytes,
                           md5=got_md5[:8], sha256_ok=True))
        if verbose:
            print("[OK] " + name.ljust(32) + " rows=" + str(len(df)).rjust(7)
                  + "  md5=" + got_md5[:8] + "...  sha256 OK")
    mu = frames["train"][FEATURE_COLUMNS].mean().abs().max()
    sd = (frames["train"][FEATURE_COLUMNS].std(ddof=0) - 1).abs().max()
    if verbose:
        print("[OK]" if (mu < 1e-3 and sd < 1e-2) else "[WARN]",
              "train columns max|mean|=" + format(mu, ".2e") + "  max|std-1|=" + format(sd, ".2e"),
              "(features were z-scored on the training split during preprocessing)")
    return frames, pd.DataFrame(report)

def self_test_checks(frames):
    """Negative self-test: corrupted copies of the test split must fail the checks."""
    ok = True
    bad_row = frames["test"].copy()
    bad_row.loc[0, "imf1_std"] = bad_row.loc[0, "imf1_std"] + 1e-9
    bad_cols = frames["test"][[META_COLUMNS[0]] + FEATURE_COLUMNS + META_COLUMNS[1:]]
    for label, frame in [("one value changed by 1e-9", bad_row), ("columns reordered", bad_cols)]:
        try:
            check_split(frame, "test")
            print("[FAIL] self-test: " + label + " was not detected")
            ok = False
        except ValueError as e:
            print("[OK]   self-test: " + label + " -> rejected ("
                  + str(e).splitlines()[0][:50] + ")")
    return ok



EXPECTED_ORIG = {
    "selected_features_train.csv": dict(rows=197001, bytes=61211818, md5="914e58b98eb3b0ab35dd1620a4a3d6cf"),
    "selected_features_val.csv":   dict(rows=21000,  bytes=6487178,  md5="b3979ec7f8dfe5cd4b13ecfaa10481a3"),
    "selected_features_test.csv":  dict(rows=62002,  bytes=18121973, md5="89c3312501be036fef897de92e78fbf1"),
}
def load_and_verify_11d(orig_dir, verbose=True):
    """11-dim raw-waveform features (Baseline / With M3): size, md5, row count and class counts."""
    ORIG_DIR = Path(orig_dir)
    FRAMES_ORIG = {}
    for name, exp in EXPECTED_ORIG.items():
        p = ORIG_DIR / name
        got_bytes, got_md5 = p.stat().st_size, md5_of(p)
        assert got_bytes == exp["bytes"] and got_md5 == exp["md5"], (name, got_bytes, got_md5)
        split = name.replace("selected_features_", "").replace(".csv", "")
        df = pd.read_csv(p)
        got = {CLASS_NAMES.index(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
        assert len(df) == exp["rows"] and got == EXPECTED_LABEL_COUNTS[split], (name, len(df), got)
        FRAMES_ORIG[split] = df
        if verbose:
            print("[OK] features_11d/" + name.ljust(32) + " rows=" + str(len(df)).rjust(7) + "  md5=" + got_md5[:8] + "...")
    return FRAMES_ORIG


# ------------------------------------------------------------------ preprocessing / loaders / metrics
CLASS_NAMES = ["Aging", "Fault", "Noise", "Normal"]   # label 0..3

def build_splits(frames):
    """StandardScaler fitted on the training split only."""
    train = frames["train"]
    feats = FEATURE_COLUMNS
    labels = sorted(train["label"].unique().tolist())
    l2i = {l: i for i, l in enumerate(labels)}
    scaler = StandardScaler().fit(train[feats].values.astype(np.float64))
    out = {}
    for split in ["train", "val", "test"]:
        df = frames[split]
        out["X_" + split] = scaler.transform(df[feats].values.astype(np.float64)).astype(np.float32)
        out["y_" + split] = df["label"].map(l2i).values.astype(np.int64)
        out["df_" + split] = df
    out["feats"], out["labels"], out["scaler"] = feats, labels, scaler
    return out

class TabDS(Dataset):
    """Supervised dataset returning (x, y)."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class SubTabDataset(Dataset):
    """pretrain mode returns two independently feature-masked views (v1, v2, x, y)."""
    def __init__(self, X, y, keep_ratio=0.9, mode="pretrain"):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.keep_ratio, self.mode = keep_ratio, mode
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.mode == "pretrain":
            m1 = torch.rand_like(x) < self.keep_ratio
            m2 = torch.rand_like(x) < self.keep_ratio
            return x * m1.float(), x * m2.float(), x, y
        return x, y

def make_loader(ds, shuffle, drop_last, persistent=False):
    kw = dict(batch_size=CFG["batch_size"], shuffle=shuffle, drop_last=drop_last,
              num_workers=CFG["num_workers"], pin_memory=(DEVICE.type == "cuda"))
    if kw["num_workers"] > 0:
        kw["prefetch_factor"] = 4
        kw["persistent_workers"] = persistent
    return DataLoader(ds, **kw)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    n, tot, loss_sum, ys, ps = 0, 0, 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        pred = logits.argmax(1)
        tot += (pred == y).sum().item()
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return loss_sum / n, tot / n, np.concatenate(ys), np.concatenate(ps)

def full_metrics(y_true, y_pred):
    """Accuracy, macro-F1, per-class F1 and the classification report."""
    f1 = f1_score(y_true, y_pred, average="macro")
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES))))
    rec = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES, digits=4)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)), macro_f1=float(f1),
                per_class_f1=[float(v) for v in per_class], report=rec)

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_splits_orig(frames):
    """Same preprocessing as build_splits for the 11-dim features (all non-meta columns)."""
    train = frames["train"]
    feats = [c for c in train.columns if c not in META_COLUMNS]
    labels = sorted(train["label"].unique().tolist())
    l2i = {l: i for i, l in enumerate(labels)}
    scaler = StandardScaler().fit(train[feats].values.astype(np.float64))
    out = {}
    for split in ["train", "val", "test"]:
        df = frames[split]
        out["X_" + split] = scaler.transform(df[feats].values.astype(np.float64)).astype(np.float32)
        out["y_" + split] = df["label"].map(l2i).values.astype(np.int64)
    out["feats"], out["labels"], out["scaler"] = feats, labels, scaler
    return out



# ------------------------------------------------------------------ models
class RunLog:
    """Log to file and (throttled) to stdout."""
    def __init__(self, path, echo_every=10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.echo_every = echo_every
    def __call__(self, msg, echo=True):
        line = "[" + format(time.time() - self.t0, "7.1f") + "s] " + str(msg)
        self.fh.write(line + "\n")
        self.fh.flush()
        if echo:
            print(line)
    def epoch(self, msg, ep):
        """Per-epoch log line; printed every echo_every epochs."""
        self(msg, echo=(ep % self.echo_every == 0))
    def close(self):
        self.fh.write("[" + format(time.time() - self.t0, "7.1f") + "s] DONE\n")
        self.fh.close()


class CrossViewFusionBlock(nn.Module):
    """Transformer cross-attention block (residual + FFN)."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, q, k, v):
        attn_output, _ = self.multihead_attn(query=q, key=k, value=v)
        out1 = self.norm1(q + self.dropout1(attn_output))
        return self.norm2(out1 + self.dropout2(self.ffn(out1)))


class CrossViewPhysicsFusion(nn.Module):
    """Global features (3 dims) as query, the 6-step IMF sequence as key/value."""
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers=2, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList([CrossViewFusionBlock(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layers)])
    def forward(self, endogenous_q, exogenous_kv):
        x = endogenous_q
        for layer in self.layers:
            x = layer(q=x, k=exogenous_kv, v=exogenous_kv)
        return x


class EncoderHybrid(nn.Module):
    """GLF-Encoding: 21 dims -> 3 global + 6x3 IMF sequence -> Transformer -> cross-view fusion -> 128-dim embedding."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, dropout=0.15):
        super().__init__()
        self.output_dim = output_dim
        assert input_dim == 21, "EncoderHybrid expects the 21-dim features"
        self.global_dim, self.imf_dim = 3, 3
        self.imf_steps = (input_dim - self.global_dim) // self.imf_dim
        self.feature_proj = nn.Linear(self.imf_dim, hidden_dim)
        self.pos_encoder = nn.Embedding(self.imf_steps, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4,
                                               dim_feedforward=hidden_dim * 2,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.seq_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.global_mlp = nn.Sequential(nn.Linear(self.global_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_fusion = CrossViewPhysicsFusion(embed_dim=hidden_dim, num_heads=4,
                                                   ff_dim=hidden_dim * 2, num_layers=2,
                                                   dropout=dropout)
        self.final_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        b = x.shape[0]
        global_x = x[:, :self.global_dim]
        imf_seq_x = x[:, self.global_dim:].view(b, self.imf_steps, self.imf_dim)
        seq_emb = self.feature_proj(imf_seq_x)
        pos = torch.arange(0, self.imf_steps, device=x.device).unsqueeze(0).expand(b, -1)
        seq_evolved = self.seq_transformer(seq_emb + self.pos_encoder(pos))
        global_q = self.global_mlp(global_x).unsqueeze(1)
        fused = self.cross_fusion(self.norm_q(global_q), self.norm_kv(seq_evolved))
        return self.final_proj(fused.squeeze(1))


class ProjectionHead(nn.Module):
    """Phase-1 projection head; L2-normalised 128-dim output."""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class DownstreamClassifier(nn.Module):
    """Phase-2 classifier: frozen encoder + linear head (linear probe)."""
    def __init__(self, encoder, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.classifier_head = nn.Linear(encoder.output_dim, num_classes)
    def forward(self, x):
        return self.classifier_head(self.encoder(x))


class VarConLoss(nn.Module):
    """VarCon loss: KL(p_theta || q_phi) + NLL, with tau_2 adapted to the target-class confidence."""
    def __init__(self, num_classes, embedding_dim, tau_1=0.07, epsilon=0.05, momentum=0.9):
        super().__init__()
        self.num_classes, self.tau_1, self.epsilon, self.momentum = (
            num_classes, tau_1, epsilon, momentum)
        self.register_buffer("centroids", F.normalize(torch.randn(num_classes, embedding_dim), dim=1))
    @torch.no_grad()
    def update_centroids(self, z, labels):
        z = F.normalize(z, dim=1)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_mean = F.normalize(z[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                self.centroids[c] = self.momentum * self.centroids[c] + (1 - self.momentum) * c_mean
        self.centroids = F.normalize(self.centroids, dim=1)
    def forward(self, z, labels, update_centroids=True):
        z, centroids = F.normalize(z, dim=1), F.normalize(self.centroids, dim=1)
        sim = torch.matmul(z, centroids.T)
        logits = sim / self.tau_1
        p_theta = F.softmax(logits, dim=1)
        p_c = p_theta.gather(1, labels.unsqueeze(1)).squeeze(1)
        tau_2 = ((self.tau_1 - self.epsilon) + 2 * self.epsilon * p_c).unsqueeze(1)
        q_phi = F.softmax(sim / tau_2, dim=1)
        loss_kl = F.kl_div(F.log_softmax(logits, dim=1), q_phi.detach(), reduction="batchmean")
        loss_nll = F.cross_entropy(logits, labels)
        if update_centroids and self.training:
            self.update_centroids(z, labels)
        return loss_kl + loss_nll, {"kl": float(loss_kl), "nll": float(loss_nll)}, p_theta


def n_all_params(m):
    """All parameters including frozen ones (n_params counts trainable only)."""
    return sum(p.numel() for p in m.parameters())


class PureBaselineMLP(nn.Module):
    """Single-hidden-layer MLP trained end-to-end (Baseline / With M1)."""
    def __init__(self, input_dim, num_classes, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, num_classes))
    def forward(self, x):
        return self.net(x)


class EncoderMLP(nn.Module):
    """MLP encoder used instead of the GLF-Encoding module (EncoderHybrid); output dim 128."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, dropout=0.15):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.net(x)


ABL = {   # data: "full" = 21-dim CEEMDAN features, "orig" = 11-dim raw-waveform features
    "M0": dict(data="orig", enc=None,     p1=None,     hidden=8,
               desc="Baseline: 11-dim raw-waveform features + MLP(8)"),
    "M1": dict(data="full", enc=None,     p1=None,     hidden=8,
               desc="With M1: 21-dim CEEMDAN features + MLP(8)"),
    "M2": dict(data="full", enc="hybrid", p1="ce",     hidden=None,
               desc="With M1+M2: GLF-Encoding pretrained with cross-entropy, linear probe"),
    "M3": dict(data="orig", enc="mlp",    p1="varcon", hidden=None,
               desc="With M3: 11-dim features + MLP encoder pretrained with VarCon, linear probe"),
    "M4": dict(data="full", enc="mlp",    p1="varcon", hidden=None,
               desc="With M1+M3: 21-dim features + MLP encoder pretrained with VarCon, linear probe"),
    "M5": dict(data="full", enc="hybrid", p1="varcon", hidden=None,
               desc="Proposed: GLF-Encoding pretrained with VarCon + consistency KL, linear probe"),
}

def make_encoder(kind, n_feat):
    return EncoderHybrid(n_feat) if kind == "hybrid" else EncoderMLP(n_feat)



# ------------------------------------------------------------------ training
def plot_phase(history, save_png, title_prefix, log):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title(title_prefix + " · Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle=":")
    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Acc", linestyle="--")
    axes[1].set_title(title_prefix + " · Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()
    log("curves saved to " + str(save_png), echo=False)



def train_ablation(mode, seed, save_dir, SPLITS, SPLITS_ORIG=None):
    """Two-phase training for one configuration in ABL. M5 is the proposed method."""
    pc = CFG["proposed"]
    spec = ABL[mode]
    S = SPLITS if spec["data"] == "full" else SPLITS_ORIG
    n_feat, n_cls = len(S["feats"]), len(S["labels"])
    set_seed(seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(save_dir / "console.log", echo_every=10)
    t0 = time.time()
    use_amp = (DEVICE.type == "cuda")
    log("=" * 66)
    log(mode + " | " + spec["desc"] + " | seed=" + str(seed) + " | input " + str(n_feat) + " dims | P1 "
        + (str(pc["pretrain_epochs"]) if spec["p1"] else "none") + " + P2 " + str(pc["finetune_epochs"])
        + " epoch | num_workers=" + str(CFG["num_workers"]) + " | TF32=" + str(CFG["tf32"]))
    log("hyper-parameters: " + json.dumps(pc))

    if spec["p1"] is not None:
        tr_pre = make_loader(SubTabDataset(S["X_train"], S["y_train"],
                                           pc["feature_keep_ratio"], "pretrain"),
                             shuffle=True, drop_last=True, persistent=True)
        va_pre = make_loader(SubTabDataset(S["X_val"], S["y_val"],
                                           pc["feature_keep_ratio"], "pretrain"),
                             shuffle=False, drop_last=True, persistent=True)
    tr_ft = make_loader(SubTabDataset(S["X_train"], S["y_train"], 1.0, "finetune"),
                        shuffle=True, drop_last=True, persistent=True)
    va_ft = make_loader(SubTabDataset(S["X_val"], S["y_val"], 1.0, "finetune"),
                        shuffle=False, drop_last=False, persistent=True)
    te_ft = make_loader(SubTabDataset(S["X_test"], S["y_test"], 1.0, "test"),
                        shuffle=False, drop_last=False)

    # ---------------- Phase 1 ----------------
    p1_path = save_dir / ("best_p1_" + mode + "_seed" + str(seed) + ".pth")
    if spec["p1"] is not None:
        is_vc = (spec["p1"] == "varcon")
        encoder = make_encoder(spec["enc"], n_feat).to(DEVICE)
        if is_vc:
            projector = ProjectionHead(input_dim=encoder.output_dim).to(DEVICE)
            varcon = VarConLoss(n_cls, 128, pc["tau_1"], pc["epsilon"],
                                pc["alpha"]).to(DEVICE)
        else:   # cross-entropy pretraining: linear head on unmasked inputs
            projector = nn.Linear(encoder.output_dim, n_cls).to(DEVICE)
            ce1 = nn.CrossEntropyLoss()
        opt1 = optim.AdamW(list(encoder.parameters()) + list(projector.parameters()),
                           lr=pc["lr_pretrain"], weight_decay=1e-4)
        sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=pc["pretrain_epochs"])
        amp = torch.amp.GradScaler("cuda") if use_amp else None

        p1_hist, best_val_loss, patience = [], float("inf"), 0
        log("[PHASE 1] " + ("VarCon pretraining" if is_vc else "cross-entropy pretraining"))
        for ep in range(pc["pretrain_epochs"]):
            encoder.train(); projector.train()
            if is_vc:
                varcon.train()
            ep_l, ep_a, nb = 0.0, 0.0, 0
            for v1, v2, _x, y in tr_pre:
                v1, v2, y = v1.to(DEVICE), v2.to(DEVICE), y.to(DEVICE)
                y_cat = torch.cat([y, y], dim=0)
                opt1.zero_grad(set_to_none=True)
                ctx = torch.amp.autocast("cuda", enabled=use_amp)
                with ctx:
                    if is_vc:
                        h = encoder(torch.cat([v1, v2], dim=0))
                        z = projector(h)
                        loss_varcon, _, p_theta = varcon(z, y_cat, True)
                        p1, p2 = torch.chunk(p_theta, 2, dim=0)
                        loss_cons = (F.kl_div(p1.log(), p2.detach(), reduction="batchmean")
                                     + F.kl_div(p2.log(), p1.detach(), reduction="batchmean")) / 2.0
                        loss = loss_varcon + pc["beta"] * loss_cons
                    else:
                        out = projector(encoder(_x.to(DEVICE)))
                        loss = ce1(out, y)
                if use_amp:
                    amp.scale(loss).backward()
                    amp.unscale_(opt1)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    amp.step(opt1)
                    amp.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    opt1.step()
                with torch.no_grad():
                    if is_vc:
                        pred = torch.matmul(F.normalize(z.float(), dim=1),
                                            F.normalize(varcon.centroids, dim=1).T).argmax(1)
                        ep_a += (pred == y_cat).float().mean().item()
                    else:
                        ep_a += (out.argmax(1) == y).float().mean().item()
                ep_l += loss.item()
                nb += 1
            sch1.step()

            # phase-1 validation; the checkpoint with the lowest val loss is kept
            encoder.eval(); projector.eval()
            if is_vc:
                varcon.eval()
            vl, va, vn = 0.0, 0.0, 0
            with torch.no_grad():
                for v1, v2, _x, y in va_pre:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        if is_vc:
                            z = projector(encoder(torch.cat([v1.to(DEVICE), v2.to(DEVICE)], 0)))
                            y_cat = torch.cat([y.to(DEVICE), y.to(DEVICE)], 0)
                            lv, _, p_theta = varcon(z, y_cat, False)
                            p1, p2 = torch.chunk(p_theta, 2, dim=0)
                            lc = (F.kl_div(p1.log(), p2.detach(), reduction="batchmean")
                                  + F.kl_div(p2.log(), p1.detach(), reduction="batchmean")) / 2.0
                            l = lv + pc["beta"] * lc
                            pred = torch.matmul(F.normalize(z.float(), dim=1),
                                                F.normalize(varcon.centroids, dim=1).T).argmax(1)
                            acc = (pred == y_cat).float().mean().item()
                        else:
                            out = projector(encoder(_x.to(DEVICE)))
                            yd = y.to(DEVICE)
                            l = ce1(out, yd)
                            acc = (out.argmax(1) == yd).float().mean().item()
                    vl += l.item(); va += acc; vn += 1
            rec = {"epoch": ep + 1, "train_loss": ep_l / nb, "train_acc": ep_a / nb,
                   "val_loss": vl / vn, "val_acc": va / vn}
            p1_hist.append(rec)
            log("P1 Ep" + format(ep + 1, "03d") + "/" + str(pc["pretrain_epochs"])
                + " | train_loss=" + format(rec["train_loss"], ".4f")
                + " train_acc=" + format(rec["train_acc"], ".4f")
                + " | val_loss=" + format(rec["val_loss"], ".4f")
                + " val_acc=" + format(rec["val_acc"], ".4f"), echo=(ep % 10 == 0 or ep == pc["pretrain_epochs"] - 1))
            if rec["val_loss"] < best_val_loss:
                best_val_loss, patience = rec["val_loss"], 0
                torch.save({"encoder": encoder.state_dict()}, p1_path)
            else:
                patience += 1
                if patience >= 150:
                    log("P1 early stop at epoch " + str(ep + 1))
                    break
        pd.DataFrame(p1_hist).to_csv(save_dir / ("p1_training_metrics_seed" + str(seed) + ".csv"), index=False)
        plot_phase(p1_hist, save_dir / ("p1_training_curves_seed" + str(seed) + ".png"), mode + " · Phase 1", log)
        log("P1 done, best val_loss=" + format(best_val_loss, ".4f"))

    # ---------------- Phase 2 ----------------
    if spec["p1"] is not None:
        encoder.load_state_dict(torch.load(p1_path, map_location=DEVICE)["encoder"])
        model2 = DownstreamClassifier(encoder, n_cls,
                                      freeze_encoder=pc["freeze_encoder"]).to(DEVICE)
    else:
        model2 = PureBaselineMLP(n_feat, n_cls, spec["hidden"]).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt2 = optim.AdamW([p for p in model2.parameters() if p.requires_grad], lr=pc["lr_finetune"])
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=pc["finetune_epochs"])
    p2_hist, best_v2 = [], 0.0
    p2_path = save_dir / ("best_p2_" + mode + "_seed" + str(seed) + ".pth")
    log("[PHASE 2] " + ("linear probe (freeze_encoder=" + str(pc["freeze_encoder"]) + ")"
                        if spec["p1"] is not None else "supervised MLP training"))
    for ep in range(pc["finetune_epochs"]):
        model2.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in tr_ft:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt2.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model2(x)
                loss = crit(logits, y)
            loss.backward(); opt2.step()
            tl += loss.item() * len(y)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(y)
        sch2.step()
        v_loss, v_acc, _, _ = evaluate(model2, va_ft, crit)
        rec = {"epoch": ep + 1, "train_loss": tl / tn, "train_acc": tc / tn,
               "val_loss": v_loss, "val_acc": v_acc}
        p2_hist.append(rec)
        log("P2 Ep" + format(ep + 1, "02d") + "/" + str(pc["finetune_epochs"])
            + " | train_acc=" + format(rec["train_acc"], ".4f")
            + " | val_loss=" + format(v_loss, ".4f") + " val_acc=" + format(v_acc, ".4f"),
            echo=(ep % 10 == 0 or ep == pc["finetune_epochs"] - 1))
        if v_acc > best_v2:
            best_v2 = v_acc
            torch.save(model2.state_dict(), p2_path)
    pd.DataFrame(p2_hist).to_csv(save_dir / ("p2_training_metrics_seed" + str(seed) + ".csv"), index=False)
    plot_phase(p2_hist, save_dir / ("p2_training_curves_seed" + str(seed) + ".png"), mode + " · Phase 2", log)

    # ---------------- test (evaluated once) ----------------
    model2.load_state_dict(torch.load(p2_path, map_location=DEVICE))
    _, t_acc, y_true, y_pred = evaluate(model2, te_ft, crit)
    m = full_metrics(y_true, y_pred)
    pred_csv = save_dir / ("test_predictions_" + mode + "_seed" + str(seed) + ".csv")
    pd.DataFrame({"true_label": y_true, "pred_label": y_pred}).to_csv(pred_csv, index=False)
    summary = {"mode": mode, "desc": spec["desc"], "seed": seed, "input_dim": n_feat,
               "params": n_all_params(model2), "trainable_params": n_params(model2),
               "test_correct": int((y_true == y_pred).sum()), "test_total": int(len(y_true)),
               "test_acc": round(t_acc, 6), "macro_f1": round(m["macro_f1"], 6),
               "best_val_acc": round(best_v2, 6),
               "epochs": (pc["pretrain_epochs"] if spec["p1"] else 0) + pc["finetune_epochs"],
               "seconds": round(time.time() - t0, 1), "per_class_f1": m["per_class_f1"],
               "pred_md5": md5_of(pred_csv)}
    (save_dir / ("summary_" + mode + "_seed" + str(seed) + ".json")).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(mode + " test: Acc " + format(t_acc, ".4f") + " | Macro-F1 " + format(m["macro_f1"], ".4f"))
    log("per-class F1: " + str([round(v, 4) for v in m["per_class_f1"]]))
    log("\n" + m["report"])
    log.close()
    return summary

def prepare(data_dir, orig_dir=None, verbose=True):
    """Verify the data files and return (SPLITS, SPLITS_ORIG). SPLITS_ORIG is None when orig_dir is not given."""
    if CFG["tf32"]:
        torch.set_float32_matmul_precision("high")
    frames, report = load_and_verify(data_dir, verbose=verbose)
    assert self_test_checks(frames), "data verification self-test failed"
    splits = build_splits(frames)
    splits_orig = None
    if orig_dir is not None:
        splits_orig = build_splits_orig(load_and_verify_11d(orig_dir, verbose=verbose))
        assert splits_orig["labels"] == splits["labels"]
    return splits, splits_orig
