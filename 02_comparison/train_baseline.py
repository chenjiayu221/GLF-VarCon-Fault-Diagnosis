# ============================================================
# configuration
# ============================================================
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

HERE      = Path(__file__).resolve().parent
_COMMON   = str(HERE.parent / 'common')
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
import core as _C

DATA_DIR  = Path(os.environ.get('DATA_DIR', HERE.parent / 'data' / 'features_21d'))
OUT_DIR   = Path(os.environ.get('OUT_DIR', HERE / 'results'))
TRAIN_CSV = DATA_DIR / 'selected_features_train.csv'
VAL_CSV   = DATA_DIR / 'selected_features_val.csv'
TEST_CSV  = DATA_DIR / 'selected_features_test.csv'

META = ['sample_id', 'label', 'window_number', 'source_folder', 'source_sample']

SEED, EPOCHS, BATCH = 163, 100, 2048   # same seed as Proposed
LR, WD      = 1e-3, 1e-4
NODE_LR     = 2e-3
CKPT_BEST   = True          # evaluate with the best-validation weights

# device_count() is checked too: with CUDA_VISIBLE_DEVICES="" is_available() can still be True while no GPU is visible
DEVICE = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'

RUN_NAMES = ['TabNet', 'FT-Transformer', 'AutoInt', 'DCN-V2', 'ExcelFormer', 'Tab-ResNet', 'NODE']

OUT_DIR.mkdir(parents=True, exist_ok=True)
print('device =', DEVICE, '| epochs =', EPOCHS, '| batch =', BATCH, '| seed =', SEED)
print('runs   =', RUN_NAMES)

# ============================================================
# data: 21 features + 4 classes; verify file integrity and fit StandardScaler on train only
# ============================================================
_frames, _ = _C.load_and_verify(DATA_DIR, verbose=False)
_bad_row = _frames['test'].copy()
_bad_row.loc[0, 'imf1_std'] = _bad_row.loc[0, 'imf1_std'] + 1e-9
_bad_cols = _frames['test'][_C.META_COLUMNS[:1] + _C.FEATURE_COLUMNS + _C.META_COLUMNS[1:]]
for _bad in (_bad_row, _bad_cols):
    try:
        _C.check_split(_bad, 'test')
    except ValueError:
        pass
    else:
        raise AssertionError('data verification negative self-test failed')

df_tr = _frames['train']
df_va = _frames['val']
df_te = _frames['test']

FEATS  = [c for c in df_tr.columns if c not in META]
LABELS = sorted(df_tr['label'].unique())
l2i    = {lab: i for i, lab in enumerate(LABELS)}

y_tr = np.array([l2i[l] for l in df_tr['label']], dtype='int64')
y_va = np.array([l2i[l] for l in df_va['label']], dtype='int64')
y_te = np.array([l2i[l] for l in df_te['label']], dtype='int64')

scaler = StandardScaler().fit(df_tr[FEATS].values)
X_tr = scaler.transform(df_tr[FEATS].values).astype('float32')
X_va = scaler.transform(df_va[FEATS].values).astype('float32')
X_te = scaler.transform(df_te[FEATS].values).astype('float32')

print('features =', len(FEATS), '| classes =', LABELS)
print('train/val/test =', X_tr.shape, X_va.shape, X_te.shape)
print('train label counts =', np.bincount(y_tr))

# ============================================================
# the seven baseline models (compact PyTorch adaptations sized for 21D features)
# Note: these are lightweight implementations adopting core structural motifs from
# their respective papers (1k-19k parameters) rather than full heavy original packages.
# ============================================================
class TabNetModel(nn.Module):
    """Compact TabNet adaptation: single-step sigmoid feature mask + MLP backbone."""

    def __init__(self, input_dim, num_classes, hidden=(32, 16), dropout=0.2):
        super().__init__()
        self.input_dim = input_dim

        # 1) feature attention mask
        self.att_transformer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),
        )

        # 2) decision backbone
        self.feat_transformers = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        return self.feat_transformers(x * self.att_transformer(x))

class FTTransformerModel(nn.Module):
    """Compact FT-Transformer adaptation: linear feature tokens + 1-layer Transformer encoder."""
    def __init__(self, input_dim, num_classes, d_token=8, n_heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token

        # one linear token-projection layer per feature dimension
        self.token_embeddings = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(input_dim)
        ])

        # feature-level self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            batch_first=True, activation='gelu', dropout=0.15
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # downstream classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim * d_token),
            nn.Linear(input_dim * d_token, 16),
            nn.GELU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # 1. tokenise the features: each scalar feature becomes a [B, 1, d_token] embedding
        tokens = [self.token_embeddings[i](x[:, i].unsqueeze(1)).unsqueeze(1) for i in range(self.input_dim)]
        # 2. concatenate them into a sequence of feature steps: [B, input_dim, d_token]
        tokens = torch.cat(tokens, dim=1)
        # 3. evolve with all-feature multi-head self-attention
        trans_out = self.transformer(tokens)
        # 4. flatten and classify
        return self.mlp_head(trans_out.view(batch_size, -1))

class AutoIntModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_token=8, n_heads=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        # feature embedding layer
        self.token_embeddings = nn.ModuleList([nn.Linear(1, d_token) for _ in range(input_dim)])
        # explicit automatic feature-interaction multi-head attention layer
        self.interacting_layer = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True)
        self.w_res = nn.Linear(d_token, d_token)  # keep the residual path

        self.mlp_head = nn.Sequential(
            nn.Linear(input_dim * d_token, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        tokens = [self.token_embeddings[i](x[:, i].unsqueeze(1)).unsqueeze(1) for i in range(self.input_dim)]
        tokens = torch.cat(tokens, dim=1)  # [B, input_dim, d_token]

        # compute the explicit interaction
        attn_out, _ = self.interacting_layer(tokens, tokens, tokens)
        res_out = attn_out + self.w_res(tokens)
        res_out = F.relu(res_out)

        return self.mlp_head(res_out.reshape(batch_size, -1))

class CrossNetworkV2(nn.Module):
    """Explicit vertical feature-interaction network"""
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x.unsqueeze(-1) # [B, input_dim, 1]
        xl = x0
        for i in range(self.num_layers):
            # xl+1 = x0 .* (W_l * xl + b_l) + xl
            xl_w = torch.matmul(self.weights[i], xl) + self.biases[i]
            xl = x0 * xl_w + xl
        return xl.squeeze(-1)

class DCNV2Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 1. explicit cross backbone
        self.cross_net = CrossNetworkV2(input_dim, num_layers=2)
        # 2. parallel deep backbone
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )
        # 3. aggregation head
        self.final_head = nn.Linear(input_dim * 2, num_classes)

    def forward(self, x):
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        # parallel concatenation
        combined = torch.cat([cross_out, deep_out], dim=1)
        return self.final_head(combined)

class ExcelFormerAttention(nn.Module):
    """Compact ExcelFormer-inspired self-attention block (full feature attention + Tanh FFN)."""
    def __init__(self, d_token, n_heads=2, dropout=0.1):
        super().__init__()
        self.q_linear = nn.Linear(d_token, d_token)
        self.k_linear = nn.Linear(d_token, d_token)
        self.v_linear = nn.Linear(d_token, d_token)
        self.n_heads = n_heads
        self.d_head = d_token // n_heads

        # gated feed-forward network
        self.gated_ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.Tanh(), # non-linear activation
            nn.Linear(d_token * 2, d_token),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_token)

    def forward(self, x):
        # x shape: [B, input_dim, d_token]
        B, N, D = x.shape
        res = x

        # project to Q, K, V
        q = self.q_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product self-attention across the features
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)

        # gated feed-forward + residual
        x = self.layer_norm(res + context)
        out = x + self.gated_ffn(x)
        return out

class ExcelFormerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_token=16, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        # per-token projection layer for the continuous features
        self.token_embeddings = nn.ModuleList([nn.Linear(1, d_token) for _ in range(input_dim)])

        # stacked feature-interaction layers
        self.blocks = nn.ModuleList([ExcelFormerAttention(d_token) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim * d_token),
            nn.Linear(input_dim * d_token, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # flatten the embeddings: [B, input_dim, d_token]
        tokens = [self.token_embeddings[i](x[:, i].unsqueeze(1)).unsqueeze(1) for i in range(self.input_dim)]
        tokens = torch.cat(tokens, dim=1)

        # evolve through the continuous-manifold layers
        for block in self.blocks:
            tokens = block(tokens)

        return self.mlp_head(tokens.reshape(batch_size, -1))

class TabResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.dropout(self.gelu(self.bn1(self.linear1(x))))
        out = self.bn2(self.linear2(out))
        return self.gelu(out + residual)


class TabResNetModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_hidden=64, num_blocks=2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, d_hidden), nn.BatchNorm1d(d_hidden), nn.GELU())
        self.blocks = nn.ModuleList([TabResNetBlock(d_hidden) for _ in range(num_blocks)])
        self.out_head = nn.Linear(d_hidden, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_head(x)

class Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        tau_min = input.min(dim=dim, keepdim=True)[0] - 1.0
        tau_max = input.max(dim=dim, keepdim=True)[0]
        for _ in range(12):
            tau = (tau_min + tau_max) / 2
            out = torch.clamp(input - tau, min=0) ** 1.5
            if out.sum(dim=dim, keepdim=True).mean() > 1.0:
                tau_min = tau
            else:
                tau_max = tau
        out = torch.clamp(input - tau_max, min=0) ** 1.5
        out = out / out.sum(dim=dim, keepdim=True)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input = grad_output * torch.sqrt(out)
        grad_input = grad_input - (grad_input.sum(dim=ctx.dim, keepdim=True) / torch.sqrt(out).sum(dim=ctx.dim,
                                                                                                   keepdim=True)) * torch.sqrt(
            out)
        return grad_input, None


def entmax15(input, dim=-1):
    return Entmax15Function.apply(input, dim)


class ODTreeBlock(nn.Module):
    def __init__(self, input_dim, num_trees=16, depth=4):
        super().__init__()
        self.depth = depth
        self.num_trees = num_trees
        self.gate_weights = nn.Parameter(torch.randn(depth, num_trees, input_dim))
        self.response_weights = nn.Parameter(torch.randn(num_trees, 2 ** depth))

    def forward(self, x):
        batch_size = x.shape[0]
        # flat_x shape: [batch_size, 1, input_dim]
        flat_x = x.view(batch_size, 1, -1)

        # flat_x corresponds to 'boi' (b=batch_size, o=1, i=input_dim)
        # gate_weights corresponds to 'dwi' (d=depth, w=num_trees, i=input_dim)
        # summing over i gives gates with shape [depth, batch_size, num_trees] = 'dbw'
        gates = torch.einsum('boi,dwi->dbw', flat_x, entmax15(self.gate_weights, dim=-1))

        # 2. simulate hashed decision-tree paths
        tree_probs = torch.ones(batch_size, self.num_trees, 1, device=x.device)
        for d in range(self.depth):
            g = gates[d].unsqueeze(-1)  # [B, num_trees, 1]
            tree_probs = torch.cat([tree_probs * g, tree_probs * (1 - g)], dim=-1)

        # tree_probs reshaped to [batch_size, num_trees, 2**depth] -> 'bwd'
        tree_probs = tree_probs.view(batch_size, self.num_trees, -1)

        # response_weights shape: [num_trees, 2**depth] -> 'wd'
        # after both matmuls responses has shape [batch_size, num_trees] -> 'bw'
        responses = torch.einsum('bwd,wd->bw', tree_probs, torch.tanh(self.response_weights))
        return responses


class NODEModel(nn.Module):
    """Compact NODE adaptation: single-layer differentiable oblivious decision trees (entmax15)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.tree_layer = ODTreeBlock(input_dim, num_trees=32, depth=3)
        self.linear_head = nn.Linear(32, num_classes)

    def forward(self, x):
        tree_out = self.tree_layer(x)
        return self.linear_head(tree_out)

# ============================================================
# builders and parameter counts
# ============================================================
BUILDERS = {
    'TabNet':          lambda: TabNetModel(len(FEATS), len(LABELS), hidden=(16, 8)),
    'FT-Transformer':  lambda: FTTransformerModel(len(FEATS), len(LABELS)),
    'AutoInt':         lambda: AutoIntModel(len(FEATS), len(LABELS), d_token=6),
    'DCN-V2':          lambda: DCNV2Model(len(FEATS), len(LABELS)),
    'ExcelFormer':     lambda: ExcelFormerModel(len(FEATS), len(LABELS)),
    'Tab-ResNet':      lambda: TabResNetModel(len(FEATS), len(LABELS)),
    'NODE':            lambda: NODEModel(len(FEATS), len(LABELS)),
}

def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print('%-16s %12s' % ('model', 'params'))
for _n in RUN_NAMES:
    print('%-16s %12d' % (_n, n_params(BUILDERS[_n]())))

# ============================================================
# Dataset / evaluation / single-model training
# ============================================================
class TabDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def make_loader(X, y, shuffle, drop_last):
    return DataLoader(TabDS(X, y), batch_size=BATCH, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0)

tr_loader = make_loader(X_tr, y_tr, True, True)
tr_eval   = make_loader(X_tr, y_tr, False, False)   # all 197,001 rows, used for train_acc
va_loader = make_loader(X_va, y_va, False, False)
te_loader = make_loader(X_te, y_te, False, False)

crit = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot = correct = 0
    loss = 0.0
    probs = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logit = model(x)
        loss += crit(logit, y).item() * x.size(0)
        correct += (logit.argmax(1) == y).sum().item()
        tot += y.size(0)
        probs.append(F.softmax(logit, dim=1).cpu().numpy())
    return loss / tot, correct / tot, np.concatenate(probs, 0)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one(name, seed=SEED):
    save_dir = OUT_DIR / ('Experiment_' + name.replace('-', ''))
    save_dir.mkdir(parents=True, exist_ok=True)
    done = save_dir / 'summary.json'
    if done.exists():
        s = json.loads(done.read_text(encoding='utf-8'))
        print('%-16s skip (done)  acc=%.4f%%' % (name, s['test_acc_percent']))
        return s

    set_seed(seed)
    model = BUILDERS[name]().to(DEVICE)
    lr = NODE_LR if name == 'NODE' else LR
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ckpt = save_dir / ('best_' + name.replace('-', '') + '.pth')

    hist, best_acc, best_ep = [], -1.0, 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, seen, run_correct = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (logit.argmax(1) == y).sum().item()
            seen += x.size(0)
        sched.step()

        v_loss, v_acc, _ = evaluate(model, va_loader)
        hist.append({'epoch': ep, 'train_loss': run_loss / seen, 'train_acc': run_correct / seen,
                     'val_loss': v_loss, 'val_acc': v_acc})

        if CKPT_BEST and v_acc > best_acc:
            best_acc, best_ep = v_acc, ep
            torch.save(model.state_dict(), ckpt)

        if ep % 10 == 0 or ep == 1:
            print('%-16s ep %3d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f'
                  % (name, ep, EPOCHS, run_loss / seen, run_correct / seen, v_loss, v_acc))

    pd.DataFrame(hist).to_csv(save_dir / 'training_metrics.csv', index=False)

    # evaluate with the best-validation weights; the test set is evaluated once
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    tr_loss_ev, tr_acc_ev, _ = evaluate(model, tr_eval)   # best weights + the full training set
    _, t_acc, probs = evaluate(model, te_loader)
    pred = probs.argmax(1)
    f1 = f1_score(y_te, pred, average='macro')

    pred_df = pd.DataFrame({'true_label': y_te, 'pred_label': pred})
    for i, lab in enumerate(LABELS):
        pred_df['prob_class_' + lab] = probs[:, i]
    pred_df.to_csv(save_dir / ('test_predictions_' + name + '.csv'), index=False)

    summary = {
        'model': name,
        'params': n_params(model),
        'seed': seed,
        'epochs': EPOCHS,
        'best_val_acc': round(best_acc, 6),
        'best_val_epoch': best_ep,
        'train_acc_eval': round(tr_acc_ev, 6),
        'train_loss_eval': round(tr_loss_ev, 6),
        'test_correct': int((pred == y_te).sum()),
        'test_total': int(len(y_te)),
        'test_acc_percent': round(t_acc * 100, 6),
        'macro_f1': round(f1, 6),
        'seconds': round(time.time() - t0, 1),
    }
    done.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('%-16s DONE  train_acc=%.4f%%  test acc = %.4f%% (%d/%d)  macroF1=%.4f  %.0fs'
          % (name, tr_acc_ev * 100, t_acc * 100, int((pred == y_te).sum()), len(y_te), f1, time.time() - t0))
    return summary

# ============================================================
# entry point for one model: python train_baseline.py <model_name>
# ============================================================
if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    r = train_one(name)
    keys = ["model", "params", "seed", "epochs", "best_val_acc", "best_val_epoch",
            "train_acc_eval", "train_loss_eval", "test_correct", "test_total",
            "test_acc_percent", "macro_f1", "seconds"]
    print("DONE " + name + " " + json.dumps({k: r.get(k) for k in keys}), flush=True)
