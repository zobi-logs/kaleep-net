
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score
from kan_layer import KANLayer
from flash_attention import FlashAttention

# ── reproducibility ──────────────────────────────────────────────
torch.backends.cudnn.enabled       = False
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ════════════════════════════════════════════════════════════════
# 1.  Dataset
# ════════════════════════════════════════════════════════════════
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ════════════════════════════════════════════════════════════════
# 2.  KAleep-Net  (paper-aligned)
# ════════════════════════════════════════════════════════════════
class KAleepNet(nn.Module):
    """
    Multi-Spectral Feature Pipeline + Temporal Attention Network

    FSF branch  (fine-scale, k=3):
        Conv1(1→64)  → KAN1(64→64)   → pool(3000→1500) → drop(0.25)
        Conv2(64→128)→ KAN2(128→128) →
        Conv3(128→128)→KAN3(128→128) → pool(1500→750)
        output: (B, 750, 128)

    CSF branch  (coarse-scale, k=5):
        Conv1(1→32)  → KAN1(32→32)   → pool(3000→1500) → drop(0.25)
        Conv2(32→64) → KAN2(64→64)   →
        Conv3(64→64) → KAN3(64→64)   → pool(1500→750)
        output: (B, 750, 64)

    Concat → (B, 750, 192)
    BiLSTM  (hidden=128, layers=2, bidirectional) → (B, 750, 256)
    TSN dropout 0.3
    FlashAttention (embed=256, block_size=5) → (B, 750, 256)
    Mean pool → (B, 256)
    Dropout 0.5 → FC(256→5)
    """
    def __init__(self, n_basis=16, lstm_hidden=128):
        super().__init__()

        # ── FSF (Fine-Scale Feature) block ──────────────────────
        self.fsf_conv1 = nn.Conv1d(1,   64,  kernel_size=3, padding=1)
        self.fsf_kan1  = KANLayer(64,   64,  n_basis)

        self.fsf_conv2 = nn.Conv1d(64,  128, kernel_size=3, padding=1)
        self.fsf_kan2  = KANLayer(128,  128, n_basis)

        self.fsf_conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fsf_kan3  = KANLayer(128,  128, n_basis)

        self.fsf_drop  = nn.Dropout(0.25)   # after first pool

        # ── CSF (Coarse-Scale Feature) block ────────────────────
        self.csf_conv1 = nn.Conv1d(1,   32,  kernel_size=5, padding=2)
        self.csf_kan1  = KANLayer(32,   32,  n_basis)

        self.csf_conv2 = nn.Conv1d(32,  64,  kernel_size=5, padding=2)
        self.csf_kan2  = KANLayer(64,   64,  n_basis)

        self.csf_conv3 = nn.Conv1d(64,  64,  kernel_size=5, padding=2)
        self.csf_kan3  = KANLayer(64,   64,  n_basis)

        self.csf_drop  = nn.Dropout(0.25)   # after first pool

        # shared max-pool (stride=2)
        self.pool = nn.MaxPool1d(2)

        # ── Temporal Sequencing Network (BiLSTM) ────────────────
        # input: 128+64 = 192 features at 750 time steps
        self.bilstm = nn.LSTM(
            input_size=192,
            hidden_size=lstm_hidden,
            num_layers=2,           # paper Table II: 2 layers
            batch_first=True,
            bidirectional=True,
            dropout=0.3             # inter-layer dropout (active when num_layers>1)
        )
        self.bilstm.flatten_parameters = lambda *a, **kw: None  # cuDNN safety

        self.tsn_drop = nn.Dropout(0.3)   # after BiLSTM output

        # ── Flash Attention ─────────────────────────────────────
        # embed_dim = 2 * lstm_hidden = 256  (paper Table II: d_a=256)
        self.flash = FlashAttention(embed_dim=2 * lstm_hidden, block_size=5)

        # ── Classifier ──────────────────────────────────────────
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * lstm_hidden, 5)

    # ── FSF forward  (Eq. 1-2) ──────────────────────────────────
    def _fsf(self, x):
        # x: (B, 1, 3000)
        z = F.relu(self.fsf_conv1(x))
        z = z + self.fsf_kan1(z.permute(0, 2, 1)).permute(0, 2, 1)   # residual
        z = self.pool(z)                # 3000 → 1500
        z = self.fsf_drop(z)

        z = F.relu(self.fsf_conv2(z))
        z = z + self.fsf_kan2(z.permute(0, 2, 1)).permute(0, 2, 1)

        z = F.relu(self.fsf_conv3(z))
        z = z + self.fsf_kan3(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.pool(z)                # 1500 → 750  (after K3, per Eq.1)
        return z.permute(0, 2, 1)       # (B, 750, 128)

    # ── CSF forward  (Eq. 4-5) ──────────────────────────────────
    def _csf(self, x):
        # x: (B, 1, 3000)
        z = F.relu(self.csf_conv1(x))
        z = z + self.csf_kan1(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.pool(z)                # 3000 → 1500
        z = self.csf_drop(z)

        z = F.relu(self.csf_conv2(z))
        z = z + self.csf_kan2(z.permute(0, 2, 1)).permute(0, 2, 1)

        z = F.relu(self.csf_conv3(z))
        z = z + self.csf_kan3(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.pool(z)                # 1500 → 750  (after K3, per Eq.4)
        return z.permute(0, 2, 1)       # (B, 750, 64)

    def forward(self, x):
        # x: (B, 3000, 1)
        x = x.permute(0, 2, 1)         # → (B, 1, 3000)

        F_fine   = self._fsf(x)         # (B, 750, 128)
        F_coarse = self._csf(x)         # (B, 750, 64)

        F_cat = torch.cat([F_fine, F_coarse], dim=-1)   # (B, 750, 192)  Eq.6

        H, _ = self.bilstm(F_cat)       # (B, 750, 256)
        H = self.tsn_drop(H)

        H = self.flash(H)               # (B, 750, 256)

        pooled = H.mean(dim=1)          # (B, 256)
        logits = self.fc(self.dropout(pooled))
        return logits


# ════════════════════════════════════════════════════════════════
# 3.  Load data
# ════════════════════════════════════════════════════════════════
npz_folder = './data'   # ← change this path to your EDF-20 .npz directory
npz_files  = sorted(glob.glob(os.path.join(npz_folder, "*.npz")))
assert len(npz_files) > 0, f"No .npz files found in {npz_folder}"

X_list, y_list, subj_list = [], [], []
for subj_idx, fn in enumerate(npz_files):
    dat = np.load(fn)
    X_list.append(dat['x'][:, :3000, :])   # (n, 3000, 1)
    y_list.append(dat['y'])                 # (n,)
    subj_list.extend([subj_idx] * len(dat['y']))

X        = np.concatenate(X_list, axis=0)
y        = np.concatenate(y_list, axis=0)
subjects = np.array(subj_list)
print(f"Total epochs: {X.shape[0]}  |  Subjects: {len(np.unique(subjects))}")


# ════════════════════════════════════════════════════════════════
# 4.  10-fold subject-wise cross-validation  (paper Table V)
# ════════════════════════════════════════════════════════════════
NUM_FOLDS  = 10
BATCH_SIZE = 64
MAX_EPOCHS = 100
ES_PATIENCE = 10     # early stopping patience
LR_PATIENCE =  3     # ReduceLROnPlateau patience

gkf = GroupKFold(n_splits=NUM_FOLDS)
all_results = []

for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects)):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold+1}/{NUM_FOLDS}")
    print(f"{'='*55}")

    X_trainval  = X[trainval_idx]
    y_trainval  = y[trainval_idx]
    grp_trainval = subjects[trainval_idx]
    X_test_f    = X[test_idx]
    y_test_f    = y[test_idx]

    # inner val split (~10% of trainval, subject-wise)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=fold)
    tr_idx, va_idx = next(gss_inner.split(X_trainval, y_trainval, groups=grp_trainval))

    X_train = X_trainval[tr_idx];  y_train = y_trainval[tr_idx]
    X_val   = X_trainval[va_idx];  y_val   = y_trainval[va_idx]

    # ── subject-leak guard (uses correct per-fold arrays) ──────
    assert set(grp_trainval[tr_idx]) & set(grp_trainval[va_idx])  == set(), "Train/val subject leak"
    assert set(grp_trainval[tr_idx]) & set(subjects[test_idx])    == set(), "Train/test subject leak"
    assert set(grp_trainval[va_idx]) & set(subjects[test_idx])    == set(), "Val/test subject leak"

    # ── class-weighted loss per Eq.22  w_c = 1/log(1+f_c) ─────
    counts  = np.bincount(y_train, minlength=5)
    w_np    = 1.0 / np.log(1.0 + counts)
    weights = torch.tensor(w_np, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── loaders ─────────────────────────────────────────────────
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(EEGDataset(X_test_f, y_test_f), batch_size=BATCH_SIZE)

    # ── model, optimiser, scheduler ────────────────────────────
    model     = KAleepNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=LR_PATIENCE, verbose=False)

    best_val_f1  = 0.0
    no_improve   = 0
    best_path    = f"kaleep_fold{fold+1}.pth"

    for epoch in range(1, MAX_EPOCHS + 1):

        # ── train ───────────────────────────────────────────────
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ────────────────────────────────────────────
        model.eval()
        val_losses, yt, yp = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                yp.extend(torch.argmax(logits, 1).cpu().numpy())
                yt.extend(yb.cpu().numpy())

        val_loss = float(np.mean(val_losses))
        val_f1   = f1_score(yt, yp, average='macro')
        scheduler.step(val_loss)   # scheduler on val_loss (paper)

        print(f"  Ep {epoch:03d} | tr_loss={np.mean(train_losses):.4f} "
              f"| val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        if no_improve >= ES_PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # ── test this fold ──────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            yp.extend(torch.argmax(model(xb), 1).cpu().numpy())
            yt.extend(yb.cpu().numpy())

    acc   = np.mean(np.array(yt) == np.array(yp)) * 100
    f1    = f1_score(yt, yp, average='macro') * 100
    kappa = cohen_kappa_score(yt, yp) * 100

    print(f"\n  Fold {fold+1} results → Acc={acc:.1f}%  F1={f1:.1f}%  κ={kappa:.1f}%")
    print(classification_report(yt, yp, target_names=['Wake','N1','N2','N3','REM'], digits=4))
    all_results.append({'fold': fold+1, 'acc': acc, 'f1': f1, 'kappa': kappa})


# ════════════════════════════════════════════════════════════════
# 5.  Aggregate results
# ════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  10-FOLD CROSS-VALIDATION SUMMARY")
print("="*55)
for r in all_results:
    print(f"  Fold {r['fold']:2d}: Acc={r['acc']:.1f}%  F1={r['f1']:.1f}%  κ={r['kappa']:.1f}%")

accs   = [r['acc']   for r in all_results]
f1s    = [r['f1']    for r in all_results]
kappas = [r['kappa'] for r in all_results]
print(f"\n  Avg ± Std:")
print(f"    Accuracy : {np.mean(accs):.1f}% ± {np.std(accs):.2f}%")
print(f"    Macro F1 : {np.mean(f1s):.1f}% ± {np.std(f1s):.2f}%")
print(f"    Cohen's κ: {np.mean(kappas):.1f}% ± {np.std(kappas):.2f}%")
