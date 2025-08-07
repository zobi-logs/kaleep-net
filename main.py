import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from kan_layer import KANLayer
from flash_attention import FlashAttention

# DATASET CLASS
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# MODEL 
class KAleepNet(nn.Module):
    def __init__(self, n_classes=5, lstm_hidden=128, time_steps=10, flash_block=2):
        super().__init__()
       
        self.fine_cnn1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.fine_cnn2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fine_pool = nn.MaxPool1d(2)
        self.fine_kan = KANLayer(128, 128)
       
        self.coarse_cnn1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.coarse_cnn2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.coarse_pool = nn.MaxPool1d(2)
        self.coarse_kan = KANLayer(64, 64)
       
        self.dropout = nn.Dropout(0.5)
        self.time_steps = time_steps
        self.bilstm = nn.LSTM(192//time_steps, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.flash_attn = FlashAttention(embed_dim=2*lstm_hidden, block_size=flash_block)
        self.fc = nn.Linear(2*lstm_hidden, n_classes)

    def forward(self, x):
        # x: (batch, 3000, 1) -> (batch, 1, 3000)
        x = x.permute(0, 2, 1)
        # Fine
        fine = self.fine_cnn1(x)
        fine = F.relu(self.fine_pool(fine))
        fine = self.fine_cnn2(fine)
        fine = F.relu(self.fine_pool(fine))
        fine = fine.permute(0, 2, 1) # (batch, seq, channels)
        fine = self.fine_kan(fine)
        fine = fine.mean(dim=1) # Global pool
        # Coarse
        coarse = self.coarse_cnn1(x)
        coarse = F.relu(self.coarse_pool(coarse))
        coarse = self.coarse_cnn2(coarse)
        coarse = F.relu(self.coarse_pool(coarse))
        coarse = coarse.permute(0, 2, 1)
        coarse = self.coarse_kan(coarse)
        coarse = coarse.mean(dim=1)
        # Merge
        merged = torch.cat([fine, coarse], dim=1) # (batch, 192)
        merged = self.dropout(merged)
        # Reshape for LSTM (batch, time_steps, feat_per_step)
        batch = merged.shape[0]
        feat_per_step = merged.shape[1] // self.time_steps
        merged_seq = merged.view(batch, self.time_steps, feat_per_step)
        bilstm_out, _ = self.bilstm(merged_seq) # (batch, time_steps, 2*lstm_hidden)
        # Flash Attention
        fa_out = self.flash_attn(bilstm_out)  n)
        pooled = fa_out.mean(dim=1)           
        logits = self.fc(pooled)
        return logits

#LOAD DATA 



npz_folder = './data'
npz_files  = sorted(glob.glob(os.path.join(npz_folder, "*.npz")))

X_list, y_list, subject_list = [], [], []
for subj_idx, fn in enumerate(npz_files):
    dat = np.load(fn)
    x   = dat['x'][:, :3000, :]      # (n_samples,3000,1)
    y   = dat['y']                   # (n_samples,)
    n   = x.shape[0]

    X_list.append(x)
    y_list.append(y)
    subject_list.extend([subj_idx]*n)

# final arrays
X        = np.concatenate(X_list, axis=0)
y        = np.concatenate(y_list, axis=0)
subjects = np.array(subject_list)


from sklearn.model_selection import GroupShuffleSplit


gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
trainval_idx, test_idx = next(gss.split(X, y, groups=subjects))

X_trainval, y_trainval, grp_trainval = X[trainval_idx], y[trainval_idx], subjects[trainval_idx]
X_test,     y_test     = X[test_idx],     y[test_idx]


gss_val = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=grp_trainval))

X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
X_val,   y_val   = X_trainval[val_idx],   y_trainval[val_idx]


assert set(subjects[train_idx]) & set(subjects[val_idx]) == set()
assert set(subjects[train_idx]) & set(subjects[test_idx]) == set()
assert set(subjects[val_idx])   & set(subjects[test_idx]) == set()


train_ds = EEGDataset(X_train, y_train)
val_ds   = EEGDataset(X_val,   y_val)
test_ds  = EEGDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
test_loader  = DataLoader(test_ds,  batch_size=64)

#  CHECKPOINTING & LR SCHEDULING 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KAleepNet(n_classes=5, lstm_hidden=128, time_steps=10, flash_block=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ---- Learning rate scheduler (ReduceLROnPlateau) ----
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
best_f1 = 0.0

for epoch in range(1, 51):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_loss = np.mean(losses)

    # Validation F1
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, 1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(pred)
    val_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Epoch {epoch:02d}: Train loss={train_loss:.4f}  Val F1={val_f1:.4f}")

  
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print(" (Best model saved)")
    
    scheduler.step(val_f1)

#  LOAD BEST MODEL FOR TESTING 
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, 1).cpu().numpy()
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(pred)
print("Test F1 score:", f1_score(y_true, y_pred, average="macro"))
print("Classification report:\n", classification_report(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
