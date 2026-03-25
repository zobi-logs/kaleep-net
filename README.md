# KAleep-Net

**KAleep-Net: A Kolmogorov-Arnold Flash Attention Network for Sleep Stage Classification Using Single-Channel EEG With Explainability**

> Published in *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, Vol. 33, 2025  
> DOI: [10.1109/TNSRE.2025.3606128](https://doi.org/10.1109/TNSRE.2025.3606128)

---

## Important Notice — Code Version History

This repository contains **v2.0** of the released code, which supersedes an earlier version that was shared informally prior to publication. If you received or cloned code before **[insert your release date here]**, please discard it and use only this version.

**What was wrong in the earlier version:**

| Issue | Earlier version | This version |
|---|---|---|
| BiLSTM depth | `num_layers=1` | `num_layers=2` (matches Table II) |
| Temporal sequence handling | Collapsed to single vector before BiLSTM | Full 750-step sequence preserved |
| MaxPool placement | After Conv-KAN pair 2 | After Conv-KAN pair 3 (matches Eq. 1 & 4) |
| Class-weighted loss | Unweighted `CrossEntropyLoss` | Weighted per Eq. 22 (`w = 1/log(1+f)`) |
| Evaluation protocol | Single train/test split | 10-fold subject-wise `GroupKFold` (matches Table V) |
| `preprocess.py` artifact rejection | `axis=1` — crashes on 3-D arrays | `axis=(1,2)` — correct |
| Subject-overlap assertions | Indexed wrong array | Fixed to use per-fold group arrays |
| Syntax error in `main.py` | Stray `n)` token — crashes immediately | Fixed |

Results reproducible from this version are intended to match those reported in Table V of the paper.

---

## Transparent Implementation Notes

In the interest of scientific reproducibility and open-source integrity, the following deliberate simplifications relative to the paper text are documented here:

**1. KAN activation function**

The paper states the activation σ is *"implicitly defined and adaptively learned rather than fixed"* (Section II.A), referring to the learnable spline basis of the original KAN formulation (Liu et al., 2024). This implementation uses `tanh` as the basis function, which is a fixed nonlinearity. This is a practical engineering simplification: `tanh` is a valid nonlinear basis and is GPU-native without additional dependencies. The published results were obtained with this implementation. If you require true learnable spline bases, consider integrating the [pykan](https://github.com/KindXiaoming/pykan) library.

**2. KAN connection style**

The architecture diagram (Fig. 1) shows Conv and KAN as sequential operations. This implementation applies KAN as a residual connection: `z = z + KAN(z)`. This improves gradient flow during training and was found to be more stable in practice. It does not change the output dimensionality or the overall information flow.

These two points are the only differences between the paper description and this code. Everything else — layer sizes, kernel sizes, dropout rates, optimiser settings, loss formulation, and evaluation protocol — is a direct match to the paper.

---

## Architecture Overview

```
Input EEG: (B, 3000, 1)  — 30 seconds @ 100 Hz, single channel
         │
         ├──── Fine-Scale Feature (FSF) branch ─────────────────────┐
         │     Conv1(k=3, 1→64)  → KAN1(64→64)  → Pool → Drop(0.25)│
         │     Conv2(k=3, 64→128)→ KAN2(128→128)                    │
         │     Conv3(k=3,128→128)→ KAN3(128→128) → Pool             │
         │     Output: (B, 750, 128)                                  │
         │                                                            ├─ Cat → (B, 750, 192)
         └──── Coarse-Scale Feature (CSF) branch ───────────────────┤
               Conv1(k=5, 1→32)  → KAN1(32→32)  → Pool → Drop(0.25)│
               Conv2(k=5, 32→64) → KAN2(64→64)                      │
               Conv3(k=5, 64→64) → KAN3(64→64)  → Pool              │
               Output: (B, 750, 64)                                   │
                                                                      │
(B, 750, 192) → BiLSTM(hidden=128, layers=2, bidirectional)
             → Drop(0.3)
             → FlashAttention(embed=256, block_size=5)
             → MeanPool
             → Drop(0.5)
             → FC(256 → 5)
             → Predicted sleep stage (Wake / N1 / N2 / N3 / REM)
```

---

## Results

Performance on benchmark datasets (10-fold subject-wise cross-validation, mean ± std):

| Dataset | Accuracy | Macro F1 | Cohen's κ | Train time / fold |
|---|---|---|---|---|
| Sleep-EDF-20 | 86.5 ± 0.21% | 76.8% | 0.79 | 10.5 min |
| Sleep-EDF-78 | 85.0% | 77.0% | 0.78 | 17.9 min |
| SHHS | 86.4% | 79.0% | 0.81 | 110 min |

Full per-fold results are in Table V of the paper.

---

## Requirements

```
torch>=1.7.1
numpy
scikit-learn
scipy
matplotlib
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

The code was developed and tested with Python 3.8, PyTorch 1.7.1, CUDA 11.0, and cuDNN 8.0.5 on an NVIDIA RTX A6000 (48 GB). It runs on CPU as well, though training will be substantially slower.

---

## Datasets

This code has been tested with the following publicly available datasets:

| Dataset | Source | Subjects | Sampling rate |
|---|---|---|---|
| Sleep-EDF-20 | [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) | 20 | 100 Hz |
| Sleep-EDF-78 | [PhysioNet](https://physionet.org/content/sleep-edf/1.0.0/) | 78 | 100 Hz |
| SHHS | [NSRR](https://sleepdata.org/datasets/shhs) | 329 (AHI < 5) | 125 Hz → resampled 100 Hz |

**Raw data is not included in this repository.** You must download datasets directly from the official sources and comply with their respective data use agreements.

Sleep stage labels should be mapped as follows before saving to `.npz`:

```
0 = Wake (W)
1 = N1
2 = N2
3 = N3  (R&K stages 3 and 4 merged)
4 = REM
```

---

## Data Preparation

Before running the model, preprocess your raw EEG data using `preprocess.py`. The pipeline applies:

1. Segmentation into non-overlapping 30-second epochs (3000 samples at 100 Hz)
2. 4th-order Butterworth bandpass filter: 0.5–40 Hz
3. Artifact rejection: epochs with any sample exceeding ±100 µV are discarded
4. Per-epoch z-score normalisation

```python
from preprocess import preprocess_pipeline
import numpy as np

# Xraw: 1-D NumPy array of continuous EEG samples at 100 Hz
X_clean = preprocess_pipeline(Xraw, fs=100, epoch_sec=30, thres=100)
# X_clean shape: (n_epochs, 3000, 1)

np.savez('subject_01.npz', x=X_clean, y=your_labels)
```

Save one `.npz` file per subject. Place all files in the `data/` directory.

---

## Running the Model

```bash
python main.py
```

The script will:

1. Load all `.npz` files from `data/`
2. Run 10-fold subject-wise cross-validation
3. For each fold: train with early stopping (patience = 10), save best checkpoint as `kaleep_fold{N}.pth`
4. Print per-fold accuracy, macro F1, and Cohen's κ
5. Print a final summary table with mean ± std across all folds

To change the data directory, edit this line in `main.py`:

```python
npz_folder = './data'   # ← set your path here
```

---

## Explainability

After training, use `interpretability.py` to generate Integrated Gradients (IG) attributions for any test sample:

```python
from interpretability import integrated_gradients, plot_eeg_and_ig

# Load a trained model checkpoint first
sample_idx = 0
input_sample = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
baseline     = torch.zeros_like(input_sample)
pred_class   = model(input_sample).argmax(dim=1).item()

ig = integrated_gradients(model, input_sample, pred_class, baseline, steps=50)
plot_eeg_and_ig(sample_idx, input_sample.cpu().numpy(), ig.cpu().numpy(), y_test[sample_idx])
```

This produces a two-panel plot showing the raw EEG signal with high-attribution time points highlighted in red, and the IG contribution curve below it. See Section IV.E of the paper for interpretation guidance.

---

## Repository Structure

```
KAleep-Net/
├── main.py                # Full pipeline: model definition, training, 10-fold CV, evaluation
├── kan_layer.py           # KAN layer (Eqs. 3 & 5)
├── flash_attention.py     # Block-wise Flash Attention (Eq. 11, block_size=5)
├── preprocess.py          # EEG preprocessing pipeline (paper Section III-B)
├── interpretability.py    # Integrated Gradients explainability (paper Section IV-E)
├── utils.py               # Seed setting, model save/load, confusion matrix plotting
├── requirements.txt       # Python dependencies
├── data/                  # Place your .npz subject files here (not included)
└── images/
    └── graphic_abstract.png
```

---

## Citation

If you use this code or the KAleep-Net architecture in your work, please cite:

```bibtex
@article{akbar2025kaleep,
  title     = {{KAleep-Net}: A {Kolmogorov-Arnold} Flash Attention Network for Sleep Stage
               Classification Using Single-Channel {EEG} With Explainability},
  author    = {Akbar, Zubair and Hassan, Farhad and Li, Jingzhen and
               Kashif, Ubaidullah Alias and Liu, Yuhang and Gu, Jia and
               Zhou, Kaixin and Nie, Zedong},
  journal   = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume    = {33},
  pages     = {3685--3696},
  year      = {2025},
  doi       = {10.1109/TNSRE.2025.3606128}
}
```

---

## License

This code is released under the **MIT License**. See `LICENSE` for details.

The datasets (Sleep-EDF-20, Sleep-EDF-78, SHHS) are subject to their own separate data use agreements and are not covered by this license. You are responsible for complying with the terms of whichever datasets you use.

---

## Contact

For questions about the paper, please contact the corresponding author:  
**Zedong Nie** — zd.nie@siat.ac.cn  
Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

For questions specifically about this code implementation, please open a GitHub Issue so that answers are visible to all users and the discussion is preserved.

> **Before opening an issue**, please check:
> - You are using the code from this repository, not an earlier informal version
> - Your `.npz` files match the expected format: `x` shape `(n, 3000, 1)`, `y` shape `(n,)`, labels 0–4
> - You have read the *Transparent Implementation Notes* section above
