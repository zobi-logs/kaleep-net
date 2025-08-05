# KAleep-Net

PyTorch implementation of KAleep-Net: Kolmogorov-Arnold Flash Attention Network for sleep stage classification.

## Requirements

- See `requirements.txt` for dependencies.

## How to Use

1. Place your `.npz` EEG data files in the `data/` folder.
   - Each file should have:
     - `x` of shape `[n_epochs, 3000, 1]`
     - `y` of shape `[n_epochs]` (labels 0-4)

2. Install requirements:

3. Run the main script:
- The best model will be saved as `best_model.pth`.
- Test results will be printed.

## Main Files

- `main.py`             # Model training, validation, and testing
- `kan_layer.py`        # KAN layer definition
- `flash_attention.py`  # Flash Attention module
- `preprocess.py`       # Preprocessing functions
- `interpretability.py` # Integrated gradients (explainability)
- `utils.py`            # Miscellaneous utilities
- `requirements.txt`    # Dependencies

## Note

- Data is not included due to licensing/privacy. Use your own or follow dataset links in the paper.
