# KAleep-Net

KAleep-Net: A Kolmogorov-Arnold Flash Attention Network for Sleep Stage Classification Using Single-Channel EEG with Explainability

## Requirements

- See `requirements.txt` for dependencies.

## Datasets

This code has been tested with the following public sleep EEG datasets:

- **EDF20**: [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
- **EDF78**: [Sleep-EDF Database](https://physionet.org/content/sleep-edf/1.0.0/)
- **SHHS**: [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs)

For each dataset:
- Please ensure you extract and use a single EEG channel (as in the paper).
- Data should be segmented into 30s epochs, sampled at 100 Hz (3000 samples per epoch).
- Map sleep stage labels to the format `[0: Wake, 1: N1, 2: N2, 3: N3, 4: REM]` as used in this code.
- **Raw data is not provided**. Please download from the official sources above and follow their data use agreements.

If using your own data, please ensure it matches the input format described above.

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
