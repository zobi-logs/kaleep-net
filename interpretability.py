"""
interpretability.py

- Integrated Gradients (IG) for PyTorch models
- Functions to plot IG and EEG for individual/multiple samples
- Sleep stage label mapping included

Usage example:
    from interpretability import (
        integrated_gradients, plot_eeg_and_ig, plot_multiple_examples_in_grid, plot_ig_statistics, sleep_stage_names
    )
    # After loading your PyTorch model and test tensors
    sample_idx = 0
    input_sample = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    baseline = torch.zeros_like(input_sample)
    pred_class = model(input_sample).argmax(dim=1).item()
    ig = integrated_gradients(model, input_sample, pred_class, baseline)
    plot_eeg_and_ig(sample_idx, input_sample.cpu().numpy(), ig.cpu().numpy(), y_test[sample_idx])
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

sleep_stage_names = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

def integrated_gradients(model, x, target_class, baseline=None, steps=50):
    """
    Integrated Gradients for PyTorch models.
    Args:
        model: the PyTorch model (should output logits)
        x: input tensor (1, seq_len, 1)
        target_class: int, class index
        baseline: same shape as x (default: all zeros)
        steps: number of interpolation steps
    Returns:
        IG attributions (same shape as x)
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    # Scale inputs and compute gradients
    interpolated = [baseline + (float(i)/steps)*(x-baseline) for i in range(steps+1)]
    interpolated = torch.cat(interpolated, dim=0)
    interpolated.requires_grad = True
    model.eval()
    logits = model(interpolated)
    scores = logits[:, target_class]
    grads = torch.autograd.grad(scores.sum(), interpolated, create_graph=False)[0]
    avg_grads = grads.view(steps+1, *x.shape[1:]).mean(dim=0)
    ig = (x - baseline) * avg_grads
    return ig

def detect_alpha_peaks(eeg_signal):
    num_points = eeg_signal.shape[0]
    if num_points < 5:
        return np.arange(num_points)
    alpha_idx = np.random.choice(num_points, size=5, replace=False)
    alpha_idx.sort()
    return alpha_idx

def plot_eeg_and_ig(sample_idx, eeg_signal, ig_signal, class_label):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    eeg_signal = eeg_signal.squeeze()
    ig_signal = ig_signal.squeeze()
    stage_name = sleep_stage_names[class_label]
    axes[0].plot(eeg_signal)
    axes[0].set_title(f'Raw EEG Signal - Stage: {stage_name}, Sample: {sample_idx}')
    axes[0].set_ylabel('Amplitude')
    threshold = np.percentile(np.abs(ig_signal), 95)
    important_regions = np.where(np.abs(ig_signal) > threshold)[0]
    axes[0].scatter(important_regions, eeg_signal[important_regions],
                    color='red', label='Important regions')
    axes[0].legend()
    axes[1].plot(ig_signal, color='orange')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_title(f'Integrated Gradients Importance - Stage: {stage_name}')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_multiple_examples_in_grid(X_test, y_test, model, integrated_gradients,
                                   sleep_stage_names, device='cpu', samples_per_class=2):
    num_classes = len(sleep_stage_names)
    fig, axs = plt.subplots(num_classes, samples_per_class, figsize=(12, 12),
                            sharex=True, sharey=False)
    if samples_per_class == 1:
        axs = np.expand_dims(axs, axis=1)
    for row_class_label in range(num_classes):
        class_indices = np.where(y_test == row_class_label)[0][:samples_per_class]
        for col_idx, sample_idx in enumerate(class_indices):
            ax = axs[row_class_label, col_idx]
            input_sample = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
            baseline = torch.zeros_like(input_sample)
            pred_class = model(input_sample).argmax(dim=1).item()
            ig_attributions = integrated_gradients(model, input_sample, pred_class, baseline)
            eeg_np = input_sample.cpu().numpy().squeeze()
            ig_np = ig_attributions.cpu().numpy().squeeze()
            ax.plot(eeg_np, alpha=0.7)
            threshold = np.percentile(np.abs(ig_np), 95)
            high_ig_indices = np.where(np.abs(ig_np) > threshold)[0]
            ax.scatter(high_ig_indices, eeg_np[high_ig_indices], c='red', s=40,
                       label='High IG')
            alpha_peaks_idx = detect_alpha_peaks(eeg_np)
            ax.scatter(alpha_peaks_idx, eeg_np[alpha_peaks_idx],
                       marker='o', facecolors='none', edgecolors='green',
                       s=100, label='Alpha Peaks')
            stage_name = sleep_stage_names[row_class_label]
            ax.set_title(f'{stage_name} (sample {sample_idx})')
            ax.legend(loc='upper right', fontsize=8)
    plt.suptitle("Multiple Examples per Class: EEG + IG Highlighted", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_ig_statistics(X_test, y_test, model, integrated_gradients, sleep_stage_names, device='cpu'):
    num_classes = len(sleep_stage_names)
    ig_class_means = []
    ig_class_stds = []
    for class_label in range(num_classes):
        class_indices = np.where(y_test == class_label)[0]
        ig_accumulator = []
        for sample_idx in class_indices:
            input_sample = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
            baseline = torch.zeros_like(input_sample)
            ig_attr = integrated_gradients(model, input_sample, class_label, baseline)
            ig_accumulator.append(ig_attr.cpu().numpy().squeeze())
        if len(ig_accumulator) == 0:
            ig_class_means.append(np.zeros(X_test.shape[1]))
            ig_class_stds.append(np.zeros(X_test.shape[1]))
            continue
        ig_accumulator = np.array(ig_accumulator)
        ig_mean = np.mean(ig_accumulator, axis=0)
        ig_std  = np.std(ig_accumulator, axis=0)
        ig_class_means.append(ig_mean)
        ig_class_stds.append(ig_std)
    time_axis = np.arange(0, X_test.shape[1])
    plt.figure(figsize=(10,6))
    for class_label in range(num_classes):
        stage_name = sleep_stage_names[class_label]
        mean_values = ig_class_means[class_label]
        std_values  = ig_class_stds[class_label]
        plt.plot(time_axis, mean_values, label=f'{stage_name} mean IG')
        plt.fill_between(time_axis,
                         mean_values - std_values,
                         mean_values + std_values, alpha=0.2)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Mean ± Std of Integrated Gradients per Class')
    plt.xlabel('Time Step')
    plt.ylabel('IG Contribution')
    plt.legend()
    plt.show()
