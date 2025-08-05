import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path):
    """Save PyTorch model state dict."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    """Load PyTorch model state dict."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    """Plot and display confusion matrix using sklearn/Matplotlib."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()
