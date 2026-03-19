"""
    extract_ckpt_info.py

    Utility script to inspect and print metadata from a PyTorch model checkpoint file.

    Features:
        - Displays backbone/model name, dropout, learning rate, number of classes, class names, validation accuracy, epoch, and any additional info in the checkpoint.
        - Works with both training and inference checkpoints saved as Python dictionaries.

    Example usage:
        python extract_ckpt_info.py
"""

import torch


def print_checkpoint_info(ckpt_path):
    """
    Loads a PyTorch checkpoint file and prints key metadata for inspection.

    Outputs information such as backbone name, hyperparameters, classes, and any extra fields found in the checkpoint dictionary.

    Args:
        ckpt_path (str): Path to the saved checkpoint (.pt or .pth file).

    Prints:
        - Backbone/model information ('arch' or 'backbone')
        - Dropout (if available)
        - Learning rate (if available)
        - Number of classes
        - Class names
        - Best validation accuracy (if present)
        - Epoch number (if present)
        - Any additional keys and their values in the checkpoint dict

    Example:
        == Checkpoint: KDEF/checkpoints/merged_best_20260316_114427.pt ==
        Backbone/Model Name : convnext_tiny
        Dropout: 0.2
        Learning Rate: 0.0003
        Number of Classes: 7
        Classes: ['angry', 'disgust', ...]
        Best Validation Accuracy: 0.9481
        Epoch: 27
        Other keys in checkpoint:
          optimizer_state: {...}
    """
    # Load the checkpoint file
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print(f"== Checkpoint: {ckpt_path} ==")
    # Print known keys if present
    for key in ["arch", "backbone"]:
        if key in ckpt:
            print(f"Backbone/Model Name : {ckpt[key]}")
    if "dropout" in ckpt:
        print(f"Dropout: {ckpt['dropout']}")
    if "lr" in ckpt:
        print(f"Learning Rate: {ckpt['lr']}")
    if "num_classes" in ckpt:
        print(f"Number of Classes: {ckpt['num_classes']}")
    if "classes" in ckpt:
        print(f"Classes: {ckpt['classes']}")
    if "best_val_acc" in ckpt:
        print(f"Best Validation Accuracy: {ckpt['best_val_acc']:.4f}")
    if "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")

    # Print any extra information found in the checkpoint
    extra_keys = set(ckpt.keys()) - {"arch", "backbone", "dropout", "lr", "num_classes", "classes", "best_val_acc",
                                     "state_dict", "epoch"}
    if extra_keys:
        print("Other keys in checkpoint:")
        for k in extra_keys:
            print(f"  {k}: {ckpt[k]}")


if __name__ == "__main__":
    ckpt_file = "KDEF/checkpoints/merged_best_20260317_083849.pt"
    print_checkpoint_info(ckpt_file)