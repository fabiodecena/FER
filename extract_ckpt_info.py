import torch
from pathlib import Path

def print_checkpoint_info(ckpt_path):
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
    extra_keys = set(ckpt.keys()) - {"arch", "backbone", "dropout", "lr", "num_classes", "classes", "best_val_acc", "state_dict", "epoch"}
    if extra_keys:
        print("Other keys in checkpoint:")
        for k in extra_keys:
            print(f"  {k}: {ckpt[k]}")

if __name__ == "__main__":
    ckpt_file = "MMA/checkpoints/mma_final_model.pt"
    print_checkpoint_info(ckpt_file)