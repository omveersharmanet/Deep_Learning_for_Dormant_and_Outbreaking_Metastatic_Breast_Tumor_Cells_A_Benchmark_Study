import os
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================================
# Dataset: each temporal frame is an independent sample
# ============================================================
class PatchSequenceDataset(Dataset):
    """
    Each sample is a full temporal sequence.
    """

    def __init__(self, patches, labels):
        """
        patches : list of tensors [(1, L, H, W), ...]
        labels  : tensor (N,)
        """
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]      # (1, L, H, W)
        y = self.labels[idx]
        return x, y

# Data loader

def load_2d_data_seq(
    root,
    batch_size,
    sequence_length=1,
    framesize=96
):
    """
    Loads 2D patch data and applies sliding window AFTER patch-level split.
    """
    dataset_file = f"Dataset_{framesize}x{framesize}_SL{sequence_length}.pt"
    save_path = os.path.join(root, dataset_file)

    if os.path.exists(save_path):

        print(f"Loading existing dataset: {save_path}")

        data = torch.load(save_path)

        train_patches = data['train_patches']
        train_labels  = data['train_labels']

        val_patches = data['val_patches']
        val_labels  = data['val_labels']

        test_patches = data['test_patches']
        test_labels  = data['test_labels']

    else:

        print("Dataset not found. Creating dataset...")
        
    

    print(f"Train sequences: {len(train_patches)}")
    print(f"Val sequences:   {len(val_patches)}")
    print(f"Test sequences:  {len(test_patches)}")

    # =====================================================
    # 4. DATASETS & LOADERS
    # =====================================================
    train_dataset = PatchSequenceDataset(train_patches, train_labels)
    val_dataset   = PatchSequenceDataset(val_patches, val_labels)
    test_dataset  = PatchSequenceDataset(test_patches, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



