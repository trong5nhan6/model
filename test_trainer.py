import torch
from torch.utils.data import Dataset, DataLoader, random_split

from tiny_recursive_model.trm import TinyRecursiveModel
from tiny_recursive_model.model import TransformerEncoder
from tiny_recursive_model.trainer import Trainer   # file Trainer bạn vừa viết

# -------------------------
# Mock dataset (ảnh + label)
# -------------------------
class DummyImageDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)          # fake image
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    torch.manual_seed(42)

    IMG_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANS = 3
    DIM = 128
    NUM_CLASSES = 10

    # -------------------------
    # Network
    # -------------------------
    trans = TransformerEncoder(
        dim=DIM,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # -------------------------
    # Model
    # -------------------------
    model = TinyRecursiveModel(
        dim=DIM,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        network=trans,
        num_refinement_blocks=3,
        num_latent_refinements=6
    )

    # -------------------------
    # Load dataset
    # -------------------------
    full_dataset = DummyImageDataset(
        num_samples=400,
        num_classes=NUM_CLASSES
    )

    # chia train/val 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # tạo DataLoaders trước khi đưa vào Trainer
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_loader,
        val_dataset=val_loader,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        log_file="logs/train.log",
        cpu=True
    )

    # -------------------------
    # TRAIN
    # -------------------------
    trainer()
