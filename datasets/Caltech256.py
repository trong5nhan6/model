from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import csv


class Caltech256(Dataset):
    def __init__(self, csv_file, data_root, transform=None):
        self.samples = []
        self.data_root = data_root

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        rel_path = item["image_path"].replace("\\", "/").lstrip("/")
        img_path = os.path.join(self.data_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        label = int(item["label_id"])

        if self.transform:
            img = self.transform(img)

        return img, label


def build_dataloader(
    csv_path,
    data_root,        
    batch_size=32,
    img_size=224,
    is_train=True,
    num_workers=4
):

    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        shuffle = False

    dataset = Caltech256(
        csv_file=csv_path,
        data_root=data_root,  
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    return loader
