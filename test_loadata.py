import torch
import matplotlib.pyplot as plt

from datasets.Caltech256 import build_dataloader


def test_dataloader(
    csv_path,
    data_root,
    batch_size=8,
    img_size=224
):
    print(f"\n Testing dataloader:")
    print(f" CSV : {csv_path}")
    print(f" ROOT: {data_root}")

    loader = build_dataloader(
        csv_path=csv_path,
        data_root=data_root,
        batch_size=batch_size,
        img_size=img_size,
        is_train=True,
        num_workers=2
    )

    images, labels = next(iter(loader))

    print(" Batch loaded successfully!")
    print(f" Images shape: {images.shape}")
    print(f" Labels shape: {labels.shape}")
    print(f" Labels: {labels.tolist()}")

    # Check lỗi
    assert images.shape[1] == 3, "Image channel != 3"
    assert not torch.isnan(images).any(), "Image contains NaN"

    # De-normalize để visualize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    images_vis = images[:4].cpu() * std + mean
    images_vis = images_vis.clamp(0, 1)

    plt.figure(figsize=(10, 4))
    for i in range(len(images_vis)):
        plt.subplot(1, len(images_vis), i + 1)
        plt.imshow(images_vis[i].permute(1, 2, 0))
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()

    print(" DataLoader OK!")


def main():
    data_root = "datasets/archive/256_ObjectCategories"
    TRAIN_CSV = "datasets/splits/train.csv"

    test_dataloader(
        csv_path=TRAIN_CSV,
        data_root=data_root,
        batch_size=8,
        img_size=224
    )


if __name__ == "__main__":
    main()
