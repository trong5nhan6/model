import torch
from tiny_recursive_model import TinyRecursiveModel, TransformerEncoder, MLPMixer1D

def print_model_params(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total += num
            print(f"{name:60s} {list(param.shape)} | {num:,}")
    print("-" * 90)
    print(f"Total trainable params: {total:,}")

if __name__ == "__main__":
    B = 2
    IMG_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANS = 3
    DIM = 128
    NUM_CLASSES = 10

    trans = TransformerEncoder(
    dim=DIM,
    depth=2,
    num_heads=4,
    mlp_ratio=4.0,
    dropout=0.1
    )

    mlp = MLPMixer1D(
    dim=DIM,
    depth=2,
    seq_len=197
    )

    model = TinyRecursiveModel(
    dim=DIM,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=IN_CHANS,
    num_classes=NUM_CLASSES,
    # num_tokens=NUM_CLASSES,
    network=trans,
    num_refinement_blocks=3,
    num_latent_refinements=6
    )
    model.eval()
    
    images = torch.randn(B, IN_CHANS, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (B,))

    with torch.no_grad():
        logits, halt_prob = model(images)

    print("logits:", logits.shape)        # (B, num_classes)
    print("logits:", logits)
    print("halt_prob:", halt_prob.shape)  # (B,)
    print("halt_prob:", halt_prob)

    # =========================
    # forward test (with loss)
    # =========================
    loss, cls_loss, halt_loss = model(images, labels)

    print("total loss:", loss.item())
    print("cls loss:", cls_loss.item())
    print("halt loss:", halt_loss.item())

    print(model)
    print_model_params(model)