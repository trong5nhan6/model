from __future__ import annotations

import torch
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA
import logging
import os
# helpers


def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def exists(v):
    return v is not None


def range_from_one(n):
    return range(1, n + 1)

# metric helpers


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    Args:
        logits: (B, C)
        labels: (B,)
        num_classes: số class

    Returns:
        dict: acc, precision, recall, f1 (tất cả macro-average)
    """
    preds = logits.argmax(dim=-1)

    # confusion matrix components
    tp = torch.zeros(num_classes, device=logits.device)
    fp = torch.zeros(num_classes, device=logits.device)
    fn = torch.zeros(num_classes, device=logits.device)

    for c in range(num_classes):
        tp[c] = ((preds == c) & (labels == c)).sum()
        fp[c] = ((preds == c) & (labels != c)).sum()
        fn[c] = ((preds != c) & (labels == c)).sum()

    # accuracy (micro)
    acc = (preds == labels).float().mean()

    # precision, recall, f1 (macro average)
    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / \
        (precision_per_class + recall_per_class + 1e-8)

    metrics = {
        "acc": acc.item(),  # micro accuracy
        "precision": precision_per_class.mean().item(),
        "recall": recall_per_class.mean().item(),
        "f1": f1_per_class.mean().item(),
    }

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, *_ = model(images)
        all_logits.append(outputs)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = classification_metrics(all_logits, all_labels, model.num_classes)
    return metrics
# trainer


class Trainer(Module):
    def __init__(
        self,
        *,
        model,
        train_dataset,
        val_dataset=None,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=1000,
        ema_decay=0.999,
        max_recurrent_steps=6,
        halt_prob_thres=0.7,
        log_file="logs/train.log",
        accelerate_kwargs: dict = dict(),
        cpu=False
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs, cpu=cpu)

        self.model = model
        self.epochs = epochs
        self.max_recurrent_steps = max_recurrent_steps
        self.halt_prob_thres = halt_prob_thres
        # dataloader
        self.train_loader = train_dataset

        if exists(val_dataset):
            self.val_loader = val_dataset

        # optimizer
        self.optim = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # scheduler (linear warmup)
        self.scheduler = LambdaLR(
            self.optim,
            lambda step: min((step + 1) / warmup_steps, 1.0)
        )

        # prepare distributed
        prepare_objs = [self.model, self.optim,
                        self.train_loader, self.scheduler]
        if exists(self.val_loader):
            prepare_objs.append(self.val_loader)
        self.model, self.optim, self.train_loader, self.scheduler = self.accelerator.prepare(
            *prepare_objs[:4])
        if exists(self.val_loader):
            self.val_loader = self.accelerator.prepare(self.val_loader)

        # EMA (only on main process)
        self.ema_model = None
        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta=ema_decay,
                forward_method_names=('forward',)
            )

        self.logger = None
        if self.accelerator.is_main_process:
            self.logger = setup_logger(log_file)

    def evaluate(self):
        self.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.model.device, non_blocking=True)
                labels = labels.to(self.model.device, non_blocking=True)

                logits, *_ = self.model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = classification_metrics(
            all_logits, all_labels, self.model.num_classes)

        del all_logits, all_labels
        torch.cuda.empty_cache()
        return metrics

    def forward(self):
        for epoch in range_from_one(self.epochs):
            self.model.train()
            correct = 0
            total = 0

            for step, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(
                    self.model.device), labels.to(self.model.device)
                # total_loss, cls_loss, halt_loss, logits, halt_logits = self.model(
                #     images, labels=labels)

                # backward
                # self.accelerator.backward(total_loss)

                # -------- INIT --------
                active_images = images
                active_labels = labels

                total_loss_step = 0.0

                for t in range(self.max_recurrent_steps):
                    if active_images.numel() == 0:
                        break

                    total_loss, cls_loss, halt_loss, logits, halt_probs = \
                        self.model(active_images, labels=active_labels)

                    self.accelerator.backward(total_loss)

                    total_loss_step += total_loss.detach()

                    # LOG MỖI STEP
                    self.accelerator.print(
                        f"[Epoch {epoch}] "
                        f"[Step {t+1}/{self.max_recurrent_steps}] "
                        f"loss={total_loss.item():.4f} | "
                        f"cls={cls_loss.item():.4f} | "
                        f"halt={halt_loss.item():.4f} | "
                        f"active={active_images.shape[0]}"
            )

                    # -------- HALTING --------
                    halt_mask = halt_probs >= self.halt_prob_thres

                    # remove halted samples
                    active_images = active_images[~halt_mask]
                    active_labels = active_labels[~halt_mask]


                self.optim.step()
                self.optim.zero_grad()
                self.scheduler.step()

                if exists(self.ema_model) and self.accelerator.is_main_process:
                    self.ema_model.update()

                # # collect logits and labels for train metrics
                # logits, _ = self.model(images)
                # train_metrics_state["all_logits"].append(logits.detach())
                # train_metrics_state["all_labels"].append(labels.detach())

                # ---- TRAIN METRICS ----
                # preds = logits.argmax(dim=-1)
                # correct += (preds == labels).sum().item()
                # total += labels.numel()

                # logging train step every 20 steps
                # if step % 20 == 0:
                #     train_acc = correct / max(total, 1)
                #     msg = (
                #         f"[Epoch {epoch}] Step {step} | "
                #         f"loss: {total_loss.item():.4f} | "
                #         f"cls_loss: {cls_loss.item():.4f} | "
                #         f"halt_loss: {halt_loss.item():.4f} | "
                #         f"train_acc: {train_acc:.4f}"
                #     )
                #     self.accelerator.print(msg)
                #     if exists(self.logger):
                #         self.logger.info(msg)

            # Evaluate at end of epoch
            if exists(self.val_loader):
                val_metrics = self.evaluate()
                msg = (
                    f"[Epoch {epoch}] Validation | "
                    f"acc: {val_metrics['acc']:.4f} | "
                    f"precision: {val_metrics['precision']:.4f} | "
                    f"recall: {val_metrics['recall']:.4f} | "
                    f"f1: {val_metrics['f1']:.4f}"
                )
                self.accelerator.print(msg)
                if exists(self.logger):
                    self.logger.info(msg)

        self.accelerator.print("Training complete")
        if exists(self.logger):
            self.logger.info("Training complete")
        if exists(self.ema_model) and self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()
