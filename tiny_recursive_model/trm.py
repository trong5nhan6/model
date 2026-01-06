from __future__ import annotations
from contextlib import nullcontext

import torch
from torch import nn, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Reduce, Rearrange

# network related

from tiny_recursive_model.model import MLPMixer1D, TransformerEncoder, PatchEmbed

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

# classes

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim=768,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.num_classes = num_classes

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # prediction heads
        self.to_pred = nn.Linear(dim, num_classes)
        self.to_halt_pred = nn.Linear(dim, 1)
        self.halt_loss_weight = halt_loss_weight

        # init
        nn.init.uniform_(self.to_halt_pred.weight, a=0.0, b=1.0)
        # nn.init.zeros_(self.to_halt_pred.weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    def refine_latent_then_output_once(
        self,
        inputs,     # (b n d)
        outputs,    # (b n d)
        latents,    # (b n d)
    ):

        # so it seems for this work, they use only one network
        # the network learns to refine the latents if input is passed in, otherwise it refines the output

        for _ in range(self.num_latent_refinements):

            latents = self.network(outputs + latents + inputs)

        outputs = self.network(outputs + latents)

        return outputs, latents

    def deep_refinement(
        self,
        inputs,    # (b n d)
        outputs,   # (b n d)
        latents,   # (b n d)
    ):

        for step in range_from_one(self.num_refinement_blocks):

            # only last round of refinement receives gradients

            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

        return outputs, latents

    @torch.no_grad()
    def predict(
        self,
        images,
        halt_prob_thres=0.5,
        max_deep_refinement_steps=12
    ):
        """
        images: (B, C, H, W)

        return:
            preds: (B,)
            exited_step_indices: (B,)
        """
        self.eval()

        device = images.device
        batch = images.size(0)

        # ------------------------------------------------
        # Embed inputs (same logic as forward)
        # ------------------------------------------------
        x = self.patch_embed(images)                  # (B, N, D)
        cls = self.cls_token.expand(batch, -1, -1)    # (B, 1, D)
        inputs = torch.cat([cls, x], dim=1)           # (B, N+1, D)

        # ------------------------------------------------
        # Initial outputs and latents
        # ------------------------------------------------
        outputs, latents = self.get_initial()

        # ------------------------------------------------
        # ACT bookkeeping
        # ------------------------------------------------
        active_batch_indices = torch.arange(batch, device=device, dtype=torch.long)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        # ------------------------------------------------
        # ACT loop
        # ------------------------------------------------
        for step in range(1, max_deep_refinement_steps + 1):
            print("step:", step)
            is_last = step == max_deep_refinement_steps

            outputs, latents = self.deep_refinement(inputs, outputs, latents) # (b, n+1, d), (b, n+1, d)
            cls_out = outputs[:, 0]                          # (b, D)
            logits = self.to_pred(cls_out)                     # (b, num_classes)
            halt_prob = self.to_halt_pred(cls_out).sigmoid().squeeze(-1) # (b,)

            should_halt = (halt_prob >= halt_prob_thres) | is_last # (b,)

            if not should_halt.any():
                continue

            # ------------------------------------------------
            # Collect predictions
            # ------------------------------------------------
            preds.append(logits[should_halt])
            exited_step_indices.extend([step] * should_halt.sum().item())
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # ------------------------------------------------
            # Remove halted samples
            # ------------------------------------------------

  
            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if inputs.numel() == 0:
                break

        # ------------------------------------------------
        # Restore original batch order
        # ------------------------------------------------
        preds = torch.cat(preds, dim=0).argmax(dim=-1)
        exited_step_indices = torch.tensor(
            exited_step_indices, device=device, dtype=torch.long
        )

        exited_batch_indices = torch.cat(exited_batch_indices, dim=0)
        sort_indices = exited_batch_indices.argsort(dim=-1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        images,
        outputs,
        latents,
        labels = None
    ):
        B = images.size(0)

        x = self.patch_embed(images)           # (B, N, D)
        cls_token = self.cls_token.repeat(B, 1, 1)
        inputs = torch.cat([cls_token, x], dim=1)    # (B, N+1, D)

        outputs, latents = self.deep_refinement(inputs, outputs, latents)

        cls_out = outputs[:, 0] 
        logits = self.to_pred(cls_out)         # (B, num_classes)
        halt_logits = self.to_halt_pred(cls_out).squeeze(-1)

        if labels is None:
            return logits, halt_logits.sigmoid()

        cls_loss = F.cross_entropy(logits, labels)

        is_correct = logits.argmax(dim=-1) == labels
        halt_loss = F.binary_cross_entropy_with_logits(
            halt_logits, is_correct.float()
        )

        total_loss = cls_loss + halt_loss * self.halt_loss_weight
        return total_loss, cls_loss, halt_loss, logits, halt_logits.sigmoid(), outputs, latents
