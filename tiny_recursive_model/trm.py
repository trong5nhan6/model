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
        # num_tokens,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
        # num_register_tokens = 0
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.output_init_embed = nn.Parameter(torch.randn(1, 1, dim))
        self.latent_init_embed = nn.Parameter(torch.randn(1, 1, dim))
        self.num_classes = num_classes
        # self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        # self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # register tokens for the self attend version

        # self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # prediction heads
        self.to_pred = nn.Linear(dim, num_classes)
        self.to_halt_pred = nn.Linear(dim, 1)

        # self.to_pred = nn.Linear(dim, num_tokens, bias = False)
        # self.to_halt_pred = nn.Sequential(
        #     Reduce('b n d -> b d', 'mean'),
        #     nn.Linear(dim, 1, bias = False),
        #     Rearrange('... 1 -> ...')
        # )

        self.halt_loss_weight = halt_loss_weight

        # init

        nn.init.zeros_(self.to_halt_pred.weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    # def embed_inputs_with_registers(
    #     self,
    #     seq
    # ):
    #     batch = seq.shape[0]
    #     inputs = self.input_embed(seq)
    #     # maybe registers
    #     registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)
    #     inputs, packed_shape = pack([registers, inputs], 'b * d')
    #     return inputs, packed_shape

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
        seq,
        halt_prob_thres = 0.5, # threshold for halting
        max_deep_refinement_steps = 12 # maximum number of refinement steps
    ):
        batch = seq.shape[0] # number of sequences in the batch

        inputs, packed_shape = self.embed_inputs_with_registers(seq) #(b n+n' d)

        # initial outputs and latents

        outputs, latents = self.get_initial() # (d)

        # active batch indices, the step it exited at, and the final output predictions

        active_batch_indices = arange(batch, device = self.device, dtype = torch.float32) # [0., 1., 2., 3.]

        preds = []
        exited_step_indices = [] # lưu step nào dung ở step này
        exited_batch_indices = [] # lưu batch nao dung ở step này

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps

            outputs, latents = self.deep_refinement(inputs, outputs, latents)

            halt_prob = self.to_halt_pred(outputs).sigmoid() # (b)

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            # maybe remove registers

            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

            # append to exited predictions

            pred = self.to_pred(outputs_for_pred[should_halt])
            preds.append(pred)

            # append the step at which early halted

            exited_step_indices.extend([step] * should_halt.sum().item()) # biết step này có bao nhieu batch halt

            # append indices for sorting back

            exited_batch_indices.append(active_batch_indices[should_halt])  # biết batch nao dung ở step này

            if is_last:
                continue

            # ready for next round

            inputs = inputs[~should_halt] # (b n+n' d) should_halt-(b) True dung False conti
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if is_empty(outputs):
                break

        preds = cat(preds).argmax(dim = -1)
        exited_step_indices = tensor(exited_step_indices)

        exited_batch_indices = cat(exited_batch_indices)
        sort_indices = exited_batch_indices.argsort(dim = -1) #(tensor([1, 0, 2, 3])) sắp xếp theo batch index 0 1 2 3

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        images,
        # outputs,
        # latents,
        labels = None
    ):
        B = images.size(0)

        x = self.patch_embed(images)           # (B, N, D)
        cls = self.cls_token.repeat(B, 1, 1)
        inputs = torch.cat([cls, x], dim=1)    # (B, N+1, D)
        outputs, latents = self.get_initial() # (B, 1, D)
        # inputs, packed_shape = self.embed_inputs_with_registers(seq)

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
        return total_loss, cls_loss, halt_loss

        # registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

        # pred = self.to_pred(outputs_for_pred)

        # halt_logits = self.to_halt_pred(outputs)

        # halt_prob = halt_logits.sigmoid()

        # outputs, latents = outputs.detach(), latents.detach()

        # return_package = (outputs, latents, pred, halt_prob)

        # if not exists(labels):
        #     return return_package

        # # calculate loss if labels passed in

        # loss = F.cross_entropy(rearrange(pred, 'b n l -> b l n'), labels, reduction = 'none')
        # loss = reduce(loss, 'b ... -> b', 'mean')

        # is_all_correct = (pred.argmax(dim = -1) == labels).all(dim = -1)

        # halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_all_correct.float(), reduction = 'none')

        # # total loss and loss breakdown

        # total_loss = (
        #     loss +
        #     halt_loss * self.halt_loss_weight
        # )

        # losses = (loss, halt_loss)

        # return (total_loss.sum(), losses, *return_package)
