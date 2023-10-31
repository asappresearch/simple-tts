import torch.nn as nn
from torch.optim import AdamW

# Implementation from Timm: https://github.com/huggingface/pytorch-image-models/blob/2d0dbd17e388953ab81a5c56f80074eff962ea6b/timm/optim/optim_factory.py#L40
# Exclude bias and normalization (e.g. LayerNorm) params
def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=.01,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_adamw_optimizer(model, lr, betas, weight_decay):
    param_groups = param_groups_weight_decay(model, weight_decay=weight_decay)

    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)
