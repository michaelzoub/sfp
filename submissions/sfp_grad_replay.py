"""Gradient-basis SFP with a light additive replay term."""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "cursor"
SETUP = "sfp_grad"


def sfp_grad_replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    replay_beta: float = 0.05,
    **kw,
) -> Tensor:
    """New-task CE + gradient-SFP preservation + small memory replay CE."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if memory_batch is None:
        return ce_loss

    replay_out = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
        labels=memory_batch["labels"],
    )
    replay_loss = replay_out.loss

    if basis is None or anchor_acts is None:
        return ce_loss + replay_beta * replay_loss

    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            if isinstance(output, Tensor):
                h = output
            elif isinstance(output, tuple):
                h = output[0]
            elif hasattr(output, "last_hidden_state"):
                h = output.last_hidden_state
            else:
                h = output[0]
            current_acts[name] = h.float().mean(dim=1)
        return hook

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    finally:
        for handle in handles:
            handle.remove()

    preserve = torch.tensor(0.0, device=ce_loss.device)
    for name in layer_names:
        u_r = basis[name].to(current_acts[name].device)
        projected = current_acts[name] @ u_r
        anchor = anchor_acts[name].to(projected.device)
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])

    return ce_loss + lam * preserve + replay_beta * replay_loss
