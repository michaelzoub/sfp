"""SFP + Memory CE replay combined.

SFP preserves the PCA subspace of activations (structural retention).
We also add CE loss on memory samples (output-level retention).
Both share the same memory forward pass, so cost is minimal.

Loss = L_new + replay_alpha * L_mem_CE + sfp_lambda * sum_l ||U_r^T a_l - anchor||^2
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "sfp"


def sfp_replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    replay_alpha: float = 0.3,
    **kw,
) -> Tensor:
    """SFP + replay: structural subspace preservation + explicit CE on memory.

    Combines two complementary retention signals:
    1. CE on memory (output-level): directly prevents forgetting outputs
    2. SFP subspace preservation (feature-level): prevents drift in key activations
    """
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if memory_batch is None:
        return ce_loss

    layer_names = list(basis.keys()) if basis is not None else []
    current_acts: dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            current_acts[name] = out.float().mean(dim=1)
        return hook

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        mem_out = model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
            labels=memory_batch["labels"],
        )
        mem_ce = mem_out.loss
    finally:
        for h in handles:
            h.remove()

    loss = ce_loss + replay_alpha * mem_ce

    if basis is not None and anchor_acts is not None and current_acts:
        preserve = torch.tensor(0.0, device=ce_loss.device)
        for name in layer_names:
            if name not in current_acts:
                continue
            u_r = basis[name].to(current_acts[name].device)
            projected = current_acts[name] @ u_r
            anchor = anchor_acts[name].to(projected.device)
            n = min(projected.shape[0], anchor.shape[0])
            preserve = preserve + F.mse_loss(projected[:n], anchor[:n])
        loss = loss + lam * preserve

    return loss
