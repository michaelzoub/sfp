"""SFP with cosine similarity preservation + memory CE replay.

Key differences from vanilla SFP:
1. Cosine loss on projected activations (scale-invariant preservation)
2. CE replay on memory (output-level retention)
3. Higher lambda to compensate for cosine loss scale

Loss = L_new + alpha * L_mem_CE + lam * sum_l (1 - cosine_sim(P_l(a), anchor_l))
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "sfp"


def sfp_cosine_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 1.0,
    replay_alpha: float = 0.3,
    **kw,
) -> Tensor:
    """SFP with cosine preservation loss + memory CE replay.

    Cosine similarity loss is scale-invariant and focuses on directional
    alignment of activations in the preserved subspace.
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
            projected = current_acts[name] @ u_r  # [batch, r]
            anchor = anchor_acts[name].to(projected.device)  # [n, r]
            n = min(projected.shape[0], anchor.shape[0])
            # cosine similarity: 1 - mean cosine → 0 when perfectly aligned
            cos_sim = F.cosine_similarity(projected[:n], anchor[:n], dim=-1)
            preserve = preserve + (1.0 - cos_sim).mean()
        loss = loss + lam * preserve

    return loss
