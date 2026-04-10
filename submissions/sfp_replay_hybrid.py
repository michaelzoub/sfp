"""SFP plus small memory replay CE — same PCA preservation as sfp, adds replay for retention."""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "sfp"


def sfp_replay_hybrid_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    replay_ratio: float = 0.08,
    **kw,
) -> Tensor:
    """L = (1-α)·CE_new + α·CE_mem + λ·preservation."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_new = out_new.loss

    if basis is None or anchor_acts is None or memory_batch is None:
        return ce_new

    layer_names = list(basis.keys())
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
        mem_kw = dict(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
        if memory_batch.get("labels") is not None:
            mem_kw["labels"] = memory_batch["labels"]
        out_mem = model(**mem_kw)
        ce_mem = out_mem.loss if hasattr(out_mem, "loss") else ce_new
    finally:
        for h in handles:
            h.remove()

    preserve = torch.tensor(0.0, device=ce_new.device)
    for name in layer_names:
        u_r = basis[name].to(current_acts[name].device)
        projected = current_acts[name] @ u_r
        anchor = anchor_acts[name].to(projected.device)
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])

    ce = (1.0 - replay_ratio) * ce_new + replay_ratio * ce_mem
    return ce + lam * preserve
