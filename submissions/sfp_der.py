"""SFP + DER++: geometric preservation + dark experience replay.

Three additive retention signals (new-task gradient never diluted):

  L = L_new                                          # plasticity (full weight)
    + alpha * CE(model(x_mem), y_mem)                # output-level retention
    + beta  * MSE(logits_now(x_mem), logits_stored)  # DER++ dark experience
    + lam   * sum_l MSE(U_r^T a_l, anchor_l)         # SFP geometric retention

DER++ stores a logit snapshot of memory samples at each task boundary.
The snapshot is carried in a module-level dict (keyed by input hash) that
persists across training steps, reset automatically when basis changes.

Key design choices vs vanilla SFP:
  - Additive, not convex mix: new-task CE weight is always 1.0
  - alpha=0.5 gives ~1:1 new/memory replay ratio (optimal per TiC-LM)
  - beta=0.3 DER++ logit distillation: penalises output drift beyond CE
  - lam=0.1 SFP subspace term unchanged

References:
  DER++: Buzzega et al., NeurIPS 2020 (arXiv:2004.07211)
  Replay ratio ~1:1: Scalable Strategies (arXiv:2505.12512), Watch Your Step (arXiv:2404.10758)
  SFP geometric: this codebase
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "sfp"

# Module-level logit store: maps a cache key to stored logit tensor.
# Reset whenever a new basis is detected (task boundary).
_logit_cache: dict[str, Tensor] = {}
_basis_key: list = [None]   # tracks which task we're on


def _get_basis_fingerprint(basis: dict) -> int:
    """Fast fingerprint of current basis to detect task boundaries."""
    if not basis:
        return 0
    first = next(iter(basis.values()))
    return id(first) + first.shape[0]


def sfp_der_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    alpha: float = 0.5,
    beta: float = 0.3,
    **kw,
) -> Tensor:
    """SFP + DER++: additive combination of geometric + output-level retention."""
    # ── New task CE (full weight) ──────────────────────────────────────────
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out_new.loss

    if memory_batch is None or basis is None or anchor_acts is None:
        return loss

    # ── Detect task boundary; flush logit cache on new basis ──────────────
    fp = _get_basis_fingerprint(basis)
    if _basis_key[0] != fp:
        _basis_key[0] = fp
        _logit_cache.clear()

    # ── Collect current activations + memory CE + stored logits ───────────
    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _hook(name: str):
        def h(_m, _i, output):
            out = output[0] if isinstance(output, tuple) else output
            current_acts[name] = out.float().mean(dim=1)
        return h

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(_hook(name)))

    try:
        out_mem = model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
            labels=memory_batch["labels"],
        )
    finally:
        for h in handles:
            h.remove()

    # ── alpha: Memory CE (~1:1 replay ratio) ──────────────────────────────
    loss = loss + alpha * out_mem.loss

    # ── beta: DER++ logit distillation ────────────────────────────────────
    # Build a cache key from the memory input ids (first 4 tokens as int tuple)
    cache_key = tuple(memory_batch["input_ids"][0, :4].tolist())
    curr_logits = out_mem.logits.float().mean(dim=1).detach()  # [B, V]

    if cache_key in _logit_cache:
        stored = _logit_cache[cache_key].to(curr_logits.device)
        n = min(curr_logits.shape[0], stored.shape[0])
        # MSE on mean-pooled logits (cheaper than per-token; captures output drift)
        loss = loss + beta * F.mse_loss(
            out_mem.logits.float().mean(dim=1)[:n],
            stored[:n],
        )
    else:
        # First time seeing this memory batch: store logits as "dark experience"
        _logit_cache[cache_key] = curr_logits.cpu()

    # ── lam: SFP subspace preservation ────────────────────────────────────
    preserve = torch.tensor(0.0, device=loss.device)
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
