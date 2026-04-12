"""SFP + DER++ + EWC-LoRA: five complementary retention signals.

Directly addresses the paper's own stated threats to validity:

  Threat 1 (LoRA confound): EWC-LoRA penalises LoRA weight drift toward
    the previous-task optimum — works in parameter space, not just
    activation space, so forgetting is constrained regardless of whether
    it is a LoRA-subspace artifact.

  Threat 2 (layer sensitivity): SFP weight reduced to 0.05; the
    EWC + logit + CE terms provide retention that does NOT depend on
    which three layer depths were chosen by sfp_setup.

  Threat 3 (task-order): Additive replay + logit distillation are
    symmetric across task orderings (no curriculum dependence).

Loss (fully additive — new-task gradient weight always 1.0):

  L = L_new                                  (plasticity, full weight)
    + 0.5  * CE(model(x_mem), y_mem)         (output retention ~1:1 ratio)
    + 0.20 * MSE(logits_now, logits_stored)   (DER++ dark experience)
    + 0.05 * SFP_preserve                    (geometric retention, 3 layers)
    + 0.10 * EWC_LoRA                        (LoRA weight anchor)

EWC-LoRA: at each task boundary (detected by basis fingerprint change),
snapshot LoRA A/B weights as theta*. Add F.mse_loss(theta, theta*) over
all trainable LoRA params. Uses uniform Fisher (L2) — robust when only
128 memory samples are available for Fisher estimation.

DER++ cache: module-level logit store, keyed by input prefix, auto-reset
at task boundaries.

References:
  DER++:      Buzzega et al. NeurIPS 2020 (arXiv:2004.07211)
  EWC-LoRA:   Zheng et al. ICLR 2026 (arXiv:2602.17559)
  Replay 1:1: Scalable Strategies (arXiv:2505.12512)
  O-LoRA:     Wang et al. EMNLP 2023 (arXiv:2310.14152)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "mzoubkoff"
SETUP = "sfp"

# Module-level state: persists across training steps, reset on task boundary
_s = {
    "fp": None,          # basis fingerprint — changes at each task boundary
    "theta_star": {},    # {param_name: tensor} — LoRA weights at task start
    "logit_cache": {},   # {cache_key: tensor}  — DER++ dark experience
}


def _basis_fp(basis: dict) -> int:
    """Fast fingerprint: id + shape of first basis tensor."""
    if not basis:
        return 0
    v = next(iter(basis.values()))
    return id(v) ^ v.shape[0]


def _reset_on_boundary(model, basis: dict) -> None:
    """Snapshot LoRA theta* when a new task boundary is detected."""
    fp = _basis_fp(basis)
    if _s["fp"] == fp:
        return
    _s["fp"] = fp
    _s["logit_cache"].clear()
    snap = {}
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            snap[name] = p.detach().clone().cpu()
    _s["theta_star"] = snap


def sfp_ewc_der_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.05,
    alpha: float = 0.5,
    beta: float = 0.20,
    lam_ewc: float = 0.10,
    **kw,
) -> Tensor:
    """SFP + DER++ + EWC-LoRA loss."""

    # ── New-task CE (weight always 1.0, never diluted) ─────────────────────
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = out_new.loss

    if memory_batch is None or basis is None or anchor_acts is None:
        return loss

    # ── Task-boundary detection: snapshot theta* for EWC ──────────────────
    _reset_on_boundary(model, basis)

    # ── Collect activations + memory CE in one forward pass ───────────────
    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _hook(name: str):
        def h(_m, _i, output):
            x = output[0] if isinstance(output, tuple) else output
            current_acts[name] = x.float().mean(dim=1)  # [B, hidden]
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

    # ── alpha: Memory CE (output-level retention, ~1:1 ratio) ─────────────
    loss = loss + alpha * out_mem.loss

    # ── beta: DER++ logit distillation (dark experience replay) ───────────
    cache_key = tuple(memory_batch["input_ids"][0, :4].tolist())
    curr_logits_pooled = out_mem.logits.float().mean(dim=1).detach()  # [B, V]

    if cache_key in _s["logit_cache"]:
        stored = _s["logit_cache"][cache_key].to(loss.device)
        live = out_mem.logits.float().mean(dim=1)
        n = min(live.shape[0], stored.shape[0])
        loss = loss + beta * F.mse_loss(live[:n], stored[:n])
    else:
        _s["logit_cache"][cache_key] = curr_logits_pooled.cpu()

    # ── lam: SFP subspace preservation (geometric retention, 3 layers) ────
    preserve = torch.tensor(0.0, device=loss.device)
    for name in layer_names:
        if name not in current_acts:
            continue
        u_r = basis[name].to(current_acts[name].device)   # [hidden, r]
        projected = current_acts[name] @ u_r               # [B, r]
        anchor = anchor_acts[name].to(projected.device)    # [n, r]
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])
    loss = loss + lam * preserve

    # ── lam_ewc: EWC-LoRA (penalise LoRA weight drift from theta*) ────────
    # Directly addresses the LoRA-confound threat: retention in parameter
    # space, not only in activation or logit space.
    if _s["theta_star"]:
        ewc = torch.tensor(0.0, device=loss.device)
        count = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in _s["theta_star"]:
                continue
            star = _s["theta_star"][name].to(p.device)
            ewc = ewc + F.mse_loss(p, star)
            count += 1
        if count > 0:
            loss = loss + lam_ewc * ewc

    return loss
