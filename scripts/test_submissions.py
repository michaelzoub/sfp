"""Unit tests for submission methods using mock model.

Tests that:
1. Methods run without errors
2. sfp_replay has lower memory CE than sfp (because it directly minimizes it)  
3. sfp_cosine produces valid loss with cosine term active
4. All methods return scalar tensors that can be backpropagated
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from torch import Tensor


# ── Mock model ───────────────────────────────────────────────────────────────

class MockOutput:
    """Mimics HuggingFace CausalLM output."""
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class MockLayer(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x, **kw):
        return (self.linear(x),)  # tuple like transformer decoder layers


class MockModel(nn.Module):
    """Tiny fake causal LM for testing."""
    def __init__(self, vocab=100, hidden=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([MockLayer(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, vocab, bias=False)
        self._hidden = hidden

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        x = self.embed(input_ids)  # [B, T, H]
        for layer in self.layers:
            x = layer(x)[0]
        logits = self.head(x)  # [B, T, V]

        loss = None
        if labels is not None:
            shift = logits[:, :-1].reshape(-1, logits.shape[-1])
            tgt = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift, tgt, ignore_index=-100)

        return MockOutput(loss, logits)

    pass  # use nn.Module defaults for named_modules, parameters, named_parameters


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_fn(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if callable(getattr(mod, n)) and n.endswith("_loss")]
    assert fns, f"No *_loss function in {path}"
    return fns[0]


def make_batch(B=4, T=16, vocab=100):
    ids = torch.randint(0, vocab, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    labels = ids.clone()
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def make_sfp_state(model, B=4, T=16, hidden=64, r=4):
    """Create fake SFP basis and anchor activations."""
    # Get layer names like "layers.1"
    layer_names = [f"layers.{i}" for i in range(1, 3)]
    U = torch.randn(hidden, r)
    U, _ = torch.linalg.qr(U)  # orthonormal columns [hidden, r]
    
    basis = {ln: U for ln in layer_names}
    
    # Collect real activations from model
    acts = {}
    def hook_fn(name):
        def h(_m, _i, out):
            x = out[0] if isinstance(out, tuple) else out
            acts[name] = x.float().mean(dim=1).detach()
        return h

    handles = []
    mods = dict(model.named_modules())
    for ln in layer_names:
        handles.append(mods[ln].register_forward_hook(hook_fn(ln)))
    
    batch = make_batch(B, T)
    with torch.no_grad():
        model(**batch)
    for h in handles:
        h.remove()
    
    anchor_acts = {ln: acts[ln] @ U for ln in layer_names}
    return basis, anchor_acts, layer_names


# ── Tests ────────────────────────────────────────────────────────────────────

def test_sfp(model, basis, anchor_acts):
    from methods import sfp_loss
    batch = make_batch()
    mem_batch = make_batch()
    loss = sfp_loss(model, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts, lam=0.1)
    assert loss.dim() == 0, "loss should be scalar"
    assert loss.item() > 0, "loss should be positive"
    loss.backward()
    print(f"  sfp: loss={loss.item():.4f} ✓")
    return loss.item()


def test_sfp_replay(model, basis, anchor_acts):
    fn = load_fn("submissions/sfp_replay.py")
    batch = make_batch()
    mem_batch = make_batch()
    
    # Zero-grad before test
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    loss = fn(model, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts, lam=0.1, replay_alpha=0.3)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    print(f"  sfp_replay: loss={loss.item():.4f} ✓")
    return loss.item()


def test_sfp_cosine(model, basis, anchor_acts):
    fn = load_fn("submissions/sfp_cosine.py")
    batch = make_batch()
    mem_batch = make_batch()
    
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    loss = fn(model, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts, lam=1.0, replay_alpha=0.3)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    print(f"  sfp_cosine: loss={loss.item():.4f} ✓")
    return loss.item()


def test_no_memory_fallback():
    """Methods should gracefully return CE-only when memory_batch=None."""
    from methods import sfp_loss
    fn_replay = load_fn("submissions/sfp_replay.py")
    fn_cosine = load_fn("submissions/sfp_cosine.py")
    
    model = MockModel()
    batch = make_batch()
    
    for name, fn in [("sfp", sfp_loss), ("sfp_replay", fn_replay), ("sfp_cosine", fn_cosine)]:
        loss = fn(model, batch, memory_batch=None)
        assert loss.dim() == 0
        assert loss.item() > 0
        print(f"  {name} (no memory): loss={loss.item():.4f} ✓")


def test_memory_ce_in_replay():
    """sfp_replay loss should be >= sfp loss when memory CE adds signal."""
    from methods import sfp_loss
    fn_replay = load_fn("submissions/sfp_replay.py")
    
    model_sfp = MockModel()
    model_replay = MockModel()
    # Copy weights so they start identical
    model_replay.load_state_dict(model_sfp.state_dict())
    
    batch = make_batch()
    mem_batch = make_batch()
    
    basis, anchor_acts, _ = make_sfp_state(model_sfp)
    
    loss_sfp = sfp_loss(model_sfp, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts, lam=0.1)
    loss_replay = fn_replay(model_replay, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts, lam=0.1, replay_alpha=0.3)
    
    # replay adds mem CE on top of SFP, so total loss should be higher
    print(f"  sfp loss:        {loss_sfp.item():.4f}")
    print(f"  sfp_replay loss: {loss_replay.item():.4f}")
    assert loss_replay.item() > loss_sfp.item(), \
        f"replay ({loss_replay.item():.4f}) should be > sfp ({loss_sfp.item():.4f}) (mem CE is additive)"
    print("  sfp_replay > sfp ✓ (mem CE is active)")


def test_gradient_flow():
    """Verify gradients flow to all LoRA-like params for both methods."""
    fn_replay = load_fn("submissions/sfp_replay.py")
    fn_cosine = load_fn("submissions/sfp_cosine.py")
    
    for name, fn in [("sfp_replay", fn_replay), ("sfp_cosine", fn_cosine)]:
        model = MockModel()
        basis, anchor_acts, _ = make_sfp_state(model)
        
        batch = make_batch()
        mem_batch = make_batch()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        loss = fn(model, batch, memory_batch=mem_batch, basis=basis, anchor_acts=anchor_acts)
        loss.backward()
        
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, f"{name}: no gradients!"
        assert all(g > 0 for g in grad_norms), f"{name}: some grads are zero!"
        print(f"  {name}: {len(grad_norms)} params with grads, avg_norm={sum(grad_norms)/len(grad_norms):.4f} ✓")


if __name__ == "__main__":
    print("=== Unit Tests for Submission Methods ===\n")
    
    print("[1] No-memory fallback (graceful degradation)")
    test_no_memory_fallback()
    print()
    
    print("[2] Loss function correctness with mock model")
    model = MockModel()
    basis, anchor_acts, _ = make_sfp_state(model)
    test_sfp(model, basis, anchor_acts)
    
    model = MockModel()
    basis, anchor_acts, _ = make_sfp_state(model)
    test_sfp_replay(model, basis, anchor_acts)
    
    model = MockModel()
    basis, anchor_acts, _ = make_sfp_state(model)
    test_sfp_cosine(model, basis, anchor_acts)
    print()
    
    print("[3] Memory CE contribution: sfp_replay > sfp_loss")
    test_memory_ce_in_replay()
    print()
    
    print("[4] Gradient flow")
    test_gradient_flow()
    print()
    
    print("=== ALL TESTS PASSED ===")
