"""Quick benchmark: compare loss functions without full training loop.

Tests sfp, sfp_replay, sfp_cosine on a few steps of training and measures
the memory CE loss (a proxy for retention) and new-task CE loss (plasticity proxy).
"""

import sys
import time
import json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import importlib.util

from model import load_model
from methods import sfp_setup, METHODS
import data


def load_submission(path):
    spec = importlib.util.spec_from_file_location("sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loss_fns = [getattr(mod, n) for n in dir(mod) if callable(getattr(mod, n)) and n.endswith("_loss")]
    return loss_fns[0]


def make_batch(samples, tokenizer, device, max_length=256, batch_size=4):
    subset = samples[:batch_size]
    texts = [f"{s['input']} {s['output']}" for s in subset]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in enc.items()} | {"labels": labels.to(device)}


def run_method(method_name, loss_fn, model, tokenizer, new_batches, mem_batch, basis, anchor_acts, n_steps=20):
    """Run n_steps of training, return avg new-task CE and avg mem CE."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    
    new_losses = []
    mem_losses = []
    
    for i, batch in enumerate(new_batches[:n_steps]):
        optimizer.zero_grad()
        
        # Compute new-task CE separately for comparison
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            new_losses.append(out.loss.item())
        
        # Compute memory CE separately
        if mem_batch is not None:
            with torch.no_grad():
                mem_out = model(
                    input_ids=mem_batch["input_ids"],
                    attention_mask=mem_batch["attention_mask"],
                    labels=mem_batch["labels"],
                )
                mem_losses.append(mem_out.loss.item())
        
        # Full loss for backward
        kw = {"basis": basis, "anchor_acts": anchor_acts, "memory_batch": mem_batch, "lam": 0.1}
        loss = loss_fn(model, batch, **kw)
        loss.backward()
        optimizer.step()
    
    return {
        "avg_new_ce": sum(new_losses) / len(new_losses) if new_losses else 0,
        "avg_mem_ce": sum(mem_losses) / len(mem_losses) if mem_losses else 0,
        "final_new_ce": new_losses[-1] if new_losses else 0,
        "final_mem_ce": mem_losses[-1] if mem_losses else 0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading model...")
    model, tokenizer = load_model(
        "HuggingFaceTB/SmolLM2-135M-Instruct", lora_rank=8, lora_alpha=16
    )
    model.to(device)

    # Load a small dataset
    print("Loading data...")
    math_data = data.load_task("math", split="train")[:50]
    safety_data = data.load_task("safety", split="train")[:50]

    # Memory buffer: math (old task)
    mem_samples = math_data[:16]
    
    # New task batches: safety
    new_batches = [make_batch(safety_data[i:i+4], tokenizer, device) for i in range(0, 40, 4)]
    mem_batch = make_batch(mem_samples, tokenizer, device)

    # Build SFP basis from math memory
    mem_buffers = {"math": mem_samples}
    print("Building SFP basis...")
    t0 = time.time()
    sfp_state = sfp_setup(model, tokenizer, mem_buffers, k=32, r=8)
    print(f"SFP setup: {time.time()-t0:.1f}s")

    results = {}

    methods_to_test = [
        ("sfp", METHODS["sfp"]),
        ("sfp_replay", load_submission("submissions/sfp_replay.py")),
        ("sfp_cosine", load_submission("submissions/sfp_cosine.py")),
    ]

    for name, fn in methods_to_test:
        print(f"\n--- Testing {name} ---")
        # Reload fresh model for fair comparison
        model2, _ = load_model("HuggingFaceTB/SmolLM2-135M-Instruct", lora_rank=8, lora_alpha=16)
        model2.to(device)
        model2.train()
        
        t0 = time.time()
        r = run_method(
            name, fn, model2, tokenizer, new_batches, mem_batch,
            sfp_state["basis"], sfp_state["anchor_acts"],
            n_steps=args.steps
        )
        elapsed = time.time() - t0
        r["elapsed_s"] = round(elapsed, 1)
        results[name] = r
        
        print(f"  avg_new_ce:  {r['avg_new_ce']:.4f}")
        print(f"  avg_mem_ce:  {r['avg_mem_ce']:.4f}")
        print(f"  final_new_ce:{r['final_new_ce']:.4f}")
        print(f"  final_mem_ce:{r['final_mem_ce']:.4f}")
        print(f"  elapsed:     {elapsed:.1f}s")
        
        del model2

    print("\n=== SUMMARY ===")
    print(f"{'Method':<20} {'avg_new_ce':>12} {'avg_mem_ce':>12} {'final_mem_ce':>14}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['avg_new_ce']:>12.4f} {r['avg_mem_ce']:>12.4f} {r['final_mem_ce']:>14.4f}")
    
    print("\nInterpretation:")
    print("  avg_new_ce  → plasticity proxy (lower = better learning)")
    print("  avg_mem_ce  → retention proxy (lower = less forgetting on memory)")
    print("  final_mem_ce → retention at end (lower = less forgetting)")

    os.makedirs("out", exist_ok=True)
    with open("out/bench_loss_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to out/bench_loss_results.json")


if __name__ == "__main__":
    main()
