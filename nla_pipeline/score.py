"""AR scoring: reconstruct activation vectors from explanations, measure fidelity.

Runs in its own Modal function with only the nla-ar critic on the GPU. On RunPod
this shared a card with the SGLang AV server, which is what made
--mem-fraction-static 0.85 risky.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from . import config as C


def reconstruct_all(critic, explanations: list[str], name: str) -> torch.Tensor:
    """Reconstruct one vector per explanation, in order.

    Sequential on purpose. Unlike AV this is in-process GPU compute, so threads
    buy nothing (GIL plus CUDA serialization). If NLACritic grows a batched
    reconstruct, that is the speedup to reach for here.
    """
    preds = [critic.reconstruct(e) for e in tqdm(explanations, desc=f"AR: {name}")]
    return torch.stack(preds)


def evaluate(critic, explanations, acts, name: str, fve_target, cos_target) -> dict:
    """Compute fve_nrm and mean cosine between reconstructed and gold vectors.

    Both sides are L2-normalized then rescaled by critic.mse_scale. This is
    inference-time scaling and is not in tension with the raw-at-rest protocol:
    the stored vectors stay raw, only the comparison is normalized.

    NOTE: fve here is the RAW figure. It must be divided by the paraphrase
    ceiling from the natural_language_autoencoders eval scripts before it is
    comparable to the reported 0.752 target. See open TODO #1.
    """
    preds = reconstruct_all(critic, explanations, name)

    mse_scale = critic.mse_scale
    gold = torch.tensor(acts, dtype=torch.float32)
    gold_n = gold / gold.norm(dim=-1, keepdim=True) * mse_scale
    pred_n = preds / preds.norm(dim=-1, keepdim=True) * mse_scale

    mu = gold_n.mean(dim=0)
    fve = 1 - ((pred_n - gold_n) ** 2).mean() / ((gold_n - mu) ** 2).mean()
    cos = torch.nn.functional.cosine_similarity(pred_n, gold_n, dim=-1)

    print(f"\n{'=' * 52}")
    print(f"  [{name}]")
    print(f"  fve_nrm  : {fve.item():.4f}   (target: ~{fve_target})")
    print(f"  Mean cos : {cos.mean().item():.4f}   (target: ~{cos_target})")
    print(f"  Mean MSE : {(2 * (1 - cos)).mean().item():.4f}")
    print(f"{'=' * 52}")

    return {"fve": fve.item(), "cos": cos.mean().item(), "n": len(explanations)}


def interpret(fve_indist: float) -> None:
    """Print the in-dist verdict. This gate comes before any HaluEval reading."""
    print("\n[Interpretation]")
    if abs(fve_indist - C.FVE_TARGET) < 0.05:
        print(f"- In-dist {fve_indist:.4f} is near target {C.FVE_TARGET}: pipeline validated.")
        print("  A lower HaluEval number is a genuine distribution-shift finding.")
    else:
        print(f"- In-dist {fve_indist:.4f} MISSES target {C.FVE_TARGET}: treat as a pipeline bug.")
        print("  Check: raw (un-normalized) vectors? greedy decoding? injection_scale?")
        print("  paraphrase-ceiling correction applied? Do not interpret HaluEval yet.")
