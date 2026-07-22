# Phase 0 — NLA Reproduction Setup

---

## Hardware and Precision

| Requirement | Value | Why |
| ----- | ----- | ----- |
| GPU | 48GB class (RTX A6000 or A40) | Session 1 (Qwen alone) needs ~15GB; Session 2 (AV + AR loaded together) needs ~40GB. The two sessions never run concurrently — see "Session split" below. |
| RAM | 50GB | Sufficient for tokenizer, parquet files, and pandas DataFrames |
| Precision | **bf16** compute, **float32 storage** | NLA weights were trained on bf16 activations; extracted vectors are upcast to float32 for storage precision, not normalization |
| Platform | RunPod, PyTorch pre-installed | Confirm actual version on your pod — observed `torch==2.8.0+cu128` on a deployed A40 pod, not always 2.9.1 |

### Session split (why VRAM never adds past 48GB)
- **Session 1** (Qwen2.5-7B-Instruct only, extracting activations): ~15GB
- **Session 2** (SGLang-served AV + AR critic, verbalization/reconstruction): ~40GB, run in a **separate, restarted kernel** after Session 1 finishes and Qwen is freed

## Scope

Using the **released, pre-trained** Qwen2.5-7B-Instruct NLA for inference only. **No training / fine-tuning.** Load the published AV + AR checkpoints and decode activations.

**Repo**: [github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)

**Checkpoints / protocol reference**: [docs/inference.md](https://github.com/kitft/natural_language_autoencoders/blob/main/docs/inference.md) — this is the source of truth for prompt template, injection token IDs, and scale factor (`nla_meta.yaml` per checkpoint); don't hardcode these, load them from the checkpoint's own sidecar file.

[Quick Start](https://github.com/kitft/natural_language_autoencoders#quick-start)

---

## NLA Requirements

### Target LLM

| Spec | Value |
| ----- | ----- |
| Model | Qwen2.5-7B-Instruct (`Qwen/Qwen2.5-7B-Instruct`) |
| Extraction layer | 20/28 |
| d_model | 3584 |
| Activation | Residual-stream output of block 20, **raw / unnormalized** (float32 storage) — **do not L2-normalize at capture time**; that was a v1 mistake that didn't match the NLA training protocol. Normalization only happens at NLA *inference* time via `injection_scale`. |

### NLA checkpoints

| Component | Direction | Checkpoint |
| ----- | ----- | ----- |
| AV (verbalizer) | vector → text | [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) |
| AR (reconstructor) | text → vector | [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) |

Config (prompt template, injection token IDs, scale factor): from each checkpoint's `nla_meta.yaml` — load it, don't reimplement it.

### Runtime

| Spec | Value |
| ----- | ----- |
| Precision | bfloat16 compute; activations stored as float32 (raw, not L2-normalized) |
| Serving | SGLang ≥ 0.5.6, activation injected via `input_embeds` |

### Metrics

These confirm the AV/AR pair is wired up correctly on Qwen2.5-7B-Instruct.

| Metric | Target | What it confirms |
| ----- | ----- | ----- |
| `fve_nrm` | **0.752** in-distribution for Qwen L20 (per checkpoint model card; training set: 50/50 WildChat + Ultra-FineWeb). Reported paper-wide range across released NLAs is 0.6–0.8 — treat 0.752 as *this* checkpoint's specific target, not the range. | The pass/fail check. Run the round trip — activation → AV text explanation → AR reconstruction — and compute Fraction of Variance Explained on the L2-normalized vectors (normalization happens here, at eval time, not at capture). A result well below this signals a setup problem (wrong injection scale, wrong layer index, `nla_meta.yaml` not loaded, or a stray L2-norm at capture time), not a property of the model. |
| cos θ | ~0.9 | Directional agreement between reconstructed and original vectors. |
| MSE_nrm | `2(1 − cos θ)` on L2-normalized vectors (if cos = 0.9 → MSE_nrm = 0.2) | Diagnostic only. Tells you why a low FVE occurred: cos near 0 = reconstruction points in unrelated direction; cos near −1 = pointing the opposite way. |

## Checklist

- [x] Qwen2.5-7B-Instruct weights (`Qwen/Qwen2.5-7B-Instruct`)
- [x] AV checkpoint — `kitft/nla-qwen2.5-7b-L20-av`
- [x] AR checkpoint — `kitft/nla-qwen2.5-7b-L20-ar`
- [x] `nla_meta.yaml` sidecar for each checkpoint (prompt template, injection token IDs, scale factor)
- [x] SGLang ≥ 0.5.6 installed and serving via `input_embeds` (note: the `[all]` pip extra no longer exists on 0.5.6 — plain `pip install sglang==0.5.6` already pulls in `flashinfer-python`)
- [x] bfloat16 inference dtype; float32 storage for extracted activation vectors (raw, not L2-normalized)
- [ ] `fve_nrm` validated against reported 0.752 baseline
