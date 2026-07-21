# NLA Hallucination Detection Pipeline

Senior Project — SIIT, Thammasat University

Investigates whether **Natural Language Autoencoders (NLA)** — verbalizing a model's internal
residual-stream activations into natural language, then reconstructing the vector back from that
text — can be used to detect hallucinated LLM answers. If an activation "explains itself" poorly
(low reconstruction fidelity), that may signal the underlying generation was unreliable.

- **Base model:** Qwen2.5-7B-Instruct
- **NLA checkpoints:** `kitft/nla-qwen2.5-7b-L20-av` (verbalizer), `kitft/nla-qwen2.5-7b-L20-ar` (reconstructor/critic)
- **Probed layer:** residual stream after decoder block 20 (`model.model.layers[20]`, pre-final-norm)
- **Eval dataset:** HaluEval QA (`pminervini/HaluEval`, `qa_samples`)
- **In-distribution baseline set:** 50% WildChat-1M + 50% Ultra-FineWeb (reproduces the paper's
  reported `fve_nrm ≈ 0.752`, cosine ≈ 0.890 — used to prove the pipeline itself is correct before
  trusting the HaluEval numbers)

## Repo contents

| File | Purpose |
|---|---|
| `setup.ipynb` | The pipeline, single canonical notebook. Run top to bottom across two kernel sessions (see below). |
| `load_qwen.py` | Minimal standalone script to sanity-load Qwen2.5-7B and check VRAM — useful for a quick pod smoke test before opening the notebook. |

> History note: this notebook absorbed two earlier drafts (`setup.ipynb` v0 exploratory, and
> `setup_refactored.ipynb`). Both are superseded and were removed from the working tree; still
> viewable via `git log` if old approaches need revisiting.

## Environment

Runs on a rented GPU pod (e.g. RunPod/vast.ai) with an empty `/workspace` each session — nothing
persists between pods except what you manually download.

- Combined VRAM need for Session 1 + Session 2 together exceeds 48 GB, so they **must run in
  separate kernels** — restart the kernel between them (the notebook has an explicit cell + warning
  for this).
- `allenai/WildChat-1M` is gated — accept the license on HuggingFace and `huggingface-cli login`
  (or set `HF_TOKEN`) before Session 1B.

## Pipeline stages (in `setup.ipynb`)

1. **Phase 0 — Setup:** environment check, install deps, download Qwen + both NLA checkpoints to `/workspace/models/`.
2. **Session 1A:** extract raw (unnormalized, float32) layer-20 activations for 200 HaluEval QA samples, last-token position. Flags samples with token position < `MIN_POSITION` (50) as immature/attention-sink features.
3. **Session 1B:** same raw extraction over the WildChat+Ultra-FineWeb in-dist set (100 each), sampling one random token per doc at position ≥ 50, to reproduce the baseline and validate the pipeline.
4. *(restart kernel)*
5. **Session 2:** start the SGLang AV server (`nla-av` checkpoint), then for both datasets — verbalize activations to text (greedy decoding, temp=0, with a determinism assertion) and reconstruct back with the AR critic (`nla-ar`), computing `fve_nrm` and mean cosine similarity. Results saved to `/workspace/results_v2.json`.

### Key protocol decisions (why the pipeline looks the way it does)

- **No L2-normalization at rest** — vectors are saved raw; scaling only happens at NLA inference time via `injection_scale`. Earlier drafts normalized at capture time, which turned out to not match the NLA training protocol.
- **`MIN_POSITION = 50`** — early token positions carry attention-sink / immature features and are excluded from the in-dist sampling (and flagged, not silently dropped, in HaluEval).
- **Greedy decoding (temperature=0)** for verbalization, with an explicit assertion that repeated calls are identical — otherwise fidelity metrics aren't reproducible.
- **In-dist run before HaluEval:** if in-dist doesn't land near the ~0.752 / ~0.89 targets, treat that as a pipeline bug to fix first — don't try to interpret HaluEval numbers on top of a broken pipeline.

## Outputs written to `/workspace/`

- `activations_L20_raw.npy`, `labels_L20.npy`, `positions_L20.npy` — HaluEval raw activations/labels/positions
- `activations_indist_raw.npy`, `sources_indist.npy` — in-dist raw activations (0=WildChat, 1=Ultra-FineWeb)
- `explanations_indist.npy`, `explanations_halueval.npy` — AV verbalizations
- `results_v2.json` — final `fve_nrm` / cosine metrics for both sets

Download these off the pod before terminating it — nothing here persists otherwise.

## Open next steps (not yet done as of this notebook)

1. **Paraphrase-ceiling correction** — the raw FVE needs dividing by the ceiling value from the `natural_language_autoencoders` repo's own eval scripts before it's comparable to the reported 0.752; that ceiling constant needs to be located and applied.
2. **Semantic theme analysis** — compare verbalization language/themes between hallucinated vs. truthful HaluEval samples (`explanations_halueval.npy` + `labels_L20.npy`).
3. **Bonus comparison** — re-run the linear probe on raw (non-normalized) layer-20 vectors, to compare against whatever normalized-vector probe result exists from earlier work.
