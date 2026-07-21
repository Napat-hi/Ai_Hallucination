# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Senior project (SIIT, Thammasat University) investigating whether **Natural Language Autoencoders
(NLA)** — verbalizing a model's internal residual-stream activation into natural language, then
reconstructing the vector back from that text — can detect hallucinated LLM answers. Poor
reconstruction fidelity on an activation is the hypothesized hallucination signal.

The entire pipeline lives in one notebook, `setup.ipynb`. There is no application code, no test
suite, and no build step — this is a research/experiment repo meant to be run top-to-bottom on a
rented GPU pod (RunPod/vast.ai) with an empty `/workspace` each session.

## Running the pipeline

There's no local dev loop — everything runs on a GPU pod (48GB VRAM class, e.g. RTX A6000) via
Jupyter. `setup.ipynb` runs in **two separate kernel sessions** because Session 1 (Qwen) + Session 2
(SGLang AV + AR) together exceed 48GB combined:

1. **Phase 0 (same kernel as Session 1):** install deps, download models.
   ```bash
   pip install huggingface_hub accelerate datasets scikit-learn tqdm -q
   pip install "sglang[all]==0.5.6" -q
   cd /workspace/natural_language_autoencoders && pip install -e . -q
   ```
   `allenai/WildChat-1M` is gated — `huggingface-cli login` (or set `HF_TOKEN`) before Session 1B.

2. **Session 1A/1B (first kernel):** load Qwen2.5-7B-Instruct, extract raw layer-20 activations for
   HaluEval QA and the in-distribution set. **Restart the kernel** after this (there's an explicit
   cell that frees Qwen and warns to restart) before moving to Session 2.

3. **Session 2 (fresh kernel):** start the SGLang AV server first, then run the verbalization/scoring
   cells:
   ```bash
   python -m sglang.launch_server \
       --model /workspace/models/nla-av \
       --port 30000 \
       --mem-fraction-static 0.85 \
       --disable-radix-cache \
       --trust-remote-code \
       --dtype bfloat16
   ```
   Wait for the server-ready message before running client cells against `http://127.0.0.1:30000`.

- `load_qwen.py` is a standalone smoke test — run it alone to sanity-check Qwen loads and check VRAM
  before opening the full notebook.
- All outputs are written to `/workspace/` (`.npy` activations/labels/positions, `explanations_*.npy`,
  `results_v2.json`) and must be downloaded before terminating the pod — nothing in `/workspace`
  persists otherwise, and `.gitignore` deliberately excludes these files from the repo.

## Architecture / protocol decisions

- **Layer 20 only.** The hook is registered on `model.model.layers[20]`, capturing the residual
  stream *after* block 20 and *before* the final norm — this is the layer both NLA checkpoints
  (`kitft/nla-qwen2.5-7b-L20-av`/`-ar`) were trained on. This is a v2 rewrite whose entire point is
  a "raw vector protocol" — treat these as load-bearing, not incidental:
  - **No L2-normalization at capture time.** Vectors are saved raw (float32); scaling only happens
    at NLA inference time via `injection_scale`. v1 normalized at capture time and that didn't match
    the NLA training protocol — don't reintroduce it.
  - **`MIN_POSITION = 50`.** Token positions below this are attention-sink/immature features. In
    HaluEval extraction they're flagged (not dropped); in the in-distribution sampler they're
    excluded outright.
  - **Greedy decoding (temperature=0)** for AV verbalization, with an explicit determinism assertion
    — otherwise the fidelity metrics (`fve_nrm`, cosine similarity) aren't reproducible run-to-run.
- **In-distribution run is the pipeline's self-check.** The WildChat-1M + Ultra-FineWeb 50/50 mix
  should reproduce the paper's reported `fve_nrm ≈ 0.752`, cosine ≈ 0.890. If it doesn't land near
  those targets, treat that as a pipeline bug to fix *first* — don't interpret HaluEval numbers on
  top of a pipeline that hasn't been validated. Conversely, HaluEval scoring *below* the in-dist
  target is an expected distribution-shift finding, not a bug.
- **`natural_language_autoencoders` is a vendored dependency**, expected at
  `/workspace/natural_language_autoencoders` (installed editable in Phase 0) — `NLAClient` /
  `NLACritic` / `nla_inference.py` come from there, not from this repo.
- Notebook history: `setup.ipynb` is the single canonical, current version — it absorbed and
  superseded two earlier drafts (an exploratory v0 and `setup_refactored.ipynb`), both removed from
  the working tree but recoverable via `git log` if an old approach needs revisiting.

## Open next steps (as of the current notebook)

1. **Paraphrase-ceiling correction** — raw FVE needs dividing by a ceiling constant from
   `natural_language_autoencoders`'s own eval scripts before it's comparable to the reported 0.752;
   that constant still needs to be located and applied.
2. **Semantic theme analysis** — compare AV verbalization language/themes between hallucinated vs.
   truthful HaluEval samples using `explanations_halueval.npy` + `labels_L20.npy`.
3. **Bonus comparison** — re-run the linear probe on raw (non-normalized) layer-20 vectors against
   whatever normalized-vector probe result exists from earlier work.
