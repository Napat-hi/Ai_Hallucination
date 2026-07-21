# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Senior project (SIIT, Thammasat University) investigating whether **Natural Language Autoencoders
(NLA)** — verbalizing a model's internal residual-stream activation into natural language, then
reconstructing the vector back from that text — can detect hallucinated LLM answers. Poor
reconstruction fidelity on an activation is the hypothesized hallucination signal.

The pipeline runs on **Modal** (`modal_app.py` + `nla_pipeline/`). There is no application code, no
test suite, and no build step: this is a research/experiment repo. `setup.ipynb` is the superseded
RunPod version, kept as the record of the original two-kernel protocol.

## Running the pipeline

There is no local dev loop. Nothing runs on a laptop: every stage is a Modal container.

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...      # WildChat-1M is gated

modal run modal_app.py                    # full pipeline
modal run modal_app.py --stage verbalize  # or any single stage
```

Five stages, one container each, handing off through a persistent Volume mounted at `/workspace`:

| Stage | GPU | Work |
|---|---|---|
| `inspect_dataset` | none | Print a dataset's columns + sample row. Precedes registering a new one. |
| `download_models` | none | Qwen + both NLA checkpoints into the Volume. Idempotent. |
| `extract_eval` | A10G | Raw layer-20 activations for the active eval dataset. |
| `extract_indist` | A10G | Same capture over the `INDIST_MIX` corpora. |
| `verbalize` | A100-40GB | SGLang on `nla-av`, greedy decoding, both arms. |
| `score` | A10G | `nla-ar` reconstruction, `fve_nrm` + cosine, writes `results_<dataset>.json`. |

Consequences of the container split, all deliberate:

- **The two-kernel session split is gone.** It was an artifact of one 48GB pod holding Qwen and
  SGLang at once, not a protocol decision. Do not reintroduce it.
- **AV and AR are separate stages.** Neither shares a card, so `--mem-fraction-static 0.85` is safe
  now. On the pod it was an OOM risk against a co-resident 7B critic.
- **Weights download once.** The Volume persists, unlike the pod's empty `/workspace` each session.
- **Verbalization is concurrent** (`AV_CONCURRENCY`, default 16). Safe because it is HTTP to a
  separate SGLang process. Extraction stays single-threaded: its forward hook holds per-instance
  state that concurrent calls would race on, and CUDA serializes anyway.

`load_qwen.py` is a standalone smoke test for checking Qwen loads and VRAM.

Artifacts land on the `nla-workspace` Volume and persist between runs. Retrieve with
`modal volume get nla-workspace <file> .`. `.gitignore` deliberately excludes them from the repo.

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
- **Datasets are declarative, in `nla_pipeline/datasets.py`.** `EvalDataset` (labeled arm) and
  `CorpusSource` (in-dist arm) carry the HF coordinates plus `prompt` / `label` / `text` lambdas.
  `extract.py` never names a dataset, so adding one touches no pipeline logic: register a spec, then
  either set `ACTIVE_EVAL` in config or pass `--eval-dataset <name>`. Use
  `--stage inspect --hf-path <repo>` to get real column names first; a wrong key otherwise fails
  only after a GPU has spun up.
- **Artifacts are namespaced by dataset name** (`activations_<name>_L20_raw.npy`,
  `explanations_<name>.npy`, `results_<eval>.json`). Switching datasets cannot clobber a prior run,
  and several datasets' outputs coexist on the Volume. Note these filenames differ from the
  notebook's flat `activations_L20_raw.npy` / `labels_L20.npy` / `results_v2.json`.
- **Changing `INDIST_MIX` invalidates the reproduction gate.** The `fve_nrm ≈ 0.752` target was
  measured on the 50/50 WildChat + Ultra-FineWeb mix. Update `FVE_TARGET` / `COS_TARGET` alongside
  it or the self-check becomes meaningless.
- **Remaining constants live in `nla_pipeline/config.py`**, not scattered through the stages.
  `FILTER_EVAL_BELOW_MIN_POSITION` is the one open knob: it defaults to `False` to match the
  notebook's flag-don't-drop behaviour, but the in-dist sampler enforces `MIN_POSITION` strictly, so
  the two arms are not position-matched until it is `True`.
- **`natural_language_autoencoders` is a vendored private dependency** supplying `NLAClient`,
  `NLACritic`, and `nla_inference.py` (repo root, not inside `nla/`). It is not on PyPI. Keep a copy
  at `vendor/natural_language_autoencoders`; the image bakes it in and installs it editable at
  `/opt/nla`. See the block comment at the top of `modal_app.py` for the Volume-based alternative.
- **Known injection failure signature:** `nla/injection.py` documents that if injection misses the
  marked position, the model emits the literal ㊗ character and outputs Chinese.
  `verbalize.check_injection_health` samples for this and warns, so it fails loudly instead of
  producing plausible-looking garbage metrics.
- History: `setup.ipynb` is the superseded RunPod notebook, kept as the record of the two-kernel
  protocol. It absorbed two earlier drafts (an exploratory v0 and `setup_refactored.ipynb`), both
  removed from the working tree but recoverable via `git log`.

## Open next steps (as of the current notebook)

1. **Paraphrase-ceiling correction** — raw FVE needs dividing by a ceiling constant from
   `natural_language_autoencoders`'s own eval scripts before it's comparable to the reported 0.752;
   that constant still needs to be located and applied.
2. **Semantic theme analysis** — compare AV verbalization language/themes between hallucinated vs.
   truthful HaluEval samples using `explanations_halueval.npy` + `labels_L20.npy`.
3. **Bonus comparison** — re-run the linear probe on raw (non-normalized) layer-20 vectors against
   whatever normalized-vector probe result exists from earlier work.
