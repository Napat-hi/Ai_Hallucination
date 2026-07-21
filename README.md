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
| `modal_app.py` | Modal orchestration. Defines the image, the persistent Volume, and one container per pipeline stage. This is the entrypoint. |
| `nla_pipeline/datasets.py` | Dataset registry. **Swapping datasets means editing this file and nothing else.** |
| `nla_pipeline/config.py` | Protocol constants, artifact paths, and which dataset is active. |
| `nla_pipeline/extract.py` | Layer-20 activation capture. Dataset-agnostic: it reads specs from the registry. |
| `nla_pipeline/verbalize.py` | SGLang AV server lifecycle, concurrent verbalization, injection-failure guard. |
| `nla_pipeline/score.py` | AR reconstruction and the `fve_nrm` / cosine metrics. |
| `load_qwen.py` | Standalone Qwen load + VRAM smoke test. |
| `setup.ipynb` | Superseded RunPod notebook, kept as the record of the original two-kernel protocol. |

## Running it

Requires a [Modal](https://modal.com) account. One-time setup:

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...
```

`allenai/WildChat-1M` is gated: accept its license on HuggingFace before running the in-dist stage.

`natural_language_autoencoders` is a private package, not on PyPI. Copy it to
`vendor/natural_language_autoencoders` so the image build can install it. See the block comment at
the top of `modal_app.py` for the Volume-based alternative.

```bash
modal run modal_app.py                    # full pipeline
modal run modal_app.py --stage download   # weights only
modal run modal_app.py --stage extract    # both extraction jobs, concurrent
modal run modal_app.py --stage verbalize
modal run modal_app.py --stage score
```

Stages hand off through the Volume, so any stage can be re-run alone without repeating the ones
before it.

## Changing the dataset

Datasets are declarative specs in `nla_pipeline/datasets.py`. Extraction never names a dataset, so
adding one touches no pipeline logic.

1. Find the real column names. A wrong key otherwise fails only after a GPU has spun up:

   ```bash
   modal run modal_app.py --stage inspect --hf-path org/dataset --split validation
   ```

2. Add an `EvalDataset` to the registry. Copy the commented template in `datasets.py`:

   ```python
   MY_DATASET = EvalDataset(
       name="my_dataset",
       hf_path="org/dataset",
       split="validation",
       prompt=lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}",
       label=lambda r: int(r["is_hallucinated"]),   # 1 hallucinated, 0 truthful
   )
   ```

   Then add it to the `EVAL_DATASETS` list.

3. Run it, either per-invocation or by changing `ACTIVE_EVAL` in `config.py`:

   ```bash
   modal run modal_app.py --eval-dataset my_dataset
   ```

Artifacts are namespaced by dataset name, so switching never clobbers a previous run.

To change the **in-dist mix** instead, add a `CorpusSource` and edit `INDIST_MIX`. Note that the
`fve_nrm ≈ 0.752` target was measured on the 50/50 WildChat + Ultra-FineWeb mix, so changing the mix
invalidates that gate unless you also update `FVE_TARGET` / `COS_TARGET`.

## Pipeline stages

| Stage | GPU | Work |
|---|---|---|
| `inspect_dataset` | none | Print a dataset's columns and a sample row. Use before registering one. |
| `download_models` | none | Fetch Qwen + both NLA checkpoints into the Volume. Idempotent. |
| `extract_eval` | A10G | Raw float32 layer-20 activations for the active eval dataset. |
| `extract_indist` | A10G | Same capture over the `INDIST_MIX` corpora, one random token per doc at position >= 50. |
| `verbalize` | A100-40GB | Boot SGLang on `nla-av`, verbalize both arms with greedy decoding. |
| `score` | A10G | Reconstruct with `nla-ar`, compute `fve_nrm` and cosine, write `results_<dataset>.json`. |

Each stage is its own container, so Qwen and SGLang never coexist. The two-kernel split the notebook
required was an artifact of one 48 GB pod, not a protocol decision. Splitting AV from AR also means
neither shares a card with the other.

### Key protocol decisions (why the pipeline looks the way it does)

- **No L2-normalization at rest** — vectors are saved raw; scaling only happens at NLA inference time via `injection_scale`. Earlier drafts normalized at capture time, which turned out to not match the NLA training protocol.
- **`MIN_POSITION = 50`** — early token positions carry attention-sink / immature features and are excluded from the in-dist sampling (and flagged, not silently dropped, on the eval arm).
- **Greedy decoding (temperature=0)** for verbalization, with an explicit assertion that repeated calls are identical — otherwise fidelity metrics aren't reproducible.
- **In-dist run before the eval set:** if in-dist doesn't land near the ~0.752 / ~0.89 targets, treat that as a pipeline bug to fix first — don't try to interpret eval numbers on top of a broken pipeline.

## Outputs written to `/workspace/`

`<name>` is a registered dataset name (`halueval_qa` by default) or `indist`.

- `activations_<name>_L20_raw.npy` — raw float32 layer-20 activations
- `labels_<name>.npy`, `positions_<name>.npy` — eval arm only (1 = hallucinated, 0 = truthful)
- `sources_<name>.npy` — in-dist only, each value indexes into `INDIST_MIX`
- `explanations_<name>.npy` — AV verbalizations
- `results_<eval-dataset>.json` — `fve_nrm` / cosine for both arms

Namespacing by dataset means several datasets' artifacts coexist on the Volume without overwriting
each other.

These live on the `nla-workspace` Modal Volume and persist between runs. Pull them down with:

```bash
modal volume get nla-workspace results_v2.json .
modal volume get nla-workspace 'explanations_*.npy' .
```

## Open next steps (not yet done as of this notebook)

1. **Paraphrase-ceiling correction** — the raw FVE needs dividing by the ceiling value from the `natural_language_autoencoders` repo's own eval scripts before it's comparable to the reported 0.752; that ceiling constant needs to be located and applied.
2. **Semantic theme analysis** — compare verbalization language/themes between hallucinated vs. truthful HaluEval samples (`explanations_halueval.npy` + `labels_L20.npy`).
3. **Bonus comparison** — re-run the linear probe on raw (non-normalized) layer-20 vectors, to compare against whatever normalized-vector probe result exists from earlier work.
