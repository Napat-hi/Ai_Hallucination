"""Raw layer-20 residual-stream extraction.

Runs inside a Modal GPU container with Qwen2.5-7B-Instruct loaded. The NLA
models are not involved at this stage.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm

from . import config as C


class LayerExtractor:
    """Captures raw residual-stream vectors from one decoder layer.

    The hook fires on `model.model.layers[LAYER_IDX]`, so the captured tensor is
    the residual stream after that block and before the final norm. Vectors are
    kept raw: no L2-normalization at capture time.

    NOT thread-safe. The forward hook writes to a single instance attribute, so
    concurrent extract() calls would race and misattribute vectors to samples.
    Keep the calling loop single-threaded. There is nothing to gain from threads
    anyway, since CUDA kernels serialize on one context.
    """

    def __init__(self, model, tokenizer, layer_idx: int = C.LAYER_IDX):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self._captured: torch.Tensor | None = None
        self._handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Full sequence, raw, moved to CPU as float32.
        self._captured = hidden.detach().float().cpu()  # [1, seq, hidden]

    def __enter__(self) -> "LayerExtractor":
        target = self.model.model.layers[self.layer_idx]
        self._handle = target.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def extract_at_position(
        self,
        text: str,
        position: int = -1,
        max_length: int = C.MAX_LEN_INDIST,
    ) -> tuple[torch.Tensor, int, int]:
        """Run one forward pass. Returns (raw_vector, token_position, seq_len).

        position=-1 selects the last token. Any other value is clamped to the
        final valid index.
        """
        if self._handle is None:
            raise RuntimeError("hook not registered, use LayerExtractor as a context manager")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.model.device)
        self.model(**inputs)

        if self._captured is None:
            raise RuntimeError(f"hook on layer {self.layer_idx} did not fire")

        seq_len = self._captured.shape[1]
        pos = seq_len - 1 if position == -1 else min(position, seq_len - 1)
        return self._captured[0, pos, :].clone(), pos, seq_len


def load_qwen(model_path: str = C.QWEN_PATH):
    """Load Qwen2.5-7B-Instruct in bfloat16 onto the GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Qwen loaded | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


# ─── Labeled eval arm (dataset-agnostic) ─────────────────────────────


def inspect_hf_dataset(hf_path: str, hf_config=None, split="train", trust_remote_code=False):
    """Print columns and one sample row. Use before registering a new dataset.

    Streams, so it does not download the corpus just to read a schema.
    """
    from datasets import load_dataset

    ds = load_dataset(
        hf_path, hf_config, split=split, streaming=True, trust_remote_code=trust_remote_code
    )
    row = next(iter(ds))
    print(f"\n{hf_path} (config={hf_config}, split={split})")
    print(f"columns: {sorted(row)}\n")
    for k, v in row.items():
        preview = str(v).replace("\n", " ")
        print(f"  {k:24s} {type(v).__name__:10s} {preview[:100]}")
    return row


def extract_eval(extractor: LayerExtractor, spec):
    """Extract activations for a labeled eval dataset.

    Returns (activations, labels, positions) as numpy arrays. The prompt
    template and label mapping come from the spec, so this function never names
    a dataset. See nla_pipeline/datasets.py.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        spec.hf_path, spec.hf_config, split=spec.split, trust_remote_code=spec.trust_remote_code
    )
    n = min(spec.n_samples, len(dataset))
    if n < spec.n_samples:
        print(f"WARNING: {spec.name} has only {n} rows, wanted {spec.n_samples}")
    samples = dataset.select(range(n))
    print(f"{spec.name}: {len(samples)} samples")

    acts, labels, positions = [], [], []
    for i, sample in enumerate(tqdm(samples, desc=f"{spec.name} L{extractor.layer_idx} (raw)")):
        try:
            prompt = spec.prompt(sample)
            label = spec.label(sample)
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"{spec.name}: prompt/label failed on row {i}: {e}. "
                f"Available columns: {sorted(sample)}. "
                f"Fix the lambdas in datasets.py."
            ) from e
        vec, pos, _ = extractor.extract_at_position(
            prompt, position=spec.position, max_length=spec.max_length
        )
        acts.append(vec)
        positions.append(pos)
        labels.append(label)

    acts_np = torch.stack(acts).numpy().astype(np.float32)  # RAW
    labels_np = np.array(labels)
    positions_np = np.array(positions)

    if C.FILTER_EVAL_BELOW_MIN_POSITION:
        keep = positions_np >= C.MIN_POSITION
        dropped = int((~keep).sum())
        acts_np, labels_np, positions_np = acts_np[keep], labels_np[keep], positions_np[keep]
        print(f"Filtered {dropped} samples with position < {C.MIN_POSITION}")

    return acts_np, labels_np, positions_np


# ─── In-distribution validation set ──────────────────────────────────


def corpus_texts(source, n: int):
    """Stream documents from one CorpusSource, pre-filtered by raw length."""
    from datasets import load_dataset

    ds = load_dataset(
        source.hf_path,
        source.hf_config,
        split=source.split,
        streaming=True,
        trust_remote_code=source.trust_remote_code,
    )
    count = 0
    for row in ds:
        try:
            text = source.text(row)
        except Exception:
            continue  # schema mismatch on this row, skip it
        if not text or len(text) < source.min_chars:
            continue
        yield text
        count += 1
        if count >= n * source.oversample:  # token-length filter drops some later
            break


def extract_indist_source(extractor: LayerExtractor, text_iter, n_target: int, desc: str):
    """Sample one token per document at a random position >= MIN_POSITION."""
    rng = random.Random(C.SEED)
    vecs, poss = [], []
    pbar = tqdm(total=n_target, desc=desc)
    for text in text_iter:
        if len(vecs) >= n_target:
            break
        # Tokenize first to pick a valid position without paying for a forward
        # pass on a document that turns out to be too short. extract_at_position
        # re-tokenizes, which is cheap next to the forward pass.
        ids = extractor.tokenizer(text, truncation=True, max_length=C.MAX_LEN_INDIST)["input_ids"]
        if len(ids) <= C.MIN_POSITION + 1:
            continue
        pos = rng.randint(C.MIN_POSITION, len(ids) - 1)
        vec, actual_pos, _ = extractor.extract_at_position(
            text, position=pos, max_length=C.MAX_LEN_INDIST
        )
        vecs.append(vec)
        poss.append(actual_pos)
        pbar.update(1)
    pbar.close()

    if len(vecs) < n_target:
        print(f"WARNING: {desc} yielded {len(vecs)}/{n_target}, source exhausted before target")
    return vecs, poss


def extract_indist(extractor: LayerExtractor, sources, n_per_source: int = C.N_INDIST_PER_SOURCE):
    """Build the in-dist validation set from an equal-parts mix of corpus sources.

    `sources` is a list of CorpusSource (see datasets.INDIST_MIX). Returns
    (activations, source_codes) where each code indexes into that list.
    """
    all_vecs, codes = [], []
    for idx, source in enumerate(sources):
        vecs, _ = extract_indist_source(
            extractor, corpus_texts(source, n_per_source), n_per_source, source.name
        )
        all_vecs.extend(vecs)
        codes.extend([idx] * len(vecs))

    if not all_vecs:
        raise RuntimeError("no in-dist vectors extracted, every source came back empty")

    acts = torch.stack(all_vecs).numpy().astype(np.float32)
    counts = ", ".join(f"{s.name} {codes.count(i)}" for i, s in enumerate(sources))
    print(f"In-dist: {counts}")
    return acts, np.array(codes)
