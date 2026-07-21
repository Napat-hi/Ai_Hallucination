"""Dataset registry.

Swapping datasets should mean editing this file and nothing else. Extraction
logic never names a dataset: it reads these specs.

To add a labeled eval dataset:
  1. `modal run modal_app.py --stage inspect --hf-path <repo> --hf-config <cfg>`
     to print the real column names and a sample row.
  2. Add an EvalDataset entry below.
  3. Point ACTIVE_EVAL in config.py at it, or pass --eval-dataset <name> per run.

Artifacts are namespaced by dataset name, so switching does not clobber a
previous run's activations or explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvalDataset:
    """A labeled dataset for the hallucination-detection arm.

    prompt: row -> the text fed to Qwen. This is what gets a vector captured.
    label: row -> 1 hallucinated, 0 truthful. Use `lambda r: -1` for an
        unlabeled exploratory set; downstream label-split analysis will be
        meaningless but extraction and scoring still work.
    position: token index to capture. -1 means last token.
    """

    name: str
    hf_path: str
    prompt: Callable[[dict], str]
    label: Callable[[dict], int]
    hf_config: str | None = None
    split: str = "train"
    n_samples: int = 200
    max_length: int = 512
    position: int = -1
    trust_remote_code: bool = False


@dataclass(frozen=True)
class CorpusSource:
    """One streaming text source for the in-distribution validation set.

    text: row -> flattened document text. Raising inside it skips the row.
    """

    name: str
    hf_path: str
    text: Callable[[dict], str]
    hf_config: str | None = None
    split: str = "train"
    min_chars: int = 400  # cheap pre-filter before paying to tokenize
    oversample: int = 3  # yield this many times the target, token filter drops some
    trust_remote_code: bool = False


# ─── Labeled eval datasets ───────────────────────────────────────────

HALUEVAL_QA = EvalDataset(
    name="halueval_qa",
    hf_path="pminervini/HaluEval",
    hf_config="qa_samples",
    split="data",
    prompt=lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}",
    label=lambda r: 1 if r["hallucination"] == "yes" else 0,
)

# ─── Template: copy, rename, fill in ─────────────────────────────────
# Field names below are placeholders. Confirm them against the real schema with
# `--stage inspect` first, since a wrong key fails only once extraction is
# already running on a GPU.
#
# MY_DATASET = EvalDataset(
#     name="my_dataset",
#     hf_path="org/dataset",
#     hf_config=None,
#     split="validation",
#     prompt=lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}",
#     label=lambda r: int(r["is_hallucinated"]),
#     n_samples=200,
#     max_length=512,
# )

EVAL_DATASETS: dict[str, EvalDataset] = {
    d.name: d
    for d in [
        HALUEVAL_QA,
    ]
}


# ─── In-distribution corpus sources ──────────────────────────────────


def _wildchat_text(row: dict) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in row["conversation"])


WILDCHAT = CorpusSource(
    name="wildchat",
    hf_path="allenai/WildChat-1M",  # gated, needs HF_TOKEN
    text=_wildchat_text,
)

ULTRA_FINEWEB = CorpusSource(
    name="ultrafineweb",
    hf_path="openbmb/Ultra-FineWeb",
    hf_config="en",
    text=lambda r: r.get("content") or r.get("text") or "",
)

CORPUS_SOURCES: dict[str, CorpusSource] = {
    s.name: s for s in [WILDCHAT, ULTRA_FINEWEB]
}

# The in-dist mix, sampled in equal parts. Order defines the integer codes
# written to sources_indist.npy (index into this list).
#
# Changing this invalidates the fve_nrm ~= 0.752 reproduction target, which was
# measured on a 50/50 WildChat + Ultra-FineWeb mix. Adjust FVE_TARGET/COS_TARGET
# in config.py if you change it, or the self-check gate becomes meaningless.
INDIST_MIX: list[str] = ["wildchat", "ultrafineweb"]


# ─── Resolution ──────────────────────────────────────────────────────


def resolve_eval(name: str) -> EvalDataset:
    if name not in EVAL_DATASETS:
        raise KeyError(
            f"unknown eval dataset {name!r}. Registered: {sorted(EVAL_DATASETS)}. "
            f"Add it to EVAL_DATASETS in nla_pipeline/datasets.py."
        )
    return EVAL_DATASETS[name]


def resolve_indist(names: list[str] | None = None) -> list[CorpusSource]:
    names = names or INDIST_MIX
    missing = [n for n in names if n not in CORPUS_SOURCES]
    if missing:
        raise KeyError(
            f"unknown corpus source(s) {missing}. Registered: {sorted(CORPUS_SOURCES)}."
        )
    return [CORPUS_SOURCES[n] for n in names]
