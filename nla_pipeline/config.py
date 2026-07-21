"""Shared constants for the NLA hallucination-detection pipeline.

Most values here are protocol decisions, not tunables. Read CLAUDE.md before
changing LAYER_IDX, MIN_POSITION, or anything under "Protocol".
"""

# ─── Protocol (load-bearing, see CLAUDE.md) ──────────────────────────
LAYER_IDX = 20
MIN_POSITION = 50  # below this: attention-sink / immature features

# v1 L2-normalized at capture time and that did not match the NLA training
# protocol. Vectors are stored raw (float32); scaling happens at NLA inference
# time via injection_scale. Do not reintroduce capture-time normalization.
NORMALIZE_AT_CAPTURE = False

# The eval arm currently flags short-position samples rather than dropping them,
# matching setup.ipynb. Set True to exclude them instead, which makes the eval
# set position-matched against the in-dist set.
FILTER_EVAL_BELOW_MIN_POSITION = False

# ─── Dataset selection ───────────────────────────────────────────────
# Which registered dataset the eval arm uses. See nla_pipeline/datasets.py for
# the registry and for how to add one. Override per run without editing this:
#     modal run modal_app.py --eval-dataset <name>
ACTIVE_EVAL = "halueval_qa"

# Per-dataset knobs (n_samples, max_length, prompt, label) live on the
# EvalDataset spec in datasets.py, not here.

# ─── In-dist sampling ────────────────────────────────────────────────
N_INDIST_PER_SOURCE = 100  # per source, so 200 total across a 2-source mix
MAX_LEN_INDIST = 1024
SEED = 42
INDIST_NAME = "indist"  # artifact namespace for the validation arm

# ─── In-dist reproduction targets (the pipeline's self-check) ────────
# If the in-dist run misses these, treat it as a pipeline bug and fix that
# before interpreting any HaluEval number.
FVE_TARGET = 0.752
COS_TARGET = 0.890

# ─── Paths inside the Modal Volume ───────────────────────────────────
# Deliberately identical to the RunPod layout so saved artifacts and any
# downstream analysis keep working unchanged.
WORKSPACE = "/workspace"
MODELS_DIR = f"{WORKSPACE}/models"
QWEN_PATH = f"{MODELS_DIR}/qwen2.5-7b-instruct"
AV_PATH = f"{MODELS_DIR}/nla-av"
AR_PATH = f"{MODELS_DIR}/nla-ar"

# Where the vendored natural_language_autoencoders package lands in the image.
NLA_SRC = "/opt/nla"

HF_REPOS = {
    "Qwen/Qwen2.5-7B-Instruct": QWEN_PATH,
    "kitft/nla-qwen2.5-7b-L20-av": AV_PATH,
    "kitft/nla-qwen2.5-7b-L20-ar": AR_PATH,
}

# ─── Output artifacts ────────────────────────────────────────────────
# Every artifact is namespaced by dataset name so switching datasets cannot
# silently clobber a previous run. `name` is a registered eval dataset name or
# INDIST_NAME.


def acts_path(name: str) -> str:
    return f"{WORKSPACE}/activations_{name}_L{LAYER_IDX}_raw.npy"


def labels_path(name: str) -> str:
    return f"{WORKSPACE}/labels_{name}.npy"


def positions_path(name: str) -> str:
    return f"{WORKSPACE}/positions_{name}.npy"


def sources_path(name: str) -> str:
    """Per-document source index, in-dist arm only. Values index INDIST_MIX."""
    return f"{WORKSPACE}/sources_{name}.npy"


def explanations_path(name: str) -> str:
    return f"{WORKSPACE}/explanations_{name}.npy"


def results_path(eval_name: str) -> str:
    return f"{WORKSPACE}/results_{eval_name}.json"


# ─── SGLang (AV server) ──────────────────────────────────────────────
SGLANG_PORT = 30000
SGLANG_URL = f"http://127.0.0.1:{SGLANG_PORT}"

# AV and AR run in separate Modal functions, so SGLang gets the whole card and
# does not have to leave room for the AR critic. On RunPod both shared one 48GB
# A40, which made 0.85 a real OOM risk.
SGLANG_MEM_FRACTION = 0.85

# Verbalization is HTTP to a separate server process, so concurrency is safe
# here. This is the pipeline's dominant cost when run serially.
AV_CONCURRENCY = 16

# Startup can include a long CUDA-graph capture on a cold container.
SGLANG_STARTUP_TIMEOUT_S = 900
