"""Modal orchestration for the NLA hallucination-detection pipeline.

Replaces the two-kernel RunPod notebook. Each stage is an independent container,
so Qwen and SGLang never coexist and the 48GB ceiling that forced the kernel
split no longer applies.

    modal run modal_app.py                    # full pipeline
    modal run modal_app.py --stage download   # one stage
    modal run modal_app.py --stage verbalize

Stages write to a persistent Volume at /workspace, using the same filenames as
setup.ipynb. Weights download once, not once per session.

One-time setup:
    pip install modal && modal setup
    modal secret create huggingface HF_TOKEN=hf_...
    # WildChat-1M is gated: accept its license on HuggingFace first.
"""

import pathlib

import modal

from nla_pipeline import config as C

# ─────────────────────────────────────────────────────────────────────
# VENDORED DEPENDENCY  <-- the one thing you must point at your own copy
#
# natural_language_autoencoders is a private research package, not on PyPI.
# It lived at /workspace/natural_language_autoencoders on the RunPod pod.
# Copy it off the pod to the path below, e.g.
#     runpodctl receive <code>       # or scp -r from the pod
#     mv natural_language_autoencoders vendor/
#
# ALTERNATIVE if you cannot keep a local copy: put it on its own Volume once
#     modal volume create nla-src
#     modal volume put nla-src ./natural_language_autoencoders /
# then drop the add_local_dir + `pip install -e` lines below, mount
# {C.NLA_SRC: modal.Volume.from_name("nla-src")} on the verbalize/score
# functions, and rely on the sys.path.insert those functions already do.
# ─────────────────────────────────────────────────────────────────────
NLA_LOCAL_DIR = "vendor/natural_language_autoencoders"

if not (pathlib.Path(NLA_LOCAL_DIR) / "nla_inference.py").exists():
    raise SystemExit(
        f"Missing vendored dependency: {NLA_LOCAL_DIR}/nla_inference.py\n\n"
        "natural_language_autoencoders is a private package, not on PyPI. It lived at\n"
        "/workspace/natural_language_autoencoders on the RunPod pod. Copy it here:\n\n"
        f"    mkdir -p vendor && cp -r <source> {NLA_LOCAL_DIR}\n\n"
        "See the block comment above this check for the Volume-based alternative."
    )

app = modal.App("nla-hallucination")

# Persistent /workspace. Survives runs, unlike the RunPod pod disk.
workspace = modal.Volume.from_name("nla-workspace", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # sglang pins torch, so install it first and let the rest resolve around it.
    .pip_install("sglang[all]==0.5.6")
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
        "datasets",
        "scikit-learn",
        "tqdm",
        "requests",
        "numpy",
    )
    .env(
        {
            "HF_HOME": f"{C.WORKSPACE}/hf_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Rust downloader, per-connection throughput
        }
    )
    .add_local_dir(
        NLA_LOCAL_DIR,
        C.NLA_SRC,
        copy=True,  # must land before the editable install below
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/.venv"],
    )
    .run_commands(f"pip install -e {C.NLA_SRC}")
    .add_local_python_source("nla_pipeline")
)

VOLUMES = {C.WORKSPACE: workspace}
HOUR = 3600


# ─── Stage 1: model download (CPU only, no GPU billing) ──────────────


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=2 * HOUR)
def download_models():
    """Fetch Qwen and both NLA checkpoints into the Volume. Idempotent."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import snapshot_download

    def download_one(repo_id, local_dir):
        if os.path.exists(local_dir) and os.listdir(local_dir):
            return repo_id, "skipped"
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        return repo_id, "downloaded"

    errors = {}
    with ThreadPoolExecutor(max_workers=len(C.HF_REPOS)) as pool:
        futures = {
            pool.submit(download_one, repo, path): repo for repo, path in C.HF_REPOS.items()
        }
        for future in as_completed(futures):
            repo = futures[future]
            try:
                _, status = future.result()
                print(f"{status}: {repo}")
            except Exception as e:
                errors[repo] = e
                print(f"FAILED: {repo} -> {e}")

    if errors:
        raise RuntimeError(f"{len(errors)} model download(s) failed: {list(errors)}")

    workspace.commit()
    for path in C.HF_REPOS.values():
        n = len(os.listdir(path)) if os.path.exists(path) else 0
        print(f"  {path.split('/')[-1]:35s} ({n} files)")


# ─── Stage 2: activation extraction (Qwen only, ~15GB) ───────────────


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=HOUR)
def inspect_dataset(hf_path: str, hf_config: str = "", split: str = "train"):
    """Print a dataset's columns and one sample row. CPU only, no GPU billing.

    Run this before registering a new dataset so the prompt/label lambdas use
    real column names instead of guesses.
    """
    from nla_pipeline.extract import inspect_hf_dataset

    inspect_hf_dataset(hf_path, hf_config or None, split)


@app.function(image=image, gpu="A10G", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * HOUR)
def extract_eval(eval_name: str):
    """Raw layer-20 activations for a registered labeled eval dataset."""
    import numpy as np

    from nla_pipeline.datasets import resolve_eval
    from nla_pipeline.extract import LayerExtractor, extract_eval, load_qwen

    spec = resolve_eval(eval_name)
    model, tokenizer = load_qwen()
    with LayerExtractor(model, tokenizer) as ext:
        acts, labels, positions = extract_eval(ext, spec)

    np.save(C.acts_path(spec.name), acts)
    np.save(C.labels_path(spec.name), labels)
    np.save(C.positions_path(spec.name), positions)
    workspace.commit()

    n_short = int((positions < C.MIN_POSITION).sum())
    print(f"\n[{spec.name}] saved RAW activations: {acts.shape} | dtype {acts.dtype}")
    print(f"  Mean vector norm: {np.linalg.norm(acts, axis=-1).mean():.2f} (>> 1 if raw)")
    print(f"  Label 1: {int((labels == 1).sum())} | Label 0: {int((labels == 0).sum())}")
    print(f"  Samples with position < {C.MIN_POSITION}: {n_short}/{len(positions)}")
    if n_short and not C.FILTER_EVAL_BELOW_MIN_POSITION:
        print(
            "  NOTE: the in-dist set enforces position >= 50 strictly while these are only\n"
            "  flagged, so the two sets are not position-matched. Set\n"
            "  FILTER_EVAL_BELOW_MIN_POSITION=True in config.py to match them."
        )
    return {"dataset": spec.name, "n": int(len(labels)), "n_short": n_short}


@app.function(image=image, gpu="A10G", volumes=VOLUMES, secrets=[hf_secret], timeout=2 * HOUR)
def extract_indist():
    """Raw layer-20 activations for the in-dist validation mix (datasets.INDIST_MIX)."""
    import numpy as np

    from nla_pipeline.datasets import INDIST_MIX, resolve_indist
    from nla_pipeline.extract import LayerExtractor, extract_indist, load_qwen

    sources = resolve_indist()
    print(f"In-dist mix: {INDIST_MIX}")

    model, tokenizer = load_qwen()
    with LayerExtractor(model, tokenizer) as ext:
        acts, codes = extract_indist(ext, sources)

    np.save(C.acts_path(C.INDIST_NAME), acts)
    np.save(C.sources_path(C.INDIST_NAME), codes)
    workspace.commit()

    print(f"\nIn-dist set: {acts.shape} | all positions >= {C.MIN_POSITION}")
    print(f"  Mean vector norm: {np.linalg.norm(acts, axis=-1).mean():.2f}")
    print(f"  Source codes index INDIST_MIX: {dict(enumerate(INDIST_MIX))}")
    return {"n": int(len(codes))}


# ─── Stage 3: AV verbalization (SGLang alone on the card) ────────────


@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, timeout=4 * HOUR)
def verbalize(eval_name: str):
    """Verbalize the in-dist and eval activation sets via the AV checkpoint."""
    import sys

    import numpy as np

    sys.path.insert(0, C.NLA_SRC)

    from nla_pipeline.verbalize import assert_deterministic, make_verbalizer, run_av, serve_av

    # In-dist first: it is the run that validates the pipeline.
    arms = [C.INDIST_NAME, eval_name]
    acts_by_arm = {name: np.load(C.acts_path(name)) for name in arms}
    for name, acts in acts_by_arm.items():
        print(f"{name}: {acts.shape}")

    with serve_av() as url:
        from nla_inference import NLAClient

        client = NLAClient(checkpoint_dir=C.AV_PATH, sglang_url=url)
        verbalizer = make_verbalizer(client)

        # Prove greedy decoding is on before paying for the full run.
        assert_deterministic(verbalizer, acts_by_arm[arms[0]][0])

        for name in arms:
            exps = run_av(verbalizer, acts_by_arm[name], name)
            np.save(C.explanations_path(name), np.array(exps, dtype=object))
            workspace.commit()
            print(f"Saved {len(exps)} -> {C.explanations_path(name)}")


# ─── Stage 4: AR scoring (critic alone on the card) ──────────────────


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=2 * HOUR)
def score(eval_name: str):
    """Reconstruct vectors from explanations and report fidelity for both arms."""
    import json
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, C.NLA_SRC)

    from nla_pipeline.datasets import INDIST_MIX
    from nla_pipeline.score import evaluate, interpret

    def load_arm(name):
        return (
            np.load(C.acts_path(name)),
            np.load(C.explanations_path(name), allow_pickle=True).tolist(),
        )

    ind_acts, exps_ind = load_arm(C.INDIST_NAME)
    eval_acts, exps_eval = load_arm(eval_name)

    from nla_inference import NLACritic

    critic = NLACritic(checkpoint_dir=C.AR_PATH, device="cuda", dtype=torch.bfloat16)

    ind = evaluate(
        critic, exps_ind, ind_acts, f"IN-DIST ({'+'.join(INDIST_MIX)})", C.FVE_TARGET, C.COS_TARGET
    )
    ev = evaluate(critic, exps_eval, eval_acts, f"{eval_name} (OOD)", "n/a", "n/a")
    interpret(ind["fve"])

    results = {
        "eval_dataset": eval_name,
        "indist_mix": INDIST_MIX,
        "fve_indist": ind["fve"],
        "cos_indist": ind["cos"],
        "fve_eval": ev["fve"],
        "cos_eval": ev["cos"],
        "note": "fve is RAW, divide by the paraphrase ceiling before comparing to 0.752",
    }
    out = C.results_path(eval_name)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    workspace.commit()
    print(f"\nSaved -> {out}")
    return results


# ─── Orchestration ───────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    stage: str = "all",
    eval_dataset: str = "",
    hf_path: str = "",
    hf_config: str = "",
    split: str = "train",
):
    """Run one stage or the whole pipeline.

    Stages hand off through the Volume, so any stage can be re-run alone without
    repeating the ones before it.

        modal run modal_app.py
        modal run modal_app.py --eval-dataset my_dataset
        modal run modal_app.py --stage verbalize --eval-dataset my_dataset
        modal run modal_app.py --stage inspect --hf-path org/dataset --split validation
    """
    from nla_pipeline.datasets import resolve_eval

    if stage == "inspect":
        if not hf_path:
            raise SystemExit("--stage inspect requires --hf-path <hf repo id>")
        inspect_dataset.remote(hf_path, hf_config, split)
        return

    stages = ["download", "extract", "verbalize", "score"]
    if stage not in stages + ["all"]:
        raise SystemExit(
            f"unknown stage {stage!r}, expected one of {stages + ['all', 'inspect']}"
        )
    run = stages if stage == "all" else [stage]

    # Resolve locally so an unregistered name fails instantly instead of after a
    # container has already spun up.
    name = eval_dataset or C.ACTIVE_EVAL
    resolve_eval(name)
    print(f"Eval dataset: {name}")

    if "download" in run:
        download_models.remote()
    if "extract" in run:
        # Independent of each other, so run both containers concurrently.
        handles = [extract_eval.spawn(name), extract_indist.spawn()]
        for h in handles:
            print(h.get())
    if "verbalize" in run:
        verbalize.remote(name)
    if "score" in run:
        print(score.remote(name))
