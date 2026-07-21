"""AV verbalization: raw activation vector to natural-language explanation.

Talks to a local SGLang server serving the nla-av checkpoint. Because that
server is a separate process, verbalization is network I/O and safe to run
concurrently, unlike the extraction stage.
"""

from __future__ import annotations

import contextlib
import inspect
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from . import config as C


@contextlib.contextmanager
def serve_av(model_path: str = C.AV_PATH, port: int = C.SGLANG_PORT):
    """Launch the SGLang AV server, wait for readiness, tear it down on exit.

    --disable-radix-cache is deliberate: every request injects a different
    activation vector, so reusing a cached shared prefix would be incorrect.
    """
    import requests

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model_path,
        "--port", str(port),
        "--mem-fraction-static", str(C.SGLANG_MEM_FRACTION),
        "--disable-radix-cache",
        "--trust-remote-code",
        "--dtype", "bfloat16",
    ]
    print(f"Starting SGLang: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    try:
        deadline = time.monotonic() + C.SGLANG_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang exited during startup with code {proc.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    print("SGLang ready")
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        else:
            raise TimeoutError(f"SGLang not ready after {C.SGLANG_STARTUP_TIMEOUT_S}s")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def make_verbalizer(client):
    """Bind greedy-decoding kwargs to client.generate.

    Greedy (temperature=0) is required: without it the fidelity metrics are not
    reproducible run to run. Only kwargs the client actually accepts are passed,
    matching the fallback behaviour in setup.ipynb.
    """
    params = inspect.signature(client.generate).parameters
    greedy_kw = {k: v for k, v in [("temperature", 0.0), ("top_p", 1.0)] if k in params}
    print(f"Greedy kwargs supported by client.generate: {greedy_kw or 'NONE (check server defaults)'}")

    def verbalize(vec):
        return client.generate(vec, **greedy_kw)

    return verbalize


def assert_deterministic(verbalize, vec) -> str:
    """Verify greedy decoding is actually active before spending a full run."""
    first = verbalize(vec)
    if verbalize(vec) != first:
        raise AssertionError("non-deterministic output, greedy decoding is NOT active")
    print("Deterministic (greedy) decoding confirmed")
    return first


def check_injection_health(explanations: list[str], sample: int = 20) -> None:
    """Warn on the known injection-failure signature.

    nla/injection.py documents it: if injection misses the marked position the
    model sees the literal marker character and emits CJK output. Cheap to check
    and it fails loudly rather than producing plausible garbage metrics.
    """
    suspect = [e for e in explanations[:sample] if "㊗" in e or _mostly_cjk(e)]
    if suspect:
        print(
            f"WARNING: {len(suspect)}/{min(sample, len(explanations))} sampled explanations "
            f"look like injection failures (marker char or CJK output).\n"
            f"  First: {suspect[0][:200]!r}\n"
            f"  Check injection_scale and that the AV checkpoint matches layer {C.LAYER_IDX}."
        )


def _mostly_cjk(text: str, threshold: float = 0.3) -> bool:
    if not text:
        return False
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / len(text) > threshold


def run_av(verbalize, acts, name: str, concurrency: int = C.AV_CONCURRENCY) -> list[str]:
    """Verbalize every activation. Returns explanations in input order.

    ThreadPoolExecutor.map preserves ordering, which matters because these line
    up positionally with labels_L20.npy.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        explanations = list(
            tqdm(pool.map(verbalize, acts), total=len(acts), desc=f"AV: {name}")
        )
    check_injection_health(explanations)
    return explanations
