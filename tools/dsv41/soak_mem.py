#!/usr/bin/env python3
"""Memory soak for a running DeepSeek V4.1-Flash server: watch `MemAvailable` while cold long prompts go through it.

Point it at an OpenAI-compatible server that is already serving (strata serve, or `strata launch`):

    python tools/dsv41/soak_mem.py --url http://127.0.0.1:8000
    python tools/dsv41/soak_mem.py --url http://127.0.0.1:8000 --pid ENGINE_PID --reserve-gib 10

For every prompt size it sends one cold prompt (a unique nonce, so no server-side cache answers it), samples `/proc/meminfo`
every `--interval` seconds in a thread, and optionally reads `/proc/<pid>/status` for `RssShmem`.  It prints one JSON line per
prompt with the lowest `MemAvailable`, the peak `RssShmem`, the approximate prompt tokens and the answer speed, then exits 0
when every read kept `MemAvailable` at or above the reserve, 1 when the floor was crossed (the worst read is named), and 2 when
the server could not be reached or a request failed.  Standard library only; Linux (`/proc`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

GIB = 1 << 30
DEFAULT_RESERVE_GIB = 8.0          # the engine's desktop reserve floor (strata::core::uma_desktop_reserve_bytes)
UMA_RESERVE_SHARE = 0.08           # ... or 8% of the installed RAM when that is larger
CHARS_PER_TOKEN = 4                # an approximate token: four characters of English text
FILLER = "the routed expert reads the same line from the model file while the prompt keeps going without a pause. "


def reserve_gib_from(value, total_gib: float | None = None) -> float:
    """The reserve in GiB: `value` (--reserve-gib, a number or a string) when given, else `DEFAULT_RESERVE_GIB` or 8% of the
    installed RAM when that is larger.  Raises ValueError for a negative or unreadable value."""
    if value is not None:
        try:
            reserve = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"not a number of GiB: {value!r}") from None
        if not reserve >= 0.0:
            raise ValueError(f"the reserve cannot be negative: {value!r}")
        return reserve
    if total_gib and total_gib > 0.0:
        return max(DEFAULT_RESERVE_GIB, round(total_gib * UMA_RESERVE_SHARE, 2))
    return DEFAULT_RESERVE_GIB


def summarize(available, rss_shmem=(), reserve_bytes: int = 0) -> dict:
    """The lowest `MemAvailable` and the peak `RssShmem` of one prompt's samples, and whether the floor was kept.  `kept` is None
    when nothing was sampled: an unread /proc is not a pass."""
    lows = [int(v) for v in available]
    if not lows:
        return {"min_available": None, "peak_rss_shmem": None, "kept": None}
    peak = max((int(v) for v in rss_shmem), default=None)
    low = min(lows)
    return {"min_available": low, "peak_rss_shmem": peak, "kept": low >= int(reserve_bytes)}


def prompt_text(tokens: int, nonce: str) -> str:
    """Roughly `tokens` tokens of text (CHARS_PER_TOKEN characters each) with `nonce` inside: a fresh nonce per prompt makes the
    server answer it cold, so no cache shortens the prompt path being measured."""
    tokens = max(1, int(tokens))
    marker = f"nonce {nonce}\n"
    target = max(len(marker), tokens * CHARS_PER_TOKEN)
    body_len = max(len(marker), target - len(marker))
    body = (marker + FILLER * (body_len // len(FILLER) + 1))[:body_len]
    return "Read this and answer with the word done.\n" + body


def parse_prompts(value: str) -> list[int]:
    """`42000,72000` -> [42000, 72000].  Raises ValueError for anything that is not a positive whole number."""
    sizes = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError(f"a prompt size must be positive: {part!r}")
        sizes.append(n)
    if not sizes:
        raise ValueError("no prompt sizes were given")
    return sizes


def parse_meminfo_available(text: str) -> int | None:
    """`MemAvailable` from /proc/meminfo in bytes, None when the line is not there."""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def parse_rss_shmem(text: str) -> int | None:
    """`RssShmem` from /proc/<pid>/status in bytes, None when the line is not there."""
    for line in text.splitlines():
        if line.startswith("RssShmem:"):
            return int(line.split()[1]) * 1024
    return None


def total_ram_gib() -> float | None:
    """The installed RAM in GiB from /proc/meminfo, None when it cannot be read."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / (1024 * 1024)
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


class Sampler(threading.Thread):
    """Samples MemAvailable (and RssShmem of `pid`, when given) every `interval` seconds on its own thread."""

    def __init__(self, interval: float = 1.0, pid: int | None = None):
        super().__init__(daemon=True)
        self.interval = max(0.05, float(interval))
        self.pid = int(pid) if pid is not None else None
        self.available: list[int] = []
        self.rss_shmem: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def sample_once(self) -> None:
        text = _read("/proc/meminfo")
        if text is not None:
            value = parse_meminfo_available(text)
            if value is not None:
                self.available.append(value)
        if self.pid is not None:
            status = _read(f"/proc/{self.pid}/status")
            if status is not None:
                rss = parse_rss_shmem(status)
                if rss is not None:
                    self.rss_shmem.append(rss)

    def stop(self) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=10.0)


def chat_once(url: str, prompt: str, max_tokens: int, api_key: str, timeout: float) -> int:
    """One non-streaming chat completion; the answer's completion tokens from `usage`."""
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": int(max_tokens), "temperature": 0.0, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8", "replace"))
    return int((body.get("usage") or {}).get("completion_tokens") or 0)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MemAvailable while cold long prompts go through a running DeepSeek server.")
    parser.add_argument("--url", required=True, help="the server's base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", help="the environment variable that holds the server's API key (never its value)")
    parser.add_argument("--pid", type=int, help="the engine process, for its peak RssShmem")
    parser.add_argument("--prompts", default="42000,72000,100000,200000", help="comma-separated approximate prompt sizes in tokens")
    parser.add_argument("--max-tokens", type=int, default=8, help="answer tokens per prompt (default 8)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between memory samples (default 1.0)")
    parser.add_argument("--reserve-gib", help="the MemAvailable floor in GiB (default max(8, 8%% of the installed RAM))")
    parser.add_argument("--timeout", type=float, default=3600.0, help="the longest one prompt may take, in seconds (default 3600)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = ""
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            print(f"{args.api_key_env} is not set: the server needs the key it was started with.", file=sys.stderr)
            return 2
    try:
        sizes = parse_prompts(args.prompts)
        reserve_gib = reserve_gib_from(args.reserve_gib, total_ram_gib())
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    reserve_bytes = int(reserve_gib * GIB)
    worst = None
    for tokens in sizes:
        nonce = f"{os.getpid()}-{time.time_ns()}-{tokens}"
        sampler = Sampler(args.interval, args.pid)
        sampler.start()
        started = time.monotonic()
        try:
            completion_tokens = chat_once(args.url, prompt_text(tokens, nonce), args.max_tokens, api_key, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            sampler.stop()
            print(json.dumps({"error": str(exc), "prompt_tokens": tokens, "url": args.url}), file=sys.stderr)
            return 2
        elapsed = max(1e-9, time.monotonic() - started)
        sampler.stop()
        summary = summarize(sampler.available, sampler.rss_shmem, reserve_bytes)
        if summary["kept"] is None:
            print(json.dumps({"error": "could not read /proc/meminfo", "prompt_tokens": tokens}), file=sys.stderr)
            return 2
        line = {"prompt_tokens": tokens,
                "min_available_gib": round(summary["min_available"] / GIB, 3),
                "peak_rss_shmem_gib": None if summary["peak_rss_shmem"] is None else round(summary["peak_rss_shmem"] / GIB, 3),
                "tokens_per_s": round(completion_tokens / elapsed, 2),
                "kept": summary["kept"]}
        print(json.dumps(line), flush=True)
        if not summary["kept"] and (worst is None or summary["min_available"] < worst["min_available_bytes"]):
            worst = {"prompt_tokens": tokens, "min_available_gib": line["min_available_gib"],
                     "min_available_bytes": summary["min_available"]}
    if worst is None:
        return 0
    print(json.dumps({"floor_crossed": True, "reserve_gib": round(reserve_gib, 3), "worst_prompt_tokens": worst["prompt_tokens"],
                      "worst_available_gib": worst["min_available_gib"]}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
