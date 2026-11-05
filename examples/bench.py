"""Benchmark brain/clicker model pairs on known-answer research tasks.

    uv run --env-file .env python examples/bench.py --combos A,B --runs 3 --out results.jsonl
    uv run python examples/bench.py --summary results.jsonl

Each run is a fresh `examples/research.py` process with the combo's settings layered over
the current environment. An answer passes when every expected pattern appears in it.
Ground truth was checked against PyPI, the GitHub API and Wikipedia on 2026-11-05.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

OLLAMA = os.environ.get("BENCH_OLLAMA_URL", "http://192.168.8.153:11434")
LITELLM = os.environ.get("BENCH_LITELLM_URL", "https://llm.lilbro.cloud")
LOCAL_OFF = '{"reasoning_effort":"none"}'


def local(clef, brain):
    return {
        "TYPESAFE_API_BASE": OLLAMA,
        "TYPESAFE_MODEL": clef,
        "TEXT_MODEL_BASE_URL": OLLAMA + "/v1",
        "TEXT_MODEL": brain,
        "AGENT_MODEL": brain,
        "TEXT_MODEL_EXTRA_BODY": LOCAL_OFF,
        "AGENT_EXTRA_BODY": LOCAL_OFF,
    }


def gateway(clef, brain):
    return {
        "TYPESAFE_API_BASE": OLLAMA,
        "TYPESAFE_MODEL": clef,
        "TEXT_MODEL_BASE_URL": LITELLM + "/v1",
        "TEXT_MODEL": brain,
        "AGENT_MODEL": brain,
        "TEXT_MODEL_EXTRA_BODY": "{}",
        "AGENT_EXTRA_BODY": "{}",
    }


COMBOS = {
    "A": ("clef:27b + qwen3.5:4b", local("clef:27b", "qwen3.5:4b")),
    "B": ("clef-flash:9b + qwen3.5:4b", local("clef-flash:9b", "qwen3.5:4b")),
    "C": ("clef-flash:9b + qwen3.5:9b", local("clef-flash:9b", "qwen3.5:9b")),
    "D": ("clef:27b + qwen3.5:9b", local("clef:27b", "qwen3.5:9b")),
    "E": ("clef-flash:9b + qwen3.8:27b", local("clef-flash:9b", "qwen3.8:latest")),
    "F": ("clef-flash:9b + claude-haiku-5-5", gateway("clef-flash:9b", "anthropic/claude-haiku-5-5")),
}

# (id, needs_navigate, question, [patterns that must all appear in the answer])
TASKS = [
    ("pypi-version", False, "What is the latest version of the Python package browser-harness on PyPI?",
     [r"0\.1\.13"]),
    ("ollama-stable", False,
     "What is the latest stable (not pre-release) version of Ollama on GitHub, and on what date was it released?",
     # Published 2026-11-03 01:40 UTC, which GitHub shows as Nov 2 in US time zones: accept both.
     [r"0\.40\.14", r"(nov(ember)?\.? [23]\b|2026-11-0[23]|\b[23] nov)"]),
    ("wiki-search-box", True,
     "Use the search box on https://en.wikipedia.org to open the article on Gödel's incompleteness theorems, "
     "and tell me the year Gödel published them.",
     [r"1931"]),
    ("repo-license", False, "What license is the GitHub repository browser-use/jev-ultrafast released under?",
     [r"\bmit\b"]),
    ("clef-multihop", False,
     "Which company released the Clef decision models, and which base model is Clef-flash built from?",
     [r"cloudflare", r"qwen\s?3\.5"]),
    ("pypi-search-box", True,
     "Use the search box on https://pypi.org (not a web search engine) to find the package httpx, open it, "
     "and tell me who its author is.",
     [r"tom christie"]),
    ("pyproject-python", False,
     "What minimum Python version does the browser-use/jev-ultrafast project require? Check its pyproject.toml.",
     [r"3\.12"]),
    ("raspberry-pi-year", False, "In what year was the first Raspberry Pi model released?", [r"2012"]),
]

TIMEOUT_S = int(os.environ.get("BENCH_TIMEOUT_S", "300"))
RESEARCH = Path(__file__).with_name("research.py")


def run_once(combo, task):
    _, overrides = COMBOS[combo]
    task_id, needs_nav, question, patterns = task
    env = {**os.environ, **overrides}
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(RESEARCH), question], env=env, capture_output=True, text=True, timeout=TIMEOUT_S
        )
        log, timed_out = proc.stdout + proc.stderr, False
    except subprocess.TimeoutExpired as error:
        log, timed_out = error.stdout or "", True
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
    elapsed = time.perf_counter() - started
    marker = re.search(r"^\[\d+s, (\d+) turns\]$", log, re.M)
    answer = log[marker.end():].strip() if marker else ""
    passed = bool(answer) and all(re.search(p, answer, re.I) for p in patterns)
    return {
        "combo": combo,
        "task": task_id,
        "pass": passed,
        "seconds": round(elapsed, 1),
        "turns": int(marker.group(1)) if marker else None,
        "tool_errors": log.count("→ error"),
        "used_navigate": "navigate(" in log,
        "needs_navigate": needs_nav,
        "timed_out": timed_out,
        "crashed": not marker and not timed_out,
        "answer": answer[:600],
    }


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["combo"]].append(r)
    print("| Combo | Correct | Median time | Tool errors | Used navigate when asked | Timeouts/crashes |")
    print("|---|---|---|---|---|---|")
    for combo in sorted(by):
        rs = by[combo]
        secs = sorted(r["seconds"] for r in rs)
        nav = [r for r in rs if r["needs_navigate"]]
        print(
            f"| {COMBOS[combo][0]} | {sum(r['pass'] for r in rs)}/{len(rs)} | {secs[len(secs) // 2]:.0f}s"
            f" | {sum(r['tool_errors'] for r in rs)} | {sum(r['used_navigate'] for r in nav)}/{len(nav)}"
            f" | {sum(r['timed_out'] or r['crashed'] for r in rs)} |"
        )
    print("\nPer task (passes / runs):\n")
    tasks = [t[0] for t in TASKS]
    print("| Task | " + " | ".join(sorted(by)) + " |")
    print("|---|" + "---|" * len(by))
    for t in tasks:
        cells = []
        for combo in sorted(by):
            rs = [r for r in by[combo] if r["task"] == t]
            cells.append(f"{sum(r['pass'] for r in rs)}/{len(rs)}" if rs else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combos", default="A,B,C")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="bench-results.jsonl")
    parser.add_argument("--summary", metavar="RESULTS")
    args = parser.parse_args()
    if args.summary:
        summarize(args.summary)
        sys.exit()
    with open(args.out, "a") as out:
        for combo in args.combos.split(","):
            for run in range(1, args.runs + 1):
                for task in TASKS:
                    result = {**run_once(combo, task), "run": run}
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    print(
                        f"{combo} run{run} {result['task']:<18} {'PASS' if result['pass'] else 'fail'}"
                        f"  {result['seconds']:>5.0f}s  errors={result['tool_errors']}",
                        flush=True,
                    )
    summarize(args.out)
