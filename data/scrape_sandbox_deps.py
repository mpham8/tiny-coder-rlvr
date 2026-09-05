#!/usr/bin/env python3
"""Scrape import libs from train/eval datasets for the sandbox image.

Walks the same datasets used by training / decontamination:

  - newfacade/LeetCodeDataset          (train, test)
  - livecodebench/code_generation_lite (test)
  - open-r1/codeforces                 (train, test; python only)

Prints a report of stdlib vs third-party top-level modules, and optionally
writes third-party packages to ``tiny_coder_rlvr/sandbox/requirements.txt``.

Usage:
  python data/scrape_sandbox_deps.py
  python data/scrape_sandbox_deps.py --write
  python data/scrape_sandbox_deps.py --write --output path/to/requirements.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.prepare_data import (  # noqa: E402
    CF_CONFIG,
    CF_DATASET,
    CF_LANGUAGE,
    DATASET_NAME,
    LCB_DATASET,
    LCB_RELEASE,
)

DEFAULT_OUTPUT = _REPO_ROOT / "tiny_coder_rlvr" / "sandbox" / "requirements.txt"

# Top-level import name -> pip distribution name when they differ.
MODULE_TO_PIP: dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "serial": "pyserial",
    "Crypto": "pycryptodome",
    "Cryptodome": "pycryptodome",
}

# Always keep these out of pip requirements even if scraped.
ALWAYS_STDLIB = {
    "__future__",
    "typing_extensions",  # often bundled / not needed for sandbox grading
}

IMPORT_FROM_RE = re.compile(r"(?:^|\n)\s*from\s+([a-zA-Z_][\w.]*)\s+import\b")
IMPORT_RE = re.compile(r"(?:^|\n)\s*import\s+([^\n#]+)")


def extract_top_level_modules(text: str) -> set[str]:
    """Return top-level module names referenced by import / from-import lines."""
    if not text:
        return set()

    modules: set[str] = set()
    for match in IMPORT_FROM_RE.finditer(text):
        name = match.group(1)
        if name.startswith("."):
            continue
        modules.add(name.split(".", 1)[0])

    for match in IMPORT_RE.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip().split(" as ", 1)[0].strip()
            if not part or part.startswith("."):
                continue
            modules.add(part.split(".", 1)[0])
    return modules


def stdlib_modules() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", ()))
    # Fallback / extras that appear in prompts but are stdlib on 3.12.
    names.update(
        {
            "builtins",
            "typing",
            "collections",
            "functools",
            "itertools",
            "heapq",
            "bisect",
            "operator",
            "math",
            "random",
            "string",
            "datetime",
            "dataclasses",
            "enum",
            "queue",
            "re",
            "sys",
            "threading",
            "json",
            "copy",
            "array",
            "fractions",
            "decimal",
            "statistics",
            "cmath",
            "io",
            "os",
            "pathlib",
            "time",
            "unittest",
            "pprint",
            "hashlib",
            "hmac",
            "base64",
            "struct",
            "typing_extensions",
        }
    )
    return names


def iter_text_fields(row: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in row.items():
        if isinstance(value, str) and value:
            out.append((key, value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    out.append((key, item))
                elif isinstance(item, dict):
                    for nested in item.values():
                        if isinstance(nested, str) and nested:
                            out.append((key, nested))
    return out


def scrape_dataset(
    name: str,
    dataset: Dataset,
    *,
    source_counts: dict[str, Counter[str]],
    module_counts: Counter[str],
    row_filter=None,
) -> int:
    n = 0
    for row in dataset:
        if row_filter is not None and not row_filter(row):
            continue
        n += 1
        for _field, text in iter_text_fields(row):
            for module in extract_top_level_modules(text):
                module_counts[module] += 1
                source_counts[module][name] += 1
    return n


def load_and_scrape() -> tuple[Counter[str], dict[str, Counter[str]], dict[str, int]]:
    module_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_counts: dict[str, int] = {}

    for split in ("train", "test"):
        key = f"leetcode:{split}"
        ds = load_dataset(DATASET_NAME, split=split)
        row_counts[key] = scrape_dataset(key, ds, source_counts=source_counts, module_counts=module_counts)

    key = f"lcb:{LCB_RELEASE}:test"
    lcb = load_dataset(LCB_DATASET, LCB_RELEASE, split="test", trust_remote_code=True)
    row_counts[key] = scrape_dataset(key, lcb, source_counts=source_counts, module_counts=module_counts)

    for split in ("train", "test"):
        key = f"codeforces:{split}"
        cf = load_dataset(CF_DATASET, CF_CONFIG, split=split, trust_remote_code=True)
        row_counts[key] = scrape_dataset(
            key,
            cf,
            source_counts=source_counts,
            module_counts=module_counts,
            row_filter=lambda row: row.get("language") == CF_LANGUAGE,
        )

    return module_counts, source_counts, row_counts


def classify_modules(
    module_counts: Counter[str],
) -> tuple[list[str], list[str]]:
    stdlib = stdlib_modules() | ALWAYS_STDLIB
    third_party: list[str] = []
    stdlib_found: list[str] = []
    for module in sorted(module_counts):
        if module in stdlib:
            stdlib_found.append(module)
        else:
            third_party.append(module)
    return third_party, stdlib_found


def modules_to_requirements(modules: list[str]) -> list[str]:
    packages = sorted({MODULE_TO_PIP.get(module, module) for module in modules})
    return packages


def format_requirements(packages: list[str], *, sources: dict[str, Counter[str]], modules: list[str]) -> str:
    lines = [
        "# Auto-generated by data/scrape_sandbox_deps.py",
        "# Third-party imports found in LeetCode / LiveCodeBench / Codeforces train+eval.",
        "# Stdlib modules are omitted (provided by the sandbox Python image).",
        "",
    ]
    if not packages:
        lines.append("# (no third-party packages detected)")
        lines.append("")
        return "\n".join(lines)

    module_by_pkg = defaultdict(list)
    for module in modules:
        module_by_pkg[MODULE_TO_PIP.get(module, module)].append(module)

    for package in packages:
        mods = ", ".join(sorted(module_by_pkg[package]))
        srcs = sorted({src for mod in module_by_pkg[package] for src in sources[mod]})
        lines.append(f"# import: {mods}  |  seen in: {', '.join(srcs)}")
        lines.append(package)
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write third-party packages to {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Requirements path used with --write",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Loading datasets and scraping imports...")
    module_counts, source_counts, row_counts = load_and_scrape()

    print("\nRows scanned:")
    for key, n in sorted(row_counts.items()):
        print(f"  {key}: {n}")

    third_party, stdlib_found = classify_modules(module_counts)
    packages = modules_to_requirements(third_party)

    print(f"\nStdlib modules ({len(stdlib_found)}):")
    for module in stdlib_found:
        srcs = ", ".join(sorted(source_counts[module]))
        print(f"  {module:20s}  count={module_counts[module]:5d}  [{srcs}]")

    print(f"\nThird-party modules ({len(third_party)}):")
    if not third_party:
        print("  (none)")
    else:
        for module in third_party:
            pkg = MODULE_TO_PIP.get(module, module)
            srcs = ", ".join(sorted(source_counts[module]))
            print(f"  {module:20s}  pip={pkg:24s}  count={module_counts[module]:5d}  [{srcs}]")

    text = format_requirements(packages, sources=source_counts, modules=third_party)
    print("\n--- requirements.txt preview ---")
    print(text)

    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
