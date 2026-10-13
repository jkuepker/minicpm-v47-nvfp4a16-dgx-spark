#!/usr/bin/env python3
"""
Make a MiniCPM-V 4.7 checkpoint loadable by vLLM + transformers without custom code.

Applies two fixes to processor_config.json, preprocessor_config.json and
tokenizer_config.json in MODEL_DIR:

  (a) remove every `auto_map` key (recursively). They point at processing_minicpmv4_7.py,
      image_processing_minicpmv4_7.py, ... which are NOT shipped in the HF repos, so vLLM
      stops with "contains custom code which must be executed" and --trust-remote-code
      cannot help (the .py files do not exist).
  (b) rename MiniCPMV4_7ImageProcessor -> MiniCPMV4_6ImageProcessor and
      MiniCPMV4_7VideoProcessor -> MiniCPMV4_6VideoProcessor. transformers registers the
      minicpmv4_7 model type with the 4.6 processor classes; the 4_7 names give
      "Unrecognized image processor".

Idempotent. Works when the files are hardlinks into the HF cache or read-only: the file is
unlinked and rewritten, so the cached original is never modified.

  python fix_processor_configs.py MODEL_DIR [MODEL_DIR ...]
"""
import json
import os
import sys
from pathlib import Path

FILES = ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
RENAMES = {
    "MiniCPMV4_7ImageProcessor": "MiniCPMV4_6ImageProcessor",
    "MiniCPMV4_7VideoProcessor": "MiniCPMV4_6VideoProcessor",
}


def fix(node, stats):
    if isinstance(node, dict):
        if "auto_map" in node:
            del node["auto_map"]
            stats["auto_map"] += 1
        for k in list(node):
            node[k] = fix(node[k], stats)
        return node
    if isinstance(node, list):
        return [fix(v, stats) for v in node]
    if isinstance(node, str):
        for old, new in RENAMES.items():
            if old in node:
                node = node.replace(old, new)
                stats["renamed"] += 1
        return node
    return node


def fix_file(path: Path) -> bool:
    data = json.loads(path.read_text())
    stats = {"auto_map": 0, "renamed": 0}
    new = fix(data, stats)
    if stats["auto_map"] == 0 and stats["renamed"] == 0:
        print(f"  {path.name}: already clean")
        return False
    text = json.dumps(new, indent=2, ensure_ascii=False) + "\n"
    # unlink first: the file may be a hardlink into ~/.cache/huggingface or read-only
    path.unlink()
    path.write_text(text)
    os.chmod(path, 0o644)
    print(f"  {path.name}: removed {stats['auto_map']} auto_map, renamed {stats['renamed']} class name(s)")
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rc = 0
    for arg in sys.argv[1:]:
        d = Path(arg).expanduser()
        print(f"{d}")
        for name in FILES:
            p = d / name
            if not p.exists():
                print(f"  {name}: MISSING")
                rc = 1
                continue
            fix_file(p)
    sys.exit(rc)


if __name__ == "__main__":
    main()
