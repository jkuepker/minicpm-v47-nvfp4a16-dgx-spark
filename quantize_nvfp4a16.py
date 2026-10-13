#!/usr/bin/env python3
"""
Data-free NVFP4A16 (W4A16, group 16, fp8-e4m3 block scales + fp32 global scale)
quantization of openbmb/MiniCPM-V-4.7-35B-A3B into a compressed-tensors checkpoint.

Approach: llm-compressor `model_free_ptq` (llmcompressor.entrypoints.model_free).
It streams the BF16 safetensors shards one at a time, WITHOUT instantiating the
transformers model, so:
  * peak RAM/VRAM ~ one shard (a few GB), no device_map/offload needed;
  * the fused 3D expert tensors (`mlp.experts.gate_up_proj` [E, 2I, H] and
    `mlp.experts.down_proj` [E, H, I]) are split by llm-compressor's
    `split_fused_moe_experts` into per-expert 2D linears
    (`mlp.experts.{e}.{gate,up,down}_proj`), quantized, and saved UNFUSED
    (same on-disk layout as nvidia/Qwen3.6-35B-A3B-NVFP4, which vLLM loads).
  * gate_proj/up_proj of every expert (and of the shared expert) share one
    weight_global_scale (llm-compressor fuses the groups), as vLLM requires.

By default only routed experts + shared experts are quantized. Everything else stays BF16
(self_attn, linear_attn, routers, shared_expert_gate, embeddings, lm_head,
vision tower, vit_merger, merger). See --quantize-attn / --quantize-lm-head below.

Usage:
  python quantize_nvfp4a16.py [--src openbmb/MiniCPM-V-4.7-35B-A3B | /local/dir]
                              [--dst ~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16]
                              [--device cuda:0|cpu] [--max-workers 1]
                              [--quantize-attn] [--quantize-lm-head]

--quantize-attn additionally quantizes the attention projections (linear_attn in_proj_qkv,
in_proj_z, out_proj; self_attn q/k/v/o_proj) to NVFP4A16. in_proj_a/in_proj_b (output dim 32,
fused into vLLM's in_proj_ba), conv1d, norms, routers, vision, merger and embeddings stay BF16.
q/k/v and in_proj_qkv+in_proj_z get ONE shared weight_global_scale per vLLM-fused layer
(vLLM takes weight_global_scale.max() over the merged partitions, so unequal scales silently
mis-scale the other partitions). --quantize-lm-head also quantizes lm_head.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_SRC = "openbmb/MiniCPM-V-4.7-35B-A3B"
DEFAULT_DST = os.path.expanduser("~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16")

# Scheme "NVFP4A16" targets "Linear", i.e. every 2D .weight not matched below.
# vLLM's compressed-tensors loader ALSO applies this list (regex entries), so it
# must name every Linear that has to stay BF16. Patterns are deliberately broad
# and name-prefix agnostic (HF "model.language_model.*" vs vLLM-remapped names).
# Reference: OpenBMB's FP8 build ignores vision_tower.*, vit_merger.*,
# merger.mlp.0.*, linear_attn.*, mlp.gate, shared_expert.*, lm_head.
IGNORE = [
    "re:.*lm_head$",
    "re:.*embed_tokens.*",
    "re:.*vision_tower.*",     # ViT encoder + vit_merger
    "re:.*vpm.*",              # vLLM renames model.vision_tower.* to vpm.*; regex ignores are not remapped
    "re:.*vit_merger.*",       # vLLM name for model.vision_tower.vit_merger.*
    "re:.*merger.*",           # model.merger.mlp.* (and vit_merger)
    "re:.*visual.*",
    "re:.*self_attn.*",        # full-attention q/k/v/o (LM) -- kept BF16 on purpose
    "re:.*linear_attn.*",      # GatedDeltaNet in_proj_qkv/z/a/b, out_proj, conv1d
    "re:.*conv1d.*",
    "re:.*mlp\\.gate$",        # MoE router
    "re:.*shared_expert_gate$",
    "re:.*mtp.*",              # not present in this checkpoint; guard anyway
    "re:.*norm.*",
]

# --quantize-attn: the blanket self_attn / linear_attn entries are dropped (that is what lets the
# LM attention projections be quantized) and replaced by the narrow entries below.
#  * The vision tower's own self_attn (q/k/v/out_proj) stays BF16 because the vision_tower / vpm
#    entries (HF and vLLM names) are kept; none of the dropped entries is needed for it.
#  * in_proj_a / in_proj_b: output dim 32 (too small for Marlin) and vLLM fuses them into
#    in_proj_ba, so both must be ignored together (a half-quantized merged layer cannot load).
#  * in_proj_qkv + in_proj_z (vLLM in_proj_qkvz) and q/k/v (vLLM qkv_proj) are quantized together.
ATTN_DROP = ("re:.*self_attn.*", "re:.*linear_attn.*")
ATTN_IGNORE_EXTRA = [
    "re:.*linear_attn\\.in_proj_[ab]$",
]


def build_ignore(quantize_attn=False, quantize_lm_head=False):
    ignore = list(IGNORE)
    if quantize_attn:
        ignore = [i for i in ignore if i not in ATTN_DROP] + ATTN_IGNORE_EXTRA
    if quantize_lm_head:
        ignore = [i for i in ignore if i != "re:.*lm_head$"]
    return ignore


def _log(msg):
    print(f"[quantize] {msg}", flush=True)


# Fusion groups: (parent regex, {set key: member tensor name}). Grouping is by PARENT module so
# that unrelated tensors can never be paired (llm-compressor 0.14.0's own matcher is order-based).
# Every member of a group gets the SAME weight_global_scale. Cross-shard: the first member is the
# "primary" and pulls the others in (see converter.get_dependencies).
_EXPERT_GROUP = (r"^(.*\.(?:experts\.\d+|shared_expert))", {"gate": "gate_proj", "up": "up_proj"})
# language_model-anchored so the vision tower's self_attn.{q,k,v}_proj can never be grouped.
_ATTN_GROUPS = [
    # vLLM qkv_proj = q_proj + k_proj + v_proj (full attention, every 4th layer)
    (r"^(.*language_model\.layers\.\d+\.self_attn)", {"q": "q_proj", "k": "k_proj", "v": "v_proj"}),
    # vLLM in_proj_qkvz = in_proj_qkv + in_proj_z (GatedDeltaNet); in_proj_a/b stay BF16
    (r"^(.*language_model\.layers\.\d+\.linear_attn)", {"qkv": "in_proj_qkv", "z": "in_proj_z"}),
]


def patch_fused_mappings(quantize_attn=False):
    """NVFP4 needs the partitions that vLLM merges into ONE linear layer to share a single
    weight_global_scale. vLLM's CompressedTensorsW4A4Fp4.process_weights_after_loading
    (compressed_tensors_w4a4_nvfp4.py L100-114) collapses weight_global_scale to a scalar with
    `.max()` and only logs a warning if the partitions differ, so unequal scales would silently
    dequantize the other partitions with the wrong global scale.
    Merged here: expert/shared-expert gate_proj+up_proj (w13) and, with quantize_attn,
    q/k/v (qkv_proj) and in_proj_qkv+in_proj_z (in_proj_qkvz).

    llm-compressor 0.14.0's fusion tables only know dense `mlp.gate_proj/up_proj`, q/k/v, ...;
    they do NOT match `mlp.experts.N.{gate,up}_proj` (produced by split_fused_moe_experts),
    `mlp.shared_expert.{gate,up}_proj` or GatedDeltaNet in_proj_qkv/in_proj_z. Its name matcher is
    also order-based (sorted names, first complete set wins), which can mis-pair or raise when a
    shard boundary cuts a group (e.g. an LM self_attn q/k/v set vs the vision tower's). So the
    whole table is replaced with explicit parent-based groups; the first member of each group is
    the cross-shard primary and pulls in the rest.
    Newer llm-compressor (post 0.14.0, observers/fused_mappings.py) does the expert part natively."""
    import re

    import llmcompressor.entrypoints.model_free.converter as cv
    import llmcompressor.entrypoints.model_free.microscale as ms
    import llmcompressor.entrypoints.model_free.validate as va

    if not hasattr(ms, "DEFAULT_FUSED_MAPPINGS"):
        _log("fused-scale groups: native in this llm-compressor, no patch")
        return

    groups = [_EXPERT_GROUP] + (_ATTN_GROUPS if quantize_attn else [])
    compiled = []
    for parent_re, members in groups:
        alt = "|".join(re.escape(v) for v in members.values())
        rev = {v: k for k, v in members.items()}
        compiled.append((re.compile(parent_re + r"\.(" + alt + r")\.weight$"), rev, len(members)))

    def get_fused_names(tensor_names):
        by_parent = {}
        for n in tensor_names:
            for gi, (pat, rev, _) in enumerate(compiled):
                m = pat.match(n)
                if m:
                    by_parent.setdefault((gi, m.group(1)), {})[rev[m.group(2)]] = n
                    break
        sets = [d for (gi, _), d in by_parent.items() if len(d) == compiled[gi][2]]
        return sets, []  # (matched sets, unmatched sets)

    ms.DEFAULT_FUSED_MAPPINGS.clear()  # same dict object is imported by converter.py
    # cross-shard dependencies: primary pattern -> partner templates
    ms.DEFAULT_FUSED_MAPPINGS[r"^(?P<prefix>.+?)\.shared_expert\.gate_proj\.weight$"] = [
        r"{prefix}.shared_expert.up_proj.weight"
    ]
    if quantize_attn:
        ms.DEFAULT_FUSED_MAPPINGS[r"^(?P<prefix>.*language_model\.layers\.\d+\.self_attn)\.q_proj\.weight$"] = [
            r"{prefix}.k_proj.weight",
            r"{prefix}.v_proj.weight",
        ]
        ms.DEFAULT_FUSED_MAPPINGS[r"^(?P<prefix>.*language_model\.layers\.\d+\.linear_attn)\.in_proj_qkv\.weight$"] = [
            r"{prefix}.in_proj_z.weight",
        ]
    for mod in (ms, cv, va):
        mod.get_fused_names = get_fused_names
    _log("fused-scale groups: patched llm-compressor 0.14.0 (experts.N + shared_expert gate/up"
         + (", self_attn q/k/v, linear_attn in_proj_qkv+in_proj_z" if quantize_attn else "")
         + " share a global scale)")


def resolve_src(src: str) -> Path:
    """Local dir -> itself; hub id -> snapshot_download (resumable, uses HF cache)."""
    if os.path.isdir(src):
        return Path(src)
    from huggingface_hub import snapshot_download

    _log(f"snapshot_download({src}) (HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})")
    return Path(snapshot_download(src))


def preflight(src_dir: Path):
    """Fail fast if the checkpoint does not look like what the recipe assumes."""
    from safetensors import safe_open

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit(f"missing {index_path}")
    wmap = json.loads(index_path.read_text())["weight_map"]
    if any(k.startswith("mtp.") or ".mtp." in k for k in wmap):
        sys.exit(
            "checkpoint contains mtp.* tensors; model_free_ptq would split their fused "
            "experts. Handle mtp separately before running this script."
        )
    n_fused = 0
    odd = []
    for shard in sorted(set(wmap.values())):
        with safe_open(src_dir / shard, "pt") as f:
            for k in f.keys():
                shp = f.get_slice(k).get_shape()
                if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
                    if len(shp) != 3:
                        odd.append((k, shp))
                    n_fused += 1
                elif len(shp) == 3 and "conv" not in k and "patch_embed" not in k:
                    odd.append((k, shp))
    _log(f"preflight: {n_fused} fused 3D expert tensors found")
    if n_fused == 0:
        sys.exit("no fused 3D experts found (already per-expert?) - recipe assumption violated")
    if odd:
        sys.exit(f"unexpected 3D/odd tensors that split_fused_moe_experts could mangle: {odd[:5]}")


AUX_SUFFIXES = {".json", ".jinja", ".txt", ".model", ".tiktoken", ".py"}


def copy_aux_files(src_dir: Path, dst: Path):
    """model_free_ptq already copies non-safetensors files; make sure every
    tokenizer / processor / chat-template / generation file is present. Skips the
    README and the model-card media under minicpm_v4dot7_assets/."""
    copied = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file() or p.suffix not in AUX_SUFFIXES:
            continue
        if p.name == "model.safetensors.index.json":
            continue  # regenerated by model_free_ptq
        out = dst / p.name
        if not out.exists():
            shutil.copyfile(p.resolve(), out)
            copied.append(p.name)
    if copied:
        _log(f"copied extra files: {copied}")
    for need in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "preprocessor_config.json", "processor_config.json",
                 "chat_template.jinja", "generation_config.json"):
        if (src_dir / need).exists() and not (dst / need).exists():
            sys.exit(f"missing {need} in output")


LM_HEAD_TARGET = "re:.*lm_head$"


def add_lm_head_target(config_path: Path):
    """llm-compressor writes targets=["Linear"] even when lm_head is quantized. vLLM's compressed-tensors
    loader matches the LM head (ParallelLMHead, not a Linear subclass) only by class name or regex, and a
    plain "lm_head" is not remapped to vLLM's language_model.lm_head prefix, so add a regex target."""
    cfg = json.loads(config_path.read_text())
    groups = cfg.get("quantization_config", {}).get("config_groups", {})
    if len(groups) != 1:
        sys.exit(f"expected exactly 1 config group in {config_path}, got {len(groups)}")
    group = next(iter(groups.values()))
    targets = group.setdefault("targets", [])
    if LM_HEAD_TARGET not in targets:
        targets.append(LM_HEAD_TARGET)
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    _log(f"--quantize-lm-head: config_groups targets now {targets} (vLLM needs the regex to load the quantized head)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: cuda:0 if available else cpu)")
    ap.add_argument("--max-workers", type=int, default=1, help="concurrent shard jobs")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--quantize-attn", action="store_true",
                    help="also quantize linear_attn in_proj_qkv/in_proj_z/out_proj and self_attn q/k/v/o_proj "
                         "(in_proj_a/in_proj_b, conv1d, norms stay BF16)")
    ap.add_argument("--quantize-lm-head", action="store_true", help="also quantize lm_head (default: BF16)")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dst = Path(args.dst)
    if dst.exists() and any(dst.glob("*.safetensors")):
        sys.exit(f"{dst} already contains safetensors; refusing to overwrite (rm it first)")

    src_dir = resolve_src(args.src)
    _log(f"src={src_dir} dst={dst} device={device}")
    if not args.skip_preflight:
        preflight(src_dir)

    patch_fused_mappings(args.quantize_attn)
    ignore = build_ignore(args.quantize_attn, args.quantize_lm_head)
    _log(f"quantize_attn={args.quantize_attn} quantize_lm_head={args.quantize_lm_head} n_ignore={len(ignore)}")
    from llmcompressor import model_free_ptq

    t0 = time.time()
    model_free_ptq(
        model_stub=str(src_dir),
        save_directory=str(dst),
        scheme="NVFP4A16",   # weights: 4-bit float, TENSOR_GROUP, group_size=16, fp8 scales; no activations
        ignore=ignore,
        max_workers=args.max_workers,
        device=device,
    )
    _log(f"model_free_ptq done in {(time.time() - t0) / 60:.1f} min")

    copy_aux_files(src_dir, dst)

    if args.quantize_lm_head:
        add_lm_head_target(dst / "config.json")

    qc = json.loads((dst / "config.json").read_text()).get("quantization_config", {})
    _log(f"quantization_config: format={qc.get('format')} quant_method={qc.get('quant_method')} "
         f"n_ignore={len(qc.get('ignore', []))}")
    _log("next: python verify_ckpt.py " + str(dst) + (" --attn" if args.quantize_attn else "")
         + (" --lm-head" if args.quantize_lm_head else ""))


if __name__ == "__main__":
    main()
