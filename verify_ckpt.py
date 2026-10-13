#!/usr/bin/env python3
"""
Post-check for the NVFP4A16 MiniCPM-V-4.7 checkpoint. Reads only config.json,
model.safetensors.index.json and the safetensors headers (no tensor data).

  python verify_ckpt.py [CKPT_DIR] [--size-gb-min 22.5 --size-gb-max 25.5]
         [--num-experts 256 --num-layers 40]   # for toy checkpoints
         [--attn] [--lm-head]    # checkpoint made with quantize_nvfp4a16.py --quantize-attn / --quantize-lm-head

--attn: also require linear_attn in_proj_qkv/in_proj_z/out_proj and self_attn q/k/v/o_proj to be NVFP4
        (weight_packed/weight_scale/weight_global_scale), in_proj_a/in_proj_b + vision + merger BF16,
        q/k/v and in_proj_qkv+in_proj_z to share one weight_global_scale (vLLM takes .max() over the
        merged partitions), and the ignore regexes to behave under both HF and vLLM module names.
--lm-head: lm_head must be NVFP4 too (default: BF16), and a config-group target must be a regex matching
        lm_head and language_model.lm_head (e.g. "re:.*lm_head$"), else vLLM cannot load it.
"""
import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT = str(Path.home() / "models/MiniCPM-V-4.7-35B-A3B-NVFP4A16")
errors, warns = [], []


def err(m):
    errors.append(m)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _ignored(name, ignore):
    for i in ignore:
        if i.startswith("re:"):
            if re.match(i[3:], name):
                return True
        elif i == name:
            return True
    return False


def check_ignore_regexes(ignore, lm_head):
    """The ignore list is applied by vLLM to ITS module names (vLLM renames model.language_model.* ->
    language_model.model.*, model.vision_tower.* -> vpm.*, vision_tower.vit_merger.* -> vit_merger.*;
    regex entries are not remapped), so it must give the right answer for both naming schemes."""
    lm = ("model.language_model.layers.{l}", "language_model.model.layers.{l}")
    must_ignore = [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj", "model.vision_tower.encoder.layers.0.self_attn.out_proj",
        "model.vision_tower.encoder.layers.0.mlp.fc1", "model.vision_tower.vit_merger.linear_1",
        "vpm.encoder.layers.0.self_attn.q_proj", "vpm.encoder.layers.0.self_attn.k_proj",
        "vpm.encoder.layers.0.self_attn.v_proj", "vpm.encoder.layers.0.self_attn.out_proj",
        "vpm.encoder.layers.0.mlp.fc1", "vit_merger.linear_1", "model.merger.mlp.0.linear_1",
        "model.language_model.embed_tokens", "language_model.model.embed_tokens",
    ]
    must_not = []
    for t in lm:
        t = t.format(l=3)
        must_ignore += [f"{t}.linear_attn.in_proj_a", f"{t}.linear_attn.in_proj_b", f"{t}.linear_attn.conv1d",
                        f"{t}.mlp.gate", f"{t}.mlp.shared_expert_gate"]
        must_not += [f"{t}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        must_not += [f"{t}.linear_attn.{n}" for n in ("in_proj_qkv", "in_proj_z", "out_proj")]
        must_not += [f"{t}.mlp.shared_expert.gate_proj", f"{t}.mlp.experts.0.down_proj"]
    (must_not if lm_head else must_ignore).append("lm_head")
    for n in must_ignore:
        if not _ignored(n, ignore):
            err(f"ignore list does NOT cover {n} (must stay BF16)")
    for n in must_not:
        if _ignored(n, ignore):
            err(f"ignore list wrongly covers {n} (must be quantized)")
    print(f"ignore regexes checked against {len(must_ignore)} must-ignore + {len(must_not)} must-quantize names (HF + vLLM naming)")


def check_attn(meta, wmap, d, n_lay, lm_head):
    """Attention projections (and optionally lm_head) are NVFP4; partitions vLLM merges share one weight_global_scale."""
    lp = "model.language_model.layers.{l}."
    n_q = {"self_attn": 0, "linear_attn": 0}
    gs_needed = []   # (group label, [global-scale tensor names that must be equal])
    mods = []
    for l in range(n_lay):
        p = lp.format(l=l)
        full = f"{p}self_attn.q_proj.weight" in meta or f"{p}self_attn.q_proj.weight_packed" in meta
        lin = f"{p}linear_attn.in_proj_qkv.weight" in meta or f"{p}linear_attn.in_proj_qkv.weight_packed" in meta
        if full == lin:
            err(f"layer {l}: expected exactly one of self_attn / linear_attn (full={full}, linear={lin})")
            continue
        if full:
            mods += [f"{p}self_attn.{n}_proj" for n in "qkvo"]
            gs_needed.append((f"{p}self_attn qkv", [f"{p}self_attn.{n}_proj.weight_global_scale" for n in "qkv"]))
            n_q["self_attn"] += 1
            for n in ("q", "k"):
                if f"{p}self_attn.{n}_norm.weight" not in meta:
                    err(f"missing {p}self_attn.{n}_norm.weight")
        else:
            mods += [f"{p}linear_attn.{n}" for n in ("in_proj_qkv", "in_proj_z", "out_proj")]
            gs_needed.append((f"{p}linear_attn qkvz", [f"{p}linear_attn.in_proj_qkv.weight_global_scale",
                                                       f"{p}linear_attn.in_proj_z.weight_global_scale"]))
            n_q["linear_attn"] += 1
            for n in ("in_proj_a", "in_proj_b"):
                e = meta.get(f"{p}linear_attn.{n}.weight")
                if e is None:
                    err(f"missing {p}linear_attn.{n}.weight")
                elif e["dtype"] != "BF16":
                    err(f"{p}linear_attn.{n}.weight dtype {e['dtype']} (want BF16)")
                for t in ("weight_packed", "weight_scale", "weight_global_scale"):
                    if f"{p}linear_attn.{n}.{t}" in meta:
                        err(f"{p}linear_attn.{n} must stay BF16 but has {t}")
            for n in ("conv1d.weight", "norm.weight", "A_log"):
                if f"{p}linear_attn.{n}" not in meta:
                    err(f"missing {p}linear_attn.{n}")
    if lm_head:
        mods.append("lm_head")
    for m in mods:
        ents = {t: meta.get(f"{m}.{t}") for t in ("weight_packed", "weight_scale", "weight_global_scale")}
        for t, dt in (("weight_packed", "U8"), ("weight_scale", "F8_E4M3"), ("weight_global_scale", "F32")):
            if ents[t] is None:
                err(f"missing {m}.{t}")
            elif ents[t]["dtype"] != dt:
                err(f"{m}.{t} dtype {ents[t]['dtype']} want {dt}")
        if f"{m}.weight" in meta:
            err(f"{m}.weight still present (unquantized)")
        wp, ws = ents["weight_packed"], ents["weight_scale"]
        if wp and ws and (wp["shape"][0] != ws["shape"][0] or wp["shape"][1] * 2 != ws["shape"][1] * 16):
            err(f"{m}: weight_packed {wp['shape']} inconsistent with weight_scale {ws['shape']}")
        if ents["weight_global_scale"] and ents["weight_global_scale"]["shape"] not in ([1], []):
            err(f"{m}.weight_global_scale shape {ents['weight_global_scale']['shape']} (want [1])")
    # fused partitions must agree on in_features, and on the global scale (vLLM: weight_global_scale.max())
    for label, names in gs_needed:
        ins = {meta[n.replace("weight_global_scale", "weight_packed")]["shape"][1] for n in names
               if n.replace("weight_global_scale", "weight_packed") in meta}
        if len(ins) > 1:
            err(f"{label}: partitions disagree on packed in_features {ins}")
    try:
        from safetensors import safe_open
        by_shard = defaultdict(list)
        for _, names in gs_needed:
            for n in names:
                if n in meta:
                    by_shard[wmap[n]].append(n)
        gs = {}
        for shard, ks in by_shard.items():
            with safe_open(d / shard, "pt") as f:
                for k in ks:
                    gs[k] = f.get_tensor(k).item()
        bad = 0
        for label, names in gs_needed:
            vals = [gs.get(n) for n in names]
            if None in vals or len(set(vals)) != 1:
                bad += 1
                err(f"{label}: partitions do not share one weight_global_scale: {vals}")
        print(f"attn fused global-scale groups checked: {len(gs_needed)}, mismatches: {bad}")
    except ImportError:
        warns.append("safetensors not importable: skipped attention global-scale equality check")
    print(f"attn quantized modules checked: {len(mods)} (self_attn layers {n_q['self_attn']}, linear_attn layers {n_q['linear_attn']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=DEFAULT)
    ap.add_argument("--size-gb-min", type=float, default=None)
    ap.add_argument("--size-gb-max", type=float, default=None)
    ap.add_argument("--attn", action="store_true", help="attention projections are expected to be NVFP4")
    ap.add_argument("--lm-head", action="store_true", help="lm_head is expected to be NVFP4")
    ap.add_argument("--num-experts", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    a = ap.parse_args()
    d = Path(a.ckpt)
    # expected size: 23.94 GB experts-only; attention (~2.55 GB BF16 -> ~0.7 GB) and lm_head (~1.0 -> ~0.3 GB) shrink it
    lo, hi = (22.5, 25.5)
    if a.attn:
        lo, hi = (20.0, 23.0)
    if a.lm_head:
        lo, hi = (lo - 0.8, hi - 0.8)
    # measured reference builds
    a.size_ref = 21.38 if (a.attn and a.lm_head) else 22.11 if a.attn else 23.94
    a.size_gb_min = lo if a.size_gb_min is None else a.size_gb_min
    a.size_gb_max = hi if a.size_gb_max is None else a.size_gb_max

    cfg = json.loads((d / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n_exp = a.num_experts or tc.get("num_experts", 256)
    n_lay = a.num_layers or tc.get("num_hidden_layers", 40)

    # ---- config.json -------------------------------------------------------
    qc = cfg.get("quantization_config")
    if not qc:
        sys.exit("FAIL: config.json has no quantization_config")
    if qc.get("quant_method") != "compressed-tensors":
        err(f"quant_method={qc.get('quant_method')}")
    if qc.get("format") != "nvfp4-pack-quantized":
        err(f"format={qc.get('format')} (want nvfp4-pack-quantized)")
    groups = qc.get("config_groups", {})
    if len(groups) != 1:
        err(f"expected exactly 1 config group, got {len(groups)}")
    for g in groups.values():
        w = g.get("weights", {})
        if (w.get("num_bits"), w.get("group_size"), w.get("type"), w.get("strategy")) != (4, 16, "float", "tensor_group"):
            err(f"weights spec is not NVFP4: {w}")
        if g.get("input_activations") is not None:
            err("input_activations is set - this should be weight-only (A16)")
        if g.get("targets") != ["Linear"]:
            print(f"note: targets={g.get('targets')}")
    if a.lm_head:
        # vLLM's loader (compressed-tensors match_name: re.match(target[3:], name)) must hit the LM head
        # under both the HF and the vLLM module name; class "Linear" never matches ParallelLMHead.
        lm_names = ("lm_head", "language_model.lm_head")
        if not any(
            t.startswith("re:") and all(re.match(t[3:], n) for n in lm_names)
            for g in groups.values() for t in g.get("targets", [])
        ):
            err("--lm-head: no config group target is a 're:' regex matching both 'lm_head' and "
                "'language_model.lm_head' (e.g. 're:.*lm_head$'); vLLM cannot load the quantized head "
                f"(targets={[g.get('targets') for g in groups.values()]})")
    ignore = qc.get("ignore", [])
    if a.attn:
        needs = ["vision_tower", "vpm", "vit_merger", "merger", "in_proj_[ab]", "mlp\\.gate$", "shared_expert_gate", "conv1d"]
        if not a.lm_head:
            needs.append("lm_head")
    else:
        needs = ["vision_tower", "linear_attn", "self_attn", "merger", "mlp\\.gate$", "shared_expert_gate"]
        if not a.lm_head:
            needs.append("lm_head")
    for need in needs:
        if not any(need in i for i in ignore):
            err(f"ignore list has no entry covering '{need}'")
    if a.attn:
        check_ignore_regexes(ignore, a.lm_head)
    print(f"config: format={qc.get('format')} ignore={len(ignore)} entries")

    # ---- index + headers ---------------------------------------------------
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    meta = {}
    for shard in sorted(set(wmap.values())):
        h = read_header(d / shard)
        h.pop("__metadata__", None)
        for k, v in h.items():
            if wmap.get(k) != shard:
                err(f"{k}: index says {wmap.get(k)} but found in {shard}")
            meta[k] = v
    missing = set(wmap) - set(meta)
    if missing:
        err(f"{len(missing)} index entries missing from shards, e.g. {sorted(missing)[:3]}")

    # ---- experts + shared experts quantized -------------------------------
    pfx = "model.language_model.layers.{l}.mlp."
    shapes = {"gate_proj": (512, 2048), "up_proj": (512, 2048), "down_proj": (2048, 512)}
    hid = tc.get("hidden_size", 2048)
    inter = tc.get("moe_intermediate_size", 512)
    shapes = {"gate_proj": (inter, hid), "up_proj": (inter, hid), "down_proj": (hid, inter)}
    n_q = 0
    for l in range(n_lay):
        p = pfx.format(l=l)
        mods = [f"{p}experts.{e}.{proj}" for e in range(n_exp) for proj in shapes]
        mods += [f"{p}shared_expert.{proj}" for proj in shapes]
        for m in mods:
            proj = m.rsplit(".", 1)[1]
            for t, dt in (("weight_packed", "U8"), ("weight_scale", "F8_E4M3"), ("weight_global_scale", "F32")):
                e = meta.get(f"{m}.{t}")
                if e is None:
                    err(f"missing {m}.{t}")
                    break
                if e["dtype"] != dt:
                    err(f"{m}.{t} dtype {e['dtype']} want {dt}")
                if t == "weight_packed" and e["shape"] != [shapes[proj][0], shapes[proj][1] // 2]:
                    err(f"{m}.{t} shape {e['shape']}")
                if t == "weight_scale" and e["shape"] != [shapes[proj][0], shapes[proj][1] // 16]:
                    err(f"{m}.{t} shape {e['shape']}")
            if f"{m}.weight" in meta:
                err(f"{m}.weight still present (unquantized)")
            n_q += 1
        for fused in ("experts.gate_up_proj", "experts.down_proj"):
            if f"{p}{fused}" in meta:
                err(f"fused 3D tensor {p}{fused} still present")
        if len(errors) > 25:
            break
    print(f"quantized modules checked: {n_q} (expect {n_lay * (n_exp * 3 + 3)})")


    # ---- gate/up of each expert must share one weight_global_scale (vLLM packs them into w13)
    try:
        from safetensors import safe_open
        gs = {}
        want = [k for k in meta if k.endswith((".gate_proj.weight_global_scale", ".up_proj.weight_global_scale"))]
        by_shard = defaultdict(list)
        for k in want:
            by_shard[wmap[k]].append(k)
        for shard, ks in by_shard.items():
            with safe_open(d / shard, "pt") as f:
                for k in ks:
                    gs[k] = f.get_tensor(k).item()
        mism = 0
        for k, v in gs.items():
            if k.endswith(".gate_proj.weight_global_scale"):
                u = gs.get(k.replace(".gate_proj.", ".up_proj."))
                if u != v:
                    mism += 1
                    if mism <= 3:
                        err(f"gate/up global scale mismatch at {k}: {v} vs {u}")
        if mism:
            err(f"{mism} experts have different gate/up weight_global_scale")
        print(f"gate/up global-scale pairs checked: {len(gs) // 2}, mismatches: {mism}")
    except ImportError:
        warns.append("safetensors not importable: skipped gate/up global-scale equality check")

    if a.attn:
        check_attn(meta, wmap, d, n_lay, a.lm_head)

    # ---- everything else must be unquantized BF16 --------------------------
    quant_suffix = re.compile(r"\.(weight_packed|weight_scale|weight_global_scale|weight_shape|input_global_scale)$")
    quantized_mods = {quant_suffix.sub("", k) for k in meta if quant_suffix.search(k)}
    allowed_q = r"\.mlp\.(experts\.\d+|shared_expert)\.(gate|up|down)_proj$"
    if a.attn:
        allowed_q += (r"|\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
                      r"|\.language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$")
    if a.lm_head:
        allowed_q += r"|^lm_head$"
    bad_q = [m for m in quantized_mods if not re.search(allowed_q, m)]
    if bad_q:
        err(f"{len(bad_q)} unexpected quantized modules, e.g. {sorted(bad_q)[:5]}")
    keep = defaultdict(int)
    keep_pats = {
        "vision_tower": r"\.vision_tower\.",
        "merger": r"\.merger\.",
        "linear_attn": r"\.linear_attn\.",
        "self_attn": r"\.language_model\.layers\.\d+\.self_attn\.",
        "router/gates": r"\.mlp\.gate\.weight$|\.shared_expert_gate\.weight$",
        "embed/lm_head": r"embed_tokens\.weight$|^lm_head\.weight$",
    }
    if a.attn:
        # quantized projections legitimately carry U8/F8/F32 tensors; only the leftovers must be BF16
        keep_pats["linear_attn"] = r"\.linear_attn\.(in_proj_[ab]\.weight|conv1d\.weight|A_log|dt_bias|norm\.weight)$"
        keep_pats["self_attn"] = r"\.language_model\.layers\.\d+\.self_attn\.(q|k)_norm\.weight$"
    if a.lm_head:
        keep_pats["embed/lm_head"] = r"embed_tokens\.weight$"
    for k, e in meta.items():
        for name, pat in keep_pats.items():
            if re.search(pat, k):
                keep[name] += 1
                if e["dtype"] not in ("BF16",) and not (e["dtype"] == "F32" and k.endswith(("A_log", "dt_bias"))):
                    err(f"{k}: dtype {e['dtype']} (want BF16)")
                break
    for name in keep_pats:
        if keep[name] == 0:
            err(f"no tensors matched group '{name}'")
    print("BF16 groups:", dict(keep))
    if any(k.startswith("mtp.") for k in meta):
        warns.append("mtp.* present")

    # ---- size --------------------------------------------------------------
    total = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    gb = total / 1e9
    print(f"safetensors total: {gb:.2f} GB (reference build: ~{a.size_ref:.2f} GB)")
    if not (a.size_gb_min <= gb <= a.size_gb_max):
        err(f"size {gb:.2f} GB outside [{a.size_gb_min}, {a.size_gb_max}]")

    # ---- aux files -----------------------------------------------------------
    for f in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
              "processor_config.json", "chat_template.jinja", "generation_config.json"):
        if not (d / f).exists():
            err(f"missing aux file {f}")

    for w in warns:
        print("WARN:", w)
    if errors:
        print(f"\nFAIL: {len(errors)} problem(s)")
        for e in errors[:40]:
            print("  -", e)
        sys.exit(1)
    print("\nOK: checkpoint structure matches the NVFP4A16 recipe")


if __name__ == "__main__":
    main()
