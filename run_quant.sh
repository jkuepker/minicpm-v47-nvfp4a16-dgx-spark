#!/usr/bin/env bash
# NVFP4A16 quantization of openbmb/MiniCPM-V-4.7-35B-A3B inside the vLLM aarch64 nightly image.
#
# Default build: every Linear in the language model is 4-bit (experts, shared experts, linear_attn
# and self_attn projections, lm_head). Router gates, shared_expert_gate, linear_attn in_proj_a/in_proj_b,
# the vision tower and merger, embeddings, conv1d and norms stay BF16. Output is about 21.4 GB.
# EXPERTS_ONLY=1 gives the older build with only the routed and shared experts in 4-bit (about 23.9 GB, ~2.6x slower decode).
#
# Runs on the CPU by default: about 10 minutes on the Grace CPU of a DGX Spark, a few GB of
# memory per worker, and it works while a vLLM server is using the GPU. Downloads ~70 GB
# (BF16 weights) into the HF cache if not already there, and writes ~21 GB. Run from a directory
# containing quantize_nvfp4a16.py and verify_ckpt.py:   ./run_quant.sh
#
# GPU instead (stop any vLLM server first):  DEVICE=cuda:0 GPU_ARGS="--gpus all" MAX_WORKERS=1 ./run_quant.sh
#
# Overridable env: IMAGE SRC DST HF_CACHE GPU_ARGS DEVICE MAX_WORKERS EXPERTS_ONLY DRY_RUN VERIFY_ARGS
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ghcr.io/jkuepker/minicpm-v47-nvfp4a16-dgx-spark:vllm0.31.1rc1.dev296}"  # or a plain vLLM nightly-aarch64 image
SRC="${SRC:-openbmb/MiniCPM-V-4.7-35B-A3B}"          # hub id, or a path visible inside the container
DST="${DST:-$HOME/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
GPU_ARGS="${GPU_ARGS-}"                               # empty = no GPU passed to the container (CPU run)
DEVICE="${DEVICE:-cpu}"                               # cpu, or cuda:0 together with GPU_ARGS="--gpus all"
MAX_WORKERS="${MAX_WORKERS:-2}"                       # concurrent shards; use 1 on the GPU
EXPERTS_ONLY="${EXPERTS_ONLY:-0}"                     # 1 => only the experts in 4-bit (older, slower build)
VERIFY_ARGS="${VERIFY_ARGS:-}"
if [ "$EXPERTS_ONLY" = 1 ]; then
  QUANT_ARGS=""
else
  QUANT_ARGS="--quantize-attn --quantize-lm-head"
  VERIFY_ARGS="--attn --lm-head $VERIFY_ARGS"
fi
DST_PARENT="$(dirname "$DST")"

if [ "${DRY_RUN:-0}" != 1 ]; then mkdir -p "$DST_PARENT" "$HF_CACHE"; fi
if ls "$DST"/*.safetensors >/dev/null 2>&1; then
  echo "refusing: $DST already holds safetensors (rm -rf it first)" >&2; exit 1
fi
avail_gb=$(df -Pk "$DST_PARENT" 2>/dev/null | awk 'NR==2{print int($4/1048576)}' || true)
echo "free disk at $DST_PARENT: ${avail_gb} GB (need ~25 GB out, plus ~70 GB in $HF_CACHE if not cached)"

# Everything below runs inside the container (single-quoted: expanded there, not here).
read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
PY=python3
# Bundled image: quantization tools are preinstalled in /opt/quant, nothing to install.
if [ -x /opt/quant/bin/python ] && /opt/quant/bin/python -c 'from llmcompressor import model_free_ptq' 2>/dev/null; then
  PY=/opt/quant/bin/python
  echo "== bundled image: using preinstalled /opt/quant (llm-compressor $($PY -c 'import llmcompressor;print(llmcompressor.__version__)'))"
  SKIP_INSTALL=1
fi
echo "== container: $($PY --version), torch $($PY -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu-only")')"
$PY -c 'import transformers;print("transformers", transformers.__version__)'

if [ -z "${SKIP_INSTALL:-}" ]; then
  # pip front-end: image may only ship uv
  if $PY -m pip --version >/dev/null 2>&1; then
    PIP() { $PY -m pip install -q "$@"; }
    FREEZE() { $PY -m pip freeze; }
  else
    PIP() { uv pip install -q --python "$(command -v $PY)" "$@"; }
    FREEZE() { uv pip freeze --python "$(command -v $PY)"; }
  fi

  # Keep the image's torch / transformers / numpy / hub stack as is. llm-compressor and
  # compressed-tensors are installed --no-deps so their (older) pins cannot downgrade it.
  FREEZE 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|transformers|numpy|safetensors|huggingface[-_]hub|tokenizers|accelerate)==' > /tmp/constraints.txt || true
  PIP --no-deps "llmcompressor==0.14.0" "compressed-tensors==0.19.0"
  PIP -c /tmp/constraints.txt loguru pydantic psutil requests pyyaml tqdm pillow datasets accelerate nvidia-ml-py

  # MiniCPM-V-4.7 needs transformers >= 5.18 only for loading the HF model class; model_free_ptq
  # never instantiates it, it just needs llmcompressor to import. Make sure the import works:
  if ! $PY -c 'from llmcompressor import model_free_ptq' 2>/tmp/imp.err; then
    cat /tmp/imp.err >&2
    echo "llmcompressor import failed with image transformers; installing transformers==5.18.0" >&2
    PIP "transformers==5.18.0"
    $PY -c 'from llmcompressor import model_free_ptq'
  fi
fi
$PY -c 'import transformers;print("transformers (final)", transformers.__version__)'

ARGS=(--src "$SRC" --dst "$DST" --max-workers "$MAX_WORKERS")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
# shellcheck disable=SC2086  # QUANT_ARGS and VERIFY_ARGS are space-separated flag lists
$PY /work/quantize_nvfp4a16.py "${ARGS[@]}" $QUANT_ARGS
# shellcheck disable=SC2086
$PY /work/verify_ckpt.py "$DST" $VERIFY_ARGS
# the container runs as root: hand outputs back to the invoking user
if [ -n "${HOST_UID:-}" ]; then
  chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$DST" /root/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4.7-35B-A3B \
    /root/.cache/huggingface/.locks 2>/dev/null || true
fi
INNER_EOF

DOCKER_CMD=(docker run --rm $GPU_ARGS --ipc=host
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_ENABLE_HF_TRANSFER=0
  -e HF_TOKEN -e SRC="$SRC" -e DST="$DST" -e DEVICE="$DEVICE" -e MAX_WORKERS="$MAX_WORKERS"
  -e VERIFY_ARGS="$VERIFY_ARGS" -e QUANT_ARGS="$QUANT_ARGS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e PYTHONUNBUFFERED=1
  -v "$HF_CACHE":/root/.cache/huggingface
  -v "$DST_PARENT":"$DST_PARENT"
  -v "$HERE":/work:ro
  ${EXTRA_DOCKER_ARGS:-}
  --entrypoint bash "$IMAGE" -c "$INNER")

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DST=$DST DEVICE=$DEVICE MAX_WORKERS=$MAX_WORKERS GPU_ARGS='$GPU_ARGS'"
  echo "QUANT_ARGS='$QUANT_ARGS' VERIFY_ARGS='$VERIFY_ARGS'"
  printf '%q ' "${DOCKER_CMD[@]:0:12}"; echo "..."; exit 0
fi
"${DOCKER_CMD[@]}"
echo "done -> $DST"
