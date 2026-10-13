#!/usr/bin/env bash
# serve.sh MODEL_PATH [QUANT_FLAG]
#
# Serve MiniCPM-V 4.7 35B-A3B with vLLM in Docker on a single DGX Spark (GB10).
#   ./serve.sh ~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16     # the NVFP4A16 build from this recipe
#   ./serve.sh openbmb/MiniCPM-V-4.7-35B-A3B-AWQ           # OpenBMB's AWQ build (config fixes needed too)
#
# Prints the docker command, then runs it. DRY_RUN=1 only prints.
# QUANT_FLAG is optional: leave it empty, vLLM reads quantization_config from the checkpoint
# (both compressed-tensors NVFP4A16 and AWQ are auto-detected).
#
# Flags that matter:
#   --reasoning-parser minicpmv      splits <think>, and repairs the literal "\n" the tokenizer emits
#   --tool-call-parser qwen3_coder   the pair tested in vLLM PR #58674 (qwen3_xml / qwen3 are not)
#   no --trust-remote-code           the model is native in vLLM main / transformers >= 5.18
#   no --moe-backend                 let vLLM choose (it picks Marlin for W4A16 MoE on GB10)
#
# DSPARK=1 adds speculative decoding with OpenBMB's DSpark draft model (4 speculative tokens, the
# best setting we measured). It helps below about 16k tokens of context (+32% decode at 4k),
# breaks even around 16k and hurts above (-23% at 32k). Leave it off for long-context work.
# The draft model is downloaded from the Hub on first start, so do not combine it with OFFLINE=1
# unless it is already in the HF cache.
#
# Env overrides: IMAGE NAME BIND_ADDR PORT MAX_LEN GPU_UTIL MAX_SEQS MAX_BATCHED KV_DTYPE
#   LIMIT_MM MM_KWARGS TRUST_REMOTE_CODE=1 OFFLINE=1 HF_HOME MODELS_DIR EXTRA DSPARK=1 DRY_RUN=1
set -euo pipefail
MODEL_PATH="${1:?usage: serve.sh MODEL_PATH [QUANT_FLAG]}"
QUANT_FLAG="${2:-}"

IMAGE="${IMAGE:-ghcr.io/jkuepker/minicpm-v47-nvfp4a16-dgx-spark:vllm0.31.1rc1.dev296}"  # or any vLLM nightly-aarch64 after 2026-10-11 (PR #58674)
NAME="${NAME:-minicpmv47}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"      # use 0.0.0.0 to expose on your LAN
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-131072}"             # the model supports 262144
GPU_UTIL="${GPU_UTIL:-0.30}"             # fraction of the 128 GB unified memory; weights ~21 GiB + KV
MAX_SEQS="${MAX_SEQS:-2}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
KV_DTYPE="${KV_DTYPE:-fp8}"              # auto = bf16 KV
LIMIT_MM="${LIMIT_MM:-{\"image\":4,\"video\":0}}"   # video 0 = no video profiling memory
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OFFLINE="${OFFLINE:-0}"                  # 1 = HF_HUB_OFFLINE, for a fully local model dir

SERVE=(vllm serve "$MODEL_PATH"
  --host "$BIND_ADDR" --port "$PORT"
  --served-model-name minicpmv47
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --kv-cache-dtype "$KV_DTYPE"
  --gpu-memory-utilization "$GPU_UTIL"
  --limit-mm-per-prompt "$LIMIT_MM"
  --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser minicpmv)
[[ -n "$QUANT_FLAG" ]] && SERVE+=(--quantization "$QUANT_FLAG")
[[ -n "${MM_KWARGS:-}" ]] && SERVE+=(--mm-processor-kwargs "$MM_KWARGS")
[[ "${TRUST_REMOTE_CODE:-0}" == 1 ]] && SERVE+=(--trust-remote-code)
[[ "${DSPARK:-0}" == 1 ]] && SERVE+=(--speculative-config
  '{"method":"dspark","model":"openbmb/MiniCPM-V-4.7-35B-A3B-DSpark","num_speculative_tokens":4,"enable_adaptive_verification":false}')
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && SERVE+=($EXTRA)

# bash entrypoint + `vllm serve ...` via -lc, so the image's own entrypoint cannot double up "vllm serve"
SERVE_STR="$(printf '%q ' "${SERVE[@]}")"
DOCKER=(docker run -d --name "$NAME" --gpus all --network host --ipc=host --shm-size 16g
  --platform linux/arm64
  -v "$HF_HOME:/root/.cache/huggingface" -v "$MODELS_DIR:$MODELS_DIR:ro"
  -e HF_HUB_OFFLINE="$OFFLINE" -e VLLM_LOGGING_LEVEL=INFO
  --entrypoint /bin/bash "$IMAGE" -lc "$SERVE_STR")
# Sampling is not set here: vLLM applies the checkpoint's generation_config.json
# (temperature 1.0, top_k 20, top_p 0.95) unless a request overrides it.
# Thinking is ON by default (the chat template emits <think>).

printf '%q ' "${DOCKER[@]}"; printf '\n'
printf '# wait:  until curl -sf http://%s:%s/health; do sleep 5; done\n' "$BIND_ADDR" "$PORT"
printf '# check: docker logs %s 2>&1 | grep -iE "moe backend|minicpm|parser|error" | tail\n' "$NAME"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec "${DOCKER[@]}"
