> Recipe and bundled engine for running [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) as an NVFP4 W4A16 checkpoint on one DGX Spark. This repo has scripts and a Dockerfile, not model weights; the same recipe is on Hugging Face at [kuepker/MiniCPM-V-4.7-35B-A3B-NVFP4A16-1x-DGX-Spark](https://huggingface.co/kuepker/MiniCPM-V-4.7-35B-A3B-NVFP4A16-1x-DGX-Spark). Scripts are Apache-2.0; the model and anything built from it stay under OpenBMB's license for the weights (also Apache-2.0).

# MiniCPM-V-4.7-35B-A3B-NVFP4A16-1x-DGX-Spark

Name key: `MiniCPM-V-4.7-35B-A3B` is OpenBMB's vision-language MoE (35B total, ~3B active); `NVFP4A16` is NVFP4 weights with 16-bit (BF16) activations, i.e. W4A16; `1x-DGX-Spark` is one NVIDIA DGX Spark (GB10), served with vLLM.

## TL;DR

- Quantize `openbmb/MiniCPM-V-4.7-35B-A3B` (BF16) to NVFP4A16 with llm-compressor `model_free_ptq`, inside the vLLM aarch64 nightly image. About 10 minutes on the Spark's CPU, a few GB of memory per worker, 21.38 GB output.
- Every Linear in the language model is 4-bit: routed and shared experts, `linear_attn` and `self_attn` projections, and `lm_head`. Router gates, `shared_expert_gate`, `linear_attn` `in_proj_a`/`in_proj_b`, the vision tower and merger, embeddings, `conv1d` and norms stay BF16.
- Decode is about 100 tok/s for a single request on one Spark. That is 2.6x the earlier experts-only NVFP4 build of this model (38.6 to 38.9 tok/s), 3.3x BF16, and faster than `nvidia/Qwen3.6-35B-A3B-NVFP4` on the same engine (77 to 79 tok/s).
- Quality cost is about 1 MMLU point: 83.1 vs 84.1 for BF16. Vision, tool calls and HumanEval are unchanged; all 18 long-context code tasks (4k to 32k tokens) complete. GSM8K and IFEval use 25 items each and are noisy.
- Optional: `DSPARK=1` in `serve.sh` turns on speculative decoding with OpenBMB's DSpark draft model. It adds roughly 15 to 18 tok/s at short context and slows decode at long context (break-even is around 16k tokens).
- You need a vLLM nightly from 2026-10-11 or later (MiniCPM-V 4.7 support is vllm-project/vllm PR #58674, not in v0.31.0), or v0.32+ once released. The numbers below are from `0.31.1rc1.dev296+gfe4df3df0`.
- Two checkpoint fixes are required before vLLM will load it (`auto_map` removal, processor class rename). `fix_processor_configs.py` does both. The same fixes apply to OpenBMB's own AWQ repo.
- vLLM serves the W4A16 MoE through Marlin on GB10. Do not pass `--moe-backend`.

Set `EXPERTS_ONLY=1` for `run_quant.sh` to get the older build that only quantizes the experts (23.94 GB, about 39 tok/s, MMLU 83.9).

Files in this recipe: `run_quant.sh`, `quantize_nvfp4a16.py`, `verify_ckpt.py`, `fix_processor_configs.py`, `serve.sh`. Put them in one directory and run everything from there.

## Faster on the same Spark: TensorFold

[TensorFold](https://github.com/jkuepker/TensorFold/blob/minicpmv47-066/docs/recipes/minicpm-v-4.7.md) (a fork branch on the 0.6.6 release) serves MiniCPM-V 4.7 from a [published MLX 4-bit checkpoint](https://huggingface.co/kuepker/MiniCPM-V-4.7-35B-A3B-MLX-4bit-TensorFold) and drafts with Qwen3.6-35B-A3B's MTP layer. MiniCPM's language model is a Qwen3.6 fine-tune, so that layer's drafts are mostly accepted. On the same Spark, measured with the same tool:

| | decode @ ~1k prompt | @ ~16k | long-context code avg | MMLU |
|---|---:|---:|---:|---:|
| This recipe (vLLM, NVFP4) | 102.5 | 93.1 | 92.0 | 83.1 |
| TensorFold, MLX 4-bit + Qwen3.6 MTP drafts | 153.2 | 142.1 | 161.7 | 83.6 |

Trade-offs: image requests run one at a time; without drafts TensorFold decodes about 10% slower than vLLM. Use this recipe when you need concurrent image requests or a stock vLLM stack.

## Requirements

- NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory), Docker with the NVIDIA runtime.
- Disk: about 70 GB for the BF16 download (`~/.cache/huggingface`), 22 GB for the output (`~/models`).
- Quantization runs on the CPU by default, so a vLLM server can keep running during it. Serving needs the GPU, and unified memory is shared with everything else on the box.
- A Hugging Face CLI on the host for the downloads: `pip install -U huggingface_hub` (provides `hf`).

## 1. Pull the image

The engine is bundled: one image with the exact vLLM build this recipe was tested on, the quantization tools, and the scripts. Both `serve.sh` and `run_quant.sh` use it by default.

```bash
docker pull ghcr.io/jkuepker/minicpm-v47-nvfp4a16-dgx-spark:vllm0.31.1rc1.dev296
```

If the pull is denied, use the plain vLLM image the bundle is built from (the scripts then install the quantization tools on each run):

```bash
export IMAGE=vllm/vllm-openai@sha256:d044fa021560985593390653000281ee57e0c8bcd5e8f3b94710ec335c0544a3
```

What is inside (built from the `Dockerfile` in this repo, about 26 GB, linux/arm64):

- vLLM 0.31.1rc1.dev296+gfe4df3df0 (transformers 5.19.0, CUDA 13.0.2), unchanged, from `vllm/vllm-openai@sha256:d044fa021560985593390653000281ee57e0c8bcd5e8f3b94710ec335c0544a3`.
- `/opt/quant`: a venv with llm-compressor 0.14.0 and compressed-tensors 0.19.0 for quantizing. It is separate because the image's vLLM needs compressed-tensors 0.17.0.
- `/recipe`: the scripts from this repo.

To use a plain vLLM image instead, set `IMAGE`, for example `export IMAGE=vllm/vllm-openai:nightly-aarch64` (any nightly after 2026-10-11 or v0.32+ should work). `run_quant.sh` then installs the quantization tools into a throwaway container on each run.

## 2. Download

BF16 source (about 70 GB), into the normal HF cache so the quantizer reuses it:

```bash
hf download openbmb/MiniCPM-V-4.7-35B-A3B
```

Processor, tokenizer and generation configs from OpenBMB's AWQ repo (small files only). The BF16 repo's `processor_config.json` is just `{}` and its `preprocessor_config.json` is minimal, so use the full ones from the AWQ repo (same `tokenizer.json` and `chat_template.jinja`):

```bash
hf download openbmb/MiniCPM-V-4.7-35B-A3B-AWQ \
  processor_config.json preprocessor_config.json tokenizer_config.json \
  tokenizer.json chat_template.jinja generation_config.json \
  --local-dir ~/models/awq-configs
```

## 3. Quantize

```bash
chmod +x run_quant.sh
./run_quant.sh
```

This starts the bundled image (CPU only, no `--gpus`), uses the preinstalled llm-compressor 0.14.0 in `/opt/quant` (with a plain vLLM image it installs `llmcompressor==0.14.0` and `compressed-tensors==0.19.0` with `pip --no-deps` first, so the image's torch/transformers stay untouched), runs `quantize_nvfp4a16.py --quantize-attn --quantize-lm-head`, then `verify_ckpt.py --attn --lm-head`. Output goes to `~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16`. On a Spark's Grace CPU the quantization step took about 9 minutes (about 10 with the install and verify), with the BF16 weights already in the HF cache.

Overridable env: `IMAGE SRC DST HF_CACHE GPU_ARGS DEVICE MAX_WORKERS EXPERTS_ONLY DRY_RUN VERIFY_ARGS`. If the source repo is gated or you are rate limited, `export HF_TOKEN=...` first. `DRY_RUN=1` prints the resolved arguments and the start of the docker command without running anything.

- `EXPERTS_ONLY=1` quantizes only the routed and shared experts (the older build).
- CPU is the default: `DEVICE=cpu`, `GPU_ARGS=""`, `MAX_WORKERS=2`. To use the GPU instead, stop any vLLM server first and run `DEVICE=cuda:0 GPU_ARGS="--gpus all" MAX_WORKERS=1 ./run_quant.sh`.

What the script does:

- `model_free_ptq(scheme="NVFP4A16")` streams the BF16 safetensors one shard at a time and never instantiates the model, so memory stays at a few GB per worker.
- The fused 3D expert tensors (`experts.gate_up_proj`, `experts.down_proj`) are split into per-expert 2D linears (`experts.{e}.{gate,up,down}_proj`) and saved unfused.
- Partitions that vLLM fuses at load time share one `weight_global_scale`: expert `gate_proj`/`up_proj` (and the shared expert's), `q/k/v` in `self_attn`, and `in_proj_qkv`/`in_proj_z` in `linear_attn`. llm-compressor 0.14.0 does not do this, and vLLM takes the max of the partitions' scales (the NVFP4A16 MoE path reads only the w13 scale) and only logs a warning, so mismatched scales would silently hurt accuracy. The patch is a no-op on llm-compressor versions that do it natively, and `verify_ckpt.py` checks every group.
- With `--quantize-lm-head` the script appends the target `re:.*lm_head$` to the `targets` in `config.json` (llm-compressor writes only `["Linear"]`). Without it vLLM does not load the quantized `lm_head` (see Gotchas).
- The skip list names the vision modules twice: by their checkpoint names (`model.vision_tower.*`) and by the names vLLM uses at load time (`vpm.*`, `vit_merger.*`). vLLM remaps exact module names in an ignore list to its own names, but not `re:` patterns, so a pattern that only says `vision_tower` leaves vLLM trying to load the BF16 vision encoder as NVFP4, and the engine fails to start.
- Result: NVFP4 group size 16, fp8-e4m3 block scales, fp32 global scale, W4A16.

Why these choices:

- **Attention and `lm_head` in 4-bit.** Decode on this model reads every active weight once per token. With only the experts quantized, the BF16 `linear_attn`, `self_attn` and `lm_head` weights made up most of the bytes read (see Why this is fast).
- **Kept in BF16.** Router gates, `shared_expert_gate` and `linear_attn` `in_proj_a`/`in_proj_b` are small, so keeping them in BF16 costs little. The vision tower and merger also stay BF16. Embeddings, `conv1d` and norms are not Linear layers.
- **No FP8.** NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` uses a modelopt MIXED_PRECISION recipe with FP8 for attention layers. We observed that FP8 linear layers route through a cuDNN FP8 GEMM that can deadlock vLLM on GB10 (sm_121) under long soak tests, so everything here is either NVFP4 W4A16 or BF16. We have not root-caused the deadlock.
- **W4A16, not W4A4.** On GB10 vLLM serves W4A16 MoE through Marlin. W4A4 NVFP4 MoE on sm_121 has a history of corrupt output when forcing the CUTLASS backend.

## 4. Fix the configs

Copy the full processor/tokenizer files over the quantized output, then apply the fixes:

```bash
OUT=~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16
cp ~/models/awq-configs/{processor_config.json,preprocessor_config.json,tokenizer_config.json,tokenizer.json,chat_template.jinja,generation_config.json} "$OUT"/
python3 fix_processor_configs.py "$OUT"
```

`fix_processor_configs.py` (idempotent; safe on read-only files and hardlinks, because it unlinks before writing):

1. removes every `auto_map` key from `processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json`. They point at `processing_minicpmv4_7.py` and friends, which are not in the HF repos, so `--trust-remote-code` cannot help;
2. renames `MiniCPMV4_7ImageProcessor` to `MiniCPMV4_6ImageProcessor` and `MiniCPMV4_7VideoProcessor` to `MiniCPMV4_6VideoProcessor`, because transformers registers `minicpmv4_7` with the 4.6 classes.

Run it on OpenBMB's AWQ checkpoint too if you want to serve that one (see step 6).

## 5. Verify

`run_quant.sh` already ran `verify_ckpt.py --attn --lm-head` once, before step 4. Run it again afterwards (it needs `safetensors` for the global-scale check, so use the container):

```bash
docker run --rm -v ~/models:/models:ro -v "$PWD":/work:ro --entrypoint python3 \
  ghcr.io/jkuepker/minicpm-v47-nvfp4a16-dgx-spark:vllm0.31.1rc1.dev296 /work/verify_ckpt.py /models/MiniCPM-V-4.7-35B-A3B-NVFP4A16 --attn --lm-head
```

Expected on the real checkpoint:

```
quantized modules checked: 30840 (expect 30840)
gate/up global-scale pairs checked: 10280, mismatches: 0
attn fused global-scale groups checked: 40, mismatches: 0
attn quantized modules checked: 131 (self_attn layers 10, linear_attn layers 30)
safetensors total: 21.38 GB
OK: checkpoint structure matches the NVFP4A16 recipe
```

The size line prints the reference size for the chosen flags (21.38 GB for this build). If you built with `EXPERTS_ONLY=1`, leave out `--attn --lm-head` here (expect 23.94 GB).

## 6. Serve

```bash
./serve.sh ~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16
```

`serve.sh` prints and runs this (set `DRY_RUN=1` to only print). It listens on `127.0.0.1:8000`; set `BIND_ADDR=0.0.0.0` to expose it:

```bash
docker run -d --name minicpmv47 --gpus all --network host --ipc=host --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models:$HOME/models:ro \
  --entrypoint /bin/bash ghcr.io/jkuepker/minicpm-v47-nvfp4a16-dgx-spark:vllm0.31.1rc1.dev296 -lc \
  'vllm serve '"$HOME"'/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16 \
     --host 127.0.0.1 --port 8000 --served-model-name minicpmv47 \
     --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 \
     --kv-cache-dtype fp8 --gpu-memory-utilization 0.30 \
     --limit-mm-per-prompt '"'"'{"image":4,"video":0}'"'"' \
     --enable-prefix-caching --enable-chunked-prefill \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser minicpmv'
```

(The script also passes `--enable-prompt-tokens-details`, `--platform linux/arm64` and `-e` settings that are left out above for readability.)

Wait for it, and check which backend was picked:

```bash
until curl -sf http://127.0.0.1:8000/health; do sleep 5; done
docker logs minicpmv47 2>&1 | grep -iE "moe backend|minicpm|parser|error" | tail
```

### Optional: DSpark speculative decoding

```bash
DSPARK=1 ./serve.sh ~/models/MiniCPM-V-4.7-35B-A3B-NVFP4A16
```

This adds the following flag (the draft model is downloaded from the Hub on first start):

```
--speculative-config '{"method":"dspark","model":"openbmb/MiniCPM-V-4.7-35B-A3B-DSpark","num_speculative_tokens":4,"enable_adaptive_verification":false}'
```

Use it when your prompts stay below about 16k tokens. In our runs it raised single-request decode from 99-103 to 115-118 tok/s at 114 to 4k tokens (+32% on real code tasks at 4k). On real code tasks it broke even at about 16k and was slower above that (-23% at 32k), because the draft cost grows with context. On a ~16k filler prompt it was already slower (62.6 vs 93.1 tok/s). Leave it off for long-context work. 4 speculative tokens measured best (7 and 15 were worse, see Results). Decode speed on very predictable text such as boilerplate code can be much higher (about 250 tok/s on HumanEval) and is not representative.

### Notes on the flags

- `--reasoning-parser minicpmv` with `--tool-call-parser qwen3_coder` is the pair tested in PR #58674 (`qwen3_xml` / `qwen3` are not).
- No `--trust-remote-code` (the model is native in vLLM main) and no `--moe-backend` (let vLLM choose).
- KV cache is tiny: only 10 of the 40 layers are full attention, so 131072 tokens at fp8 KV is about 1.3 GB. You can lower `GPU_UTIL` if you share the box.
- `--limit-mm-per-prompt '{"image":4,"video":0}'` skips video profiling memory. Raise `video` to test video.
- The tokenizer decodes newlines as a literal `\n`; the `minicpmv` reasoning parser repairs that. Do not swap it for another parser.

What to expect in the log for this build:

- `Using MarlinNvFp4LinearKernel for NVFP4 GEMM` and `Using 'MARLIN' NvFp4 MoE backend`.
- `Your GPU does not have native support for FP4 computation ...` is expected and harmless: this checkpoint is weight-only FP4 (W4A16), which is exactly what Marlin runs.
- `Model loading took 19.97 GiB memory` (about 2 minutes from a warm page cache).
- vLLM also suggests `VLLM_MARLIN_USE_ATOMIC_ADD=1` for small `size_n` matrices (the experts are 512 wide). The numbers below were measured without it.

Serving OpenBMB's AWQ build instead (after running `fix_processor_configs.py` on its directory; log shows `Using 'MARLIN' WNA16 MoE backend`):

```bash
hf download openbmb/MiniCPM-V-4.7-35B-A3B-AWQ --local-dir ~/models/MiniCPM-V-4.7-35B-A3B-AWQ
python3 fix_processor_configs.py ~/models/MiniCPM-V-4.7-35B-A3B-AWQ
./serve.sh ~/models/MiniCPM-V-4.7-35B-A3B-AWQ
```

## 7. Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool | head
```

Text, thinking on (the default). The reasoning comes back in `message.reasoning` (`reasoning_content` on older vLLM), the answer in `message.content`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "minicpmv47",
  "messages": [{"role": "user", "content": "How many r letters are in strawberry?"}],
  "max_tokens": 1024
}' | python3 -m json.tool
```

Thinking off, per request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "minicpmv47",
  "messages": [{"role": "user", "content": "Reply with one word: hello"}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 -m json.tool
```

Image request (any local PNG):

```bash
IMG=$(base64 -w0 < photo.png)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<JSON | python3 -m json.tool
{
  "model": "minicpmv47",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,${IMG}"}},
    {"type": "text", "text": "Describe this image. If it contains digits, read them out."}
  ]}],
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
```

Tool call:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "minicpmv47",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 1024
}' | python3 -m json.tool
```

Look for `tool_calls` with `get_weather` and `{"city": "Paris"}`. Add `"stream": true` to check streaming tool calls.

What passed in our runs: a large-digit vision probe 7/7, streaming and non-streaming tool calls, and the reasoning/content split.

## Results

All numbers are from a single DGX Spark (GB10, 128 GB unified memory), vLLM nightly-aarch64 `0.31.1rc1.dev296+gfe4df3df0`, `--gpu-memory-utilization 0.30`, fp8 KV cache, `--max-model-len 131072`, prefix caching, temperature 0, thinking off. The speed harness sends streamed OpenAI-style requests with 128 output tokens and takes the median of 3 repetitions. The Qwen3.6 row is `nvidia/Qwen3.6-35B-A3B-NVFP4` on the same nightly and flags.

### Builds

| Build | What is 4-bit (NVFP4 W4A16, group 16) | Checkpoint | Load |
|---|---|---:|---:|
| Experts only (`EXPERTS_ONLY=1`) | routed and shared experts | 23.94 GB | 22.36 GiB |
| Attention | experts, plus `linear_attn` `in_proj_qkv`/`in_proj_z`/`out_proj` and `self_attn` q/k/v/o | 22.11 GB | ~20.6 GiB |
| **Attention + lm_head (this recipe)** | all of the above, plus `lm_head` | **21.38 GB** | **19.97 GiB** |

### Speed: decode tok/s, single request, by prompt length

The prompt is filler text of about 114 tokens, 1k, 4k and 16k.

| Build | 114 | 1k | 4k | 16k | 2 streams, total | Prefill @16k |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B NVFP4 | 77.4 | 79.0 | 78.1 | 73.6 | 90.6 | 6,969 |
| MiniCPM-V 4.7 BF16 | 30.6 | 30.6 | 30.5 | 29.8 | 48.5 | 4,821 |
| MiniCPM-V 4.7 AWQ (OpenBMB) | 38.2 | 38.2 | 38.0 | 37.4 | 70.4 | 6,495 |
| NVFP4A16, experts only | 38.6 | 38.9 | 38.8 | 37.9 | 70.9 | 6,441 |
| NVFP4A16, attention | 69.0 | 70.3 | 69.8 | 66.1 | 97.0 | 6,237 |
| **NVFP4A16, attention + lm_head (this recipe)** | **99.4** | **102.5** | **101.3** | **93.1** | **115.1** | **6,485** |
| attention + lm_head, `DSPARK=1` | 117.4 | 118.2 | 115.3 | 62.6 | 136.4 | 6,185 |

Prefill is unchanged by the quantization changes (about 6,200 to 6,500 tok/s at 16k).

### Speed: decode tok/s on real long-context code tasks

18 tasks, 6 each at about 4k, 16k and 32k prompt tokens. All 18 pass in every row. Decode tok/s is averaged per bucket.

| Build | ~4k | ~16k | ~32k | Average |
|---|---:|---:|---:|---:|
| NVFP4A16, experts only | 38.7 | 37.8 | ~37 | 37.7 |
| NVFP4A16, attention | 69.0 | 65.6 | 61.5 | 65.4 |
| **NVFP4A16, attention + lm_head** | **99.9** | **92.3** | **83.9** | **92.0** |
| attention + lm_head, `DSPARK=1` | 131.6 | 92.4 | 65.0 | 96.3 |

### DSpark sweep

Number of speculative tokens, measured on the experts-only and attention builds before the lm_head build existed (average decode tok/s on the real code tasks):

| Build | Average |
|---|---:|
| experts only, no DSpark | 37.7 |
| experts only, 15 tokens | 48.0 |
| attention, no DSpark | 65.4 |
| attention, 7 tokens | 69.9 |
| attention, 4 tokens | 75.1 |

4 tokens was best (7 and 15 were worse), so `serve.sh` uses 4. Mean acceptance length was about 2.4 to 3.2 tokens on short prompts and about 1.6 to 1.8 on the 16k filler prompt. The gain is real below about 16k tokens of context (+32% at 4k for the final build), break-even at about 16k and a loss beyond (-23% at 32k).

### Quality (same harness for every build)

| Build | MMLU (1,425) | GSM8K (25) | IFEval (25) | HumanEval (40) | Long-ctx code completed (18) | Large-digit vision (7) | Tool calls (3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B NVFP4 (older vLLM 0.26 build) | 84.6 | 56 | 88 | 100 | 18/18 done | - | - |
| MiniCPM-V 4.7 BF16 | 84.1 | 44 | 92 | 100 | 18/18 done | 7/7 | 3/3 |
| MiniCPM-V 4.7 AWQ (OpenBMB) | 83.0 | 48 | 88 | 100 | 18/18 done | 7/7 | 3/3 |
| NVFP4A16, experts only | 83.9 | 48 | 92 | 100 | 18/18 done | 7/7 | 3/3 |
| NVFP4A16, attention | 83.1 | 44 | 92 | 100 | 18/18 done | 7/7 | 3/3 |
| **NVFP4A16, attention + lm_head** | **83.1** | **48** | **92** | **100** | **18/18 done** | **7/7** | **3/3** |

Quantizing attention and `lm_head` costs about 1 MMLU point against BF16 (83.1 vs 84.1) and 0.8 against the experts-only build (83.9). The attention build already scored 83.1, so adding `lm_head` did not lower MMLU further. GSM8K and IFEval use 25 items each: one item is 4 points, so differences of 4 to 8 points are noise. MMLU (1,425 items) is the precise quality signal. HumanEval, vision and tool calls are unchanged from BF16, but they are small checks (n = 40, 7, 3), not benchmarks. The long-context code column counts completed tasks (a speed set run with `--judge none`); it is not a correctness score.

## Why this is fast

Single-request decode is limited by how many weight bytes are read per token. Measured per token:

- Experts-only build: 4.26 GB (BF16 `linear_attn` 2.02 GB, BF16 `self_attn` 0.55 GB, BF16 `lm_head` 1.02 GB, plus 4-bit experts and small layers).
- Attention build: 2.43 GB.
- Attention + lm_head build: about 1.70 GB. For comparison `nvidia/Qwen3.6-35B-A3B-NVFP4` reads 2.37 GB (FP8 attention, NVFP4 `lm_head`).

The engine is not the cause of the old slowness: Qwen3.6 NVFP4 on the same nightly, box, flags and Marlin MoE kernel decodes at 77 to 79 tok/s. The `lm_head` (vocabulary 248,144 x 2048) was disproportionately expensive in BF16: quantizing it alone took decode from 70 to 101 tok/s.

## Gotchas and troubleshooting

- **Quantized `lm_head` fails to load** with `There is no module or parameter named 'lm_head.weight_global_scale' in Qwen3_5ForCausalLM`. vLLM loads a compressed-tensors quantized `lm_head` only if a config group target matches it by regex. llm-compressor writes only `["Linear"]`, and a plain `"lm_head"` target does not work. The `targets` list must include `"re:.*lm_head$"`. `quantize_nvfp4a16.py --quantize-lm-head` adds it. If you have a checkpoint without it, add the string to `config_groups.*.targets` in `config.json`.
- **Fused partitions need one shared `weight_global_scale`** (q/k/v; `in_proj_qkv` + `in_proj_z`; expert gate/up). If they differ, vLLM takes the max and only logs a warning, which silently mis-dequantizes the others. The quantizer groups them and `verify_ckpt.py` checks it, so keep running the verify step.
- **Vision tower ignore list.** vLLM renames `model.vision_tower.*` to `vpm.*`, and regex ignores are not remapped, so `quantization_config.ignore` must include `re:.*vpm.*` and `re:.*vit_merger.*` as well as the checkpoint names. Otherwise vLLM tries to load the BF16 vision encoder as NVFP4 and the engine fails to start (`Engine core initialization failed`; OpenBMB's AWQ loads fine). You can add the two patterns to an existing checkpoint's `config.json` without re-quantizing. The `quantize_nvfp4a16.py` in this recipe already includes them.
- **Quantizing on the GPU next to a running vLLM server fails** llm-compressor's capacity check. Use the default CPU run (`DEVICE=cpu`, `MAX_WORKERS=2`, `GPU_ARGS=""`): about 10 minutes for the 66 GB BF16 model on the Grace CPU, no GPU needed, so the server can keep running.
- **`contains custom code which must be executed`** (vLLM startup). `processor_config.json`, `preprocessor_config.json` and `tokenizer_config.json` carry `auto_map` entries pointing to `processing_minicpmv4_7.py` and similar files that are not in the HF repos. `--trust-remote-code` does not help because the files do not exist. Run `fix_processor_configs.py` (step 4). Applies to both OpenBMB's AWQ repo and the quantized output.
- **`Unrecognized image processor`**. transformers registers `minicpmv4_7` with the 4.6 classes, but the configs name `MiniCPMV4_7ImageProcessor` / `MiniCPMV4_7VideoProcessor`. Same script fixes it.
- **Model architecture not recognized / not supported.** Your vLLM is older than PR #58674 (merged 2026-10-11; not in v0.31.0). Pull a newer nightly.
- **Output contains literal `\n` instead of line breaks.** The tokenizer decodes newlines that way. Use `--reasoning-parser minicpmv`, which repairs it.
- **Thinking is on by default**, so short answers still spend tokens on `<think>`. Disable per request with `"chat_template_kwargs": {"enable_thinking": false}`, and size `max_tokens` accordingly when it is on.
- **Do not pass `--moe-backend`.** Let vLLM pick (Marlin for W4A16 MoE here). Forcing CUTLASS NVFP4 MoE on sm_121 has produced corrupt output.
- **If you rebuild the checkpoint with FP8 layers** (e.g. mimicking a modelopt mixed-precision recipe): we observed FP8 linear layers going through a cuDNN FP8 GEMM that can deadlock vLLM on GB10 under long soaks. That is why this recipe uses NVFP4 W4A16 or BF16 only.
- **Quantizer refuses to run** with `already contains safetensors`: delete the output directory first. The preflight also stops if the source has `mtp.*` tensors or no fused 3D experts, since the script assumes this checkpoint layout.
- **llm-compressor import fails** in the image: `run_quant.sh` falls back to installing `transformers==5.18.0`. If you install `llmcompressor` without `--no-deps` it may downgrade the image's torch/transformers.
- **Out of memory at load.** GB10 uses unified memory shared with the OS and other containers. Lower `GPU_UTIL`, `MAX_LEN` or `MAX_SEQS`, and stop other GPU workloads.

## Credits

- OpenBMB for MiniCPM-V 4.7 and the BF16 and AWQ checkpoints: https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B
- vLLM, in particular PR #58674 (MiniCPM-V 4.7 support): https://github.com/vllm-project/vllm/pull/58674
- llm-compressor and compressed-tensors (`model_free_ptq`, NVFP4A16 scheme): https://github.com/vllm-project/llm-compressor
- NVIDIA's `nvidia/Qwen3.6-35B-A3B-NVFP4` for the reference on-disk expert layout and the comparison point.
