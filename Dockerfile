# vLLM engine + quantization tools + recipe scripts in one image, so the recipe keeps
# working after the nightly it was tested on is gone.
#   serving:      vLLM exactly as in the base image (system python untouched)
#   quantization: /opt/quant venv layered on top (llm-compressor 0.14.0, compressed-tensors 0.19.0)
FROM vllm/vllm-openai@sha256:d044fa021560985593390653000281ee57e0c8bcd5e8f3b94710ec335c0544a3

LABEL org.opencontainers.image.source="https://github.com/jkuepker/minicpm-v47-nvfp4a16-dgx-spark" \
      org.opencontainers.image.description="vLLM 0.31.1rc1.dev296 (aarch64, CUDA 13) + llm-compressor 0.14.0 + MiniCPM-V 4.7 NVFP4A16 recipe scripts for one DGX Spark" \
      org.opencontainers.image.licenses="Apache-2.0"

# The image's vLLM needs compressed-tensors 0.17.0; the quantizer needs 0.19.0. Keep them apart:
# a venv that sees the system packages (torch, transformers, safetensors) but overrides these two.
# pip (not uv): it treats the image's local torch build (+cu130) as already satisfying the pins.
RUN set -eux; \
    python3 -m venv --system-site-packages /opt/quant; \
    python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|numpy|safetensors|huggingface[-_]hub|tokenizers|accelerate)==' > /tmp/constraints.txt; \
    /opt/quant/bin/python -m pip install -q --no-cache-dir --no-deps "llmcompressor==0.14.0" "compressed-tensors==0.19.0"; \
    /opt/quant/bin/python -m pip install -q --no-cache-dir -c /tmp/constraints.txt loguru pydantic psutil requests pyyaml tqdm pillow datasets accelerate nvidia-ml-py; \
    /opt/quant/bin/python -c 'from llmcompressor import model_free_ptq; import compressed_tensors as c; assert c.__version__.startswith("0.19"), c.__version__'; \
    python3 -c 'import compressed_tensors as c; assert c.__version__.startswith("0.17"), c.__version__'; \
    rm -f /tmp/constraints.txt

COPY quantize_nvfp4a16.py verify_ckpt.py fix_processor_configs.py run_quant.sh serve.sh /recipe/
