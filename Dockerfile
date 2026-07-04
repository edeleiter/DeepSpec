# DeepSpec training image (model-agnostic).
#
# Bakes the pinned Python deps into the base image so no run reinstalls them.
# The repo itself is NOT copied in — it is bind-mounted at /workspace at runtime,
# so host code edits stay live. Only requirements.txt is copied, so the (large)
# dependency layer is cached and rebuilds only when requirements.txt changes.
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# Install the CUDA 12.8 build of the pinned torch FIRST. The target GPU is an
# RTX 5070 Ti (Blackwell, CUDA capability sm_120); cu126 wheels only ship kernels
# up to sm_90 and fail at the first CUDA op with "no kernel image is available
# for execution on the device". cu128 wheels include sm_120. The base image's
# system CUDA is 12.6, but torch wheels bundle their own CUDA runtime, so the
# cu128 wheel runs fine on top of it.
#
# torchvision MUST match: the base image ships a torchvision built for torch 2.10,
# whose compiled ops (torchvision::nms) fail to register against torch 2.9.1 ->
# and transformers imports torchvision via image_utils even for the text-only
# path of the multimodal qwen3_5 model. torchvision 0.24.1 pairs with torch 2.9.1.
#
# --break-system-packages: the base image's Python is PEP 668 "externally
# managed". We deliberately install into it (not a venv) so deps live in the
# image, outside the bind-mounted /workspace.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 \
    && pip install --break-system-packages -r /tmp/requirements.txt

WORKDIR /workspace

# The repo has no packaging config; `deepspec` is imported by having the repo
# root on sys.path. Scripts under scripts/ (e.g. prepare_target_cache.py) put
# their own dir on sys.path, not the root, so set it explicitly for all entry
# points (pipeline CMD, debug bash, manual runs).
ENV PYTHONPATH=/workspace

# Default: run whichever per-model pipeline $PIPELINE points at (each is fully
# env-parameterized). Pick the model by overriding PIPELINE, e.g.
#   docker run -e PIPELINE=scripts/ornith/run_pipeline.sh ...
# scripts/docker_run.sh wraps this with the standard volume mounts. Override the
# command entirely with `docker run ... bash` to drop into an interactive shell.
CMD ["bash", "-lc", "exec bash \"${PIPELINE:-scripts/qwen3_4b/run_pipeline.sh}\""]
