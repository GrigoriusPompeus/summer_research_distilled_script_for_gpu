#!/usr/bin/env bash
# =============================================================================
# run_ictc.sh  —  Launch ICTC pipeline on a GPU cluster node
#
# Usage:
#   chmod +x run_ictc.sh
#   ./run_ictc.sh                                   # full run with defaults
#   ./run_ictc.sh --max_images 100                  # smoke test (100 images)
#   ./run_ictc.sh --steps 2a,2b,3                   # skip VLM, resume from captions
#   ./run_ictc.sh --num_gpus 4                      # 4-GPU node
#   ./run_ictc.sh --quantization awq                # use AWQ quantized models
#
# The script:
#   1. Creates (or attaches to) a tmux session named "ictc" so the job survives
#      SSH disconnections.
#   2. Activates the conda/venv environment if configured below.
#   3. Runs ictc_cluster.py — all output is also written to OUTPUT_DIR/logs/.
#   4. Check progress any time:
#        tmux attach -t ictc
#        tail -f <OUTPUT_DIR>/logs/ictc_*.log
# =============================================================================

set -euo pipefail

# ─── REQUIRED: set these for your cluster ────────────────────────────────────
ADS_DIR="/data/dataset/ads"           # path to your ads directory on the cluster
OUTPUT_DIR="/data/results/ictc_run1"  # where checkpoints and results go
# ─────────────────────────────────────────────────────────────────────────────

# ─── MODEL SELECTION ─────────────────────────────────────────────────────────
# VLM for image captioning (Step 1)
#   Options: Qwen/Qwen3-VL-7B-Instruct      (fast, ~14 GB)
#            Qwen/Qwen3-VL-30B-Instruct     (balanced, ~60 GB on 2x A100-80GB)  <- default
#            Qwen/Qwen3-VL-72B-Instruct     (best, ~144 GB — needs 4x A100 or AWQ)
#            Qwen/Qwen3-VL-72B-Instruct-AWQ (best + 4-bit, ~57 GB on 2x A100-80GB)
#            Qwen/Qwen2.5-VL-7B-Instruct    (if Qwen3 not available)
#            Qwen/Qwen2.5-VL-32B-Instruct   (Qwen2.5 high quality)
VLM_MODEL="Qwen/Qwen3-VL-30B-Instruct"

# LLM for clustering steps (Steps 2a/2b/3)
#   Options: meta-llama/Llama-3.1-8B-Instruct   (fast, ~16 GB)   <- default
#            meta-llama/Llama-3.1-70B-Instruct  (higher quality, ~140 GB)
#            meta-llama/Llama-3.3-70B-Instruct  (latest 70B)
#            Qwen/Qwen3-8B-Instruct             (alternative fast LLM)
#            mistralai/Mistral-7B-Instruct-v0.3 (alternative)
LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
# ─────────────────────────────────────────────────────────────────────────────

# ─── GPU CONFIGURATION ───────────────────────────────────────────────────────
# Number of GPUs to use (sets tensor-parallel degree for both VLM and LLM).
# Override individually with VLM_TP / LLM_TP below.
NUM_GPUS=2

# Fine-grained control (leave blank to use NUM_GPUS for both):
VLM_TP=""   # e.g. "2" — tensor parallel for VLM  (overrides NUM_GPUS)
LLM_TP=""   # e.g. "1" — tensor parallel for LLM  (overrides NUM_GPUS)

# Restrict to specific GPU device IDs (leave blank to use all visible GPUs):
#   GPU_IDS="0,1"      # use GPUs 0 and 1 only
#   GPU_IDS="2,3"      # use GPUs 2 and 3 (useful on shared nodes)
GPU_IDS=""

GPU_UTIL=0.90         # vLLM memory utilization per GPU (lower if you see OOM on load)
ENFORCE_EAGER=""      # set to "--enforce_eager" to disable CUDA graph (saves VRAM)
SWAP_SPACE=4          # CPU swap space per GPU in GB (increase for large models)
# ─────────────────────────────────────────────────────────────────────────────

# ─── PRECISION / QUANTIZATION ────────────────────────────────────────────────
DTYPE="bfloat16"      # bfloat16 (best on A100/H100) | float16 (for V100/T4) | auto
QUANTIZATION=""       # leave blank for none, or: awq | gptq | fp8
#   If you use an AWQ model above, set QUANTIZATION="awq"
#   Example: VLM_MODEL="Qwen/Qwen3-VL-72B-Instruct-AWQ"  QUANTIZATION="awq"
# ─────────────────────────────────────────────────────────────────────────────

# ─── CLUSTERING PARAMETERS ───────────────────────────────────────────────────
NUM_CLUSTERS=5        # K — number of output clusters
TOP_HOOKS=60          # top-N hooks fed to Step 2b cluster synthesis
CRITERION="Marketing Strategy"  # what to cluster by — change freely:
#   "Marketing Strategy" | "Visual Design Style" | "Target Audience"
#   "Emotional Tone" | "Call-to-Action Type" | "Product Category"
# ─────────────────────────────────────────────────────────────────────────────

# ─── PERFORMANCE TUNING ──────────────────────────────────────────────────────
BATCH_VLM=8           # images per VLM batch; reduce to 2-4 if OOM during Step 1
BATCH_LLM=128         # prompts per LLM batch; can be 256-512 for 8B models
CKPT_VLM=1000         # save Step 1 checkpoint every N images
CKPT_LLM=5000         # save Steps 2a/3 checkpoint every N items
VLM_MAX_MODEL_LEN=4096
LLM_MAX_MODEL_LEN=4096
MAX_IMAGE_TOKENS=1280 # higher = better quality on text-heavy ads, more VRAM
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# ─── PIPELINE CONTROL ────────────────────────────────────────────────────────
STEPS="1,2a,2b,3"     # all steps; change to "2a,2b,3" to skip VLM captioning
MAX_IMAGES=""         # leave blank for all; set e.g. "100" for smoke test
# ─────────────────────────────────────────────────────────────────────────────

# ─── PYTHON ENVIRONMENT ──────────────────────────────────────────────────────
# Uncomment ONE of the following to activate your environment:
# CONDA_ENV="ictc"                        # conda activate $CONDA_ENV
# VENV_PATH="/home/user/ictc/.venv"       # source $VENV_PATH/bin/activate
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="ictc"

# Build argument string
build_args() {
    local args=""
    args+=" --ads_dir ${ADS_DIR}"
    args+=" --output_dir ${OUTPUT_DIR}"
    args+=" --vlm_model ${VLM_MODEL}"
    args+=" --llm_model ${LLM_MODEL}"
    args+=" --num_gpus ${NUM_GPUS}"
    [[ -n "${VLM_TP}" ]]       && args+=" --vlm_tp ${VLM_TP}"
    [[ -n "${LLM_TP}" ]]       && args+=" --llm_tp ${LLM_TP}"
    [[ -n "${GPU_IDS}" ]]      && args+=" --gpu_ids ${GPU_IDS}"
    args+=" --gpu_util ${GPU_UTIL}"
    args+=" --dtype ${DTYPE}"
    [[ -n "${QUANTIZATION}" ]] && args+=" --quantization ${QUANTIZATION}"
    [[ -n "${ENFORCE_EAGER}" ]] && args+=" ${ENFORCE_EAGER}"
    args+=" --swap_space ${SWAP_SPACE}"
    args+=" --num_clusters ${NUM_CLUSTERS}"
    args+=" --top_hooks ${TOP_HOOKS}"
    args+=" --criterion \"${CRITERION}\""
    args+=" --batch_vlm ${BATCH_VLM}"
    args+=" --batch_llm ${BATCH_LLM}"
    args+=" --ckpt_vlm ${CKPT_VLM}"
    args+=" --ckpt_llm ${CKPT_LLM}"
    args+=" --vlm_max_model_len ${VLM_MAX_MODEL_LEN}"
    args+=" --llm_max_model_len ${LLM_MAX_MODEL_LEN}"
    args+=" --max_image_tokens ${MAX_IMAGE_TOKENS}"
    args+=" --seed ${SEED}"
    args+=" --steps ${STEPS}"
    [[ -n "${MAX_IMAGES}" ]]   && args+=" --max_images ${MAX_IMAGES}"
    # Forward any extra args passed to this script
    args+=" $*"
    echo "${args}"
}

# Environment activation command
ENV_CMD=""
if [[ -n "${CONDA_ENV:-}" ]]; then
    ENV_CMD="conda activate ${CONDA_ENV} && "
elif [[ -n "${VENV_PATH:-}" ]]; then
    ENV_CMD="source ${VENV_PATH}/bin/activate && "
fi

PYTHON_CMD="${ENV_CMD}python ${SCRIPT_DIR}/ictc_cluster.py $(build_args "$@")"

# ── Print config summary ──────────────────────────────────────────────────────
echo "======================================================================"
echo "  ICTC Production Run"
echo "  ads_dir      : ${ADS_DIR}"
echo "  output_dir   : ${OUTPUT_DIR}"
echo "  vlm_model    : ${VLM_MODEL}  (num_gpus=${NUM_GPUS})"
echo "  llm_model    : ${LLM_MODEL}"
echo "  criterion    : ${CRITERION}"
echo "  num_clusters : ${NUM_CLUSTERS}"
echo "  steps        : ${STEPS}"
[[ -n "${MAX_IMAGES}" ]] && echo "  max_images   : ${MAX_IMAGES}  (TEST MODE)"
echo "======================================================================"

# ── Launch via tmux (SSH-resilient) or nohup fallback ─────────────────────────
if command -v tmux &>/dev/null; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo ""
        echo "Session '${SESSION}' already exists."
        echo "  To attach: tmux attach -t ${SESSION}"
        echo "  To kill and restart: tmux kill-session -t ${SESSION} && ./run_ictc.sh"
        echo "  To tail logs: tail -f ${OUTPUT_DIR}/logs/ictc_*.log"
        tmux attach -t "${SESSION}"
    else
        echo ""
        echo "Starting tmux session '${SESSION}'"
        echo "  Detach (keep running): Ctrl+B then D"
        echo "  Re-attach:             tmux attach -t ${SESSION}"
        echo "  Follow logs:           tail -f ${OUTPUT_DIR}/logs/ictc_*.log"
        echo ""
        tmux new-session -d -s "${SESSION}" -x 220 -y 50
        tmux send-keys -t "${SESSION}" "${PYTHON_CMD}" Enter
        tmux attach -t "${SESSION}"
    fi
else
    mkdir -p "${OUTPUT_DIR}/logs"
    LOG_FILE="${OUTPUT_DIR}/logs/nohup_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "tmux not available — launching with nohup"
    echo "  Log file : ${LOG_FILE}"
    echo "  Monitor  : tail -f ${LOG_FILE}"
    # shellcheck disable=SC2094
    nohup bash -c "${PYTHON_CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "  PID      : ${PID}"
    echo "  Kill     : kill ${PID}"
fi
