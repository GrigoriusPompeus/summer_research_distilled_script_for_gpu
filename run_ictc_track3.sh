#!/usr/bin/env bash
# =============================================================================
# run_ictc_track3.sh  —  Launch ICTC Track 3: Cultural Representation & Social Values
#
# Uses the SAME images as Tracks 1 & 2 but with different prompts that
# analyse cultural content, representation, and social values in ads.
#
# Usage:
#   chmod +x run_ictc_track3.sh
#   ./run_ictc_track3.sh                          # full run with defaults
#   ./run_ictc_track3.sh --end_index 100          # smoke test (100 ads)
#   ./run_ictc_track3.sh --steps 2a,2b,3          # skip VLM, resume from captions
#
# The script:
#   1. Creates (or attaches to) a tmux session named "ictc_t3" so the job survives
#      SSH disconnections.
#   2. Activates the venv environment.
#   3. Runs ictc_cluster_single_gpu_track3.py — all output also goes to OUTPUT_DIR/logs/.
#   4. Check progress any time:
#        tmux attach -t ictc_t3
#        tail -f <OUTPUT_DIR>/track3_cultural/logs/ictc_*.log
# =============================================================================

set -euo pipefail

# ─── REQUIRED: set these for your server ────────────────────────────────────
ADS_DIR="/mnt/nvme0n1/data/output/exported-data/ads"
OUTPUT_DIR="/mnt/nvme0n1/data/output/ictc_production"
# ─────────────────────────────────────────────────────────────────────────────

# ─── MODEL SELECTION ─────────────────────────────────────────────────────────
VLM_MODEL="Qwen/Qwen3.5-9B"
# ─────────────────────────────────────────────────────────────────────────────

# ─── GPU CONFIGURATION ───────────────────────────────────────────────────────
GPU_UTIL=0.90
DTYPE="bfloat16"
QUANTIZATION=""       # leave blank for none
# ─────────────────────────────────────────────────────────────────────────────

# ─── CLUSTERING PARAMETERS ───────────────────────────────────────────────────
NUM_CLUSTERS=5
CRITERION="Cultural Representation & Social Values"
# ─────────────────────────────────────────────────────────────────────────────

# ─── PERFORMANCE TUNING ──────────────────────────────────────────────────────
BATCH_VLM=4           # images per VLM batch (A100-40GB safe with 9B model)
BATCH_LLM=64          # prompts per text batch
CKPT_VLM=500          # save Step 1 checkpoint every N images
VLM_MAX_MODEL_LEN=8192
MAX_IMAGE_TOKENS=1280
SEED=42
# ─────────────────────────────────────────────────────────────────────────────

# ─── PIPELINE CONTROL ────────────────────────────────────────────────────────
STEPS="1,2a,2b,3"
SHARD_ID="track3_cultural"
# ─────────────────────────────────────────────────────────────────────────────

# ─── PYTHON ENVIRONMENT ──────────────────────────────────────────────────────
VENV_PATH="$HOME/ictc_env"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="ictc_t3"

# Build argument string
build_args() {
    local args=""
    args+=" --ads_dir ${ADS_DIR}"
    args+=" --output_dir ${OUTPUT_DIR}"
    args+=" --vlm_model ${VLM_MODEL}"
    args+=" --gpu_util ${GPU_UTIL}"
    args+=" --dtype ${DTYPE}"
    [[ -n "${QUANTIZATION}" ]] && args+=" --quantization ${QUANTIZATION}"
    args+=" --num_clusters ${NUM_CLUSTERS}"
    args+=" --criterion \"${CRITERION}\""
    args+=" --batch_vlm ${BATCH_VLM}"
    args+=" --batch_llm ${BATCH_LLM}"
    args+=" --ckpt_vlm ${CKPT_VLM}"
    args+=" --max_model_len ${VLM_MAX_MODEL_LEN}"
    args+=" --max_image_tokens ${MAX_IMAGE_TOKENS}"
    args+=" --seed ${SEED}"
    args+=" --steps ${STEPS}"
    args+=" --shard_id ${SHARD_ID}"
    args+=" --verbose"
    # Forward any extra args passed to this script (e.g. --end_index 100)
    args+=" $*"
    echo "${args}"
}

ENV_CMD="source ${VENV_PATH}/bin/activate && HF_HOME=/mnt/nvme0n1/cache/huggingface"
PYTHON_CMD="${ENV_CMD} python3 ${SCRIPT_DIR}/ictc_cluster_single_gpu_track3.py $(build_args "$@")"

# ── Print config summary ──────────────────────────────────────────────────────
echo "======================================================================"
echo "  ICTC Track 3: Cultural Representation & Social Values"
echo "  ads_dir      : ${ADS_DIR}"
echo "  output_dir   : ${OUTPUT_DIR}/${SHARD_ID}"
echo "  vlm_model    : ${VLM_MODEL}"
echo "  criterion    : ${CRITERION}"
echo "  num_clusters : ${NUM_CLUSTERS}"
echo "  steps        : ${STEPS}"
echo "  shard_id     : ${SHARD_ID}"
echo "  tmux session : ${SESSION}"
echo "======================================================================"

# ── Launch via tmux (SSH-resilient) ───────────────────────────────────────────
if command -v tmux &>/dev/null; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo ""
        echo "Session '${SESSION}' already exists."
        echo "  To attach: tmux attach -t ${SESSION}"
        echo "  To kill and restart: tmux kill-session -t ${SESSION} && ./run_ictc_track3.sh"
        echo "  To tail logs: tail -f ${OUTPUT_DIR}/${SHARD_ID}/logs/ictc_*.log"
        tmux attach -t "${SESSION}"
    else
        echo ""
        echo "Starting tmux session '${SESSION}'"
        echo "  Detach (keep running): Ctrl+B then D"
        echo "  Re-attach:             tmux attach -t ${SESSION}"
        echo "  Follow logs:           tail -f ${OUTPUT_DIR}/${SHARD_ID}/logs/ictc_*.log"
        echo ""
        tmux new-session -d -s "${SESSION}" -x 220 -y 50
        tmux send-keys -t "${SESSION}" "${PYTHON_CMD}" Enter
        tmux attach -t "${SESSION}"
    fi
else
    mkdir -p "${OUTPUT_DIR}/${SHARD_ID}/logs"
    LOG_FILE="${OUTPUT_DIR}/${SHARD_ID}/logs/nohup_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "tmux not available — launching with nohup"
    echo "  Log file : ${LOG_FILE}"
    echo "  Monitor  : tail -f ${LOG_FILE}"
    nohup bash -c "${PYTHON_CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "  PID      : ${PID}"
    echo "  Kill     : kill ${PID}"
fi
