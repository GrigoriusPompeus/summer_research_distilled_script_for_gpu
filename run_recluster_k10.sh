#!/usr/bin/env bash
# =============================================================================
# run_recluster_k10.sh — Re-cluster all 3 tracks with K=10
#
# Reuses existing VLM captions (Step 1) and hooks (Step 2a) from each track,
# then re-runs Steps 2b and 3 with 10 clusters instead of 5.
#
# Output goes to NEW directories so the original k=5 results are preserved:
#   track1_k10/   track2_k10/   track3_k10/
#
# Usage:
#   chmod +x run_recluster_k10.sh
#   ./run_recluster_k10.sh
#
# The script runs all 3 tracks sequentially in a tmux session "ictc_k10".
# =============================================================================

set -euo pipefail

# ─── PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR="/mnt/nvme0n1/data/output/ictc_production"
TRACK1_SOURCE="${BASE_DIR}/full_run"
TRACK2_SOURCE="${BASE_DIR}/track2_identity"
TRACK3_SOURCE="${BASE_DIR}/track3_cultural"
# ─────────────────────────────────────────────────────────────────────────────

# ─── MODEL ──────────────────────────────────────────────────────────────────
VLM_MODEL="Qwen/Qwen3.5-9B"
GPU_UTIL=0.90
DTYPE="bfloat16"
MAX_MODEL_LEN=4096    # text-only steps need less context than VLM
# ─────────────────────────────────────────────────────────────────────────────

# ─── CLUSTERING ─────────────────────────────────────────────────────────────
K=10
BATCH_LLM=64
TOP_HOOKS=60
# ─────────────────────────────────────────────────────────────────────────────

# ─── ENVIRONMENT ────────────────────────────────────────────────────────────
VENV_PATH="$HOME/ictc_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="ictc_k10"
# ─────────────────────────────────────────────────────────────────────────────

ENV_CMD="source ${VENV_PATH}/bin/activate && export HF_HOME=/mnt/nvme0n1/cache/huggingface"

# Build the command that runs all 3 tracks sequentially
RECLUSTER="${SCRIPT_DIR}/ictc_recluster.py"

CMD="${ENV_CMD}"
CMD+=" && echo '====== Track 1: Marketing Strategy (k=${K}) ======'"
CMD+=" && python3 ${RECLUSTER}"
CMD+="   --source_dir ${TRACK1_SOURCE}"
CMD+="   --output_dir ${BASE_DIR}/track1_k${K}"
CMD+="   --track 1 --num_clusters ${K}"
CMD+="   --vlm_model ${VLM_MODEL} --gpu_util ${GPU_UTIL} --dtype ${DTYPE}"
CMD+="   --max_model_len ${MAX_MODEL_LEN} --batch_llm ${BATCH_LLM} --top_hooks ${TOP_HOOKS}"
CMD+="   --verbose"

CMD+=" && echo '====== Track 2: Algorithmic Identity (k=${K}) ======'"
CMD+=" && python3 ${RECLUSTER}"
CMD+="   --source_dir ${TRACK2_SOURCE}"
CMD+="   --output_dir ${BASE_DIR}/track2_k${K}"
CMD+="   --track 2 --num_clusters ${K}"
CMD+="   --vlm_model ${VLM_MODEL} --gpu_util ${GPU_UTIL} --dtype ${DTYPE}"
CMD+="   --max_model_len ${MAX_MODEL_LEN} --batch_llm ${BATCH_LLM} --top_hooks ${TOP_HOOKS}"
CMD+="   --verbose"

CMD+=" && echo '====== Track 3: Cultural Representation (k=${K}) ======'"
CMD+=" && python3 ${RECLUSTER}"
CMD+="   --source_dir ${TRACK3_SOURCE}"
CMD+="   --output_dir ${BASE_DIR}/track3_k${K}"
CMD+="   --track 3 --num_clusters ${K}"
CMD+="   --vlm_model ${VLM_MODEL} --gpu_util ${GPU_UTIL} --dtype ${DTYPE}"
CMD+="   --max_model_len ${MAX_MODEL_LEN} --batch_llm ${BATCH_LLM} --top_hooks ${TOP_HOOKS}"
CMD+="   --verbose"

CMD+=" && echo '====== ALL 3 TRACKS COMPLETE (k=${K}) ======'"

# ── Print config summary ──────────────────────────────────────────────────────
echo "======================================================================"
echo "  ICTC Re-clustering — All 3 Tracks with K=${K}"
echo "  Track 1 source : ${TRACK1_SOURCE}"
echo "  Track 2 source : ${TRACK2_SOURCE}"
echo "  Track 3 source : ${TRACK3_SOURCE}"
echo "  Output dirs    : track1_k${K}/  track2_k${K}/  track3_k${K}/"
echo "  model          : ${VLM_MODEL}"
echo "  tmux session   : ${SESSION}"
echo "======================================================================"

# ── Launch via tmux ──────────────────────────────────────────────────────────
if command -v tmux &>/dev/null; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo ""
        echo "Session '${SESSION}' already exists."
        echo "  To attach: tmux attach -t ${SESSION}"
        echo "  To kill and restart: tmux kill-session -t ${SESSION} && ./run_recluster_k10.sh"
        tmux attach -t "${SESSION}"
    else
        echo ""
        echo "Starting tmux session '${SESSION}'"
        echo "  Detach (keep running): Ctrl+B then D"
        echo "  Re-attach:             tmux attach -t ${SESSION}"
        echo ""
        tmux new-session -d -s "${SESSION}" -x 220 -y 50
        tmux send-keys -t "${SESSION}" "${CMD}" Enter
        tmux attach -t "${SESSION}"
    fi
else
    mkdir -p "${BASE_DIR}/logs"
    LOG_FILE="${BASE_DIR}/logs/recluster_k${K}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "tmux not available — launching with nohup"
    echo "  Log file : ${LOG_FILE}"
    nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "  PID      : ${PID}"
fi
