# ICTC VLM Clustering — GPU Cluster Script

Production-ready script implementing **Image Clustering Conditioned on Text Criteria (ICTC)** for large-scale image clustering on multi-GPU clusters.

Designed to run on **2× A100 GPUs** with **200 k+ images** over multiple days, with full checkpoint/resume support and SSH resilience.

---

## Pipeline

```
Step 0  Discovery   Scan dataset, build image map
Step 1  VLM         Qwen3-VL captions each image → {category, brand, text, summary}
Step 2a LLM         Extract 2-4 word "marketing hook" per ad
Step 2b LLM         Synthesize top hooks → K cluster definitions  (single call)
Step 3  LLM         Assign each ad to its best-fitting cluster
```

Default models: **Qwen3-VL-30B** (VLM) + **Llama-3.1-8B** (LLM)
Any vLLM-supported model works — swap via `--vlm_model` / `--llm_model`.

---

## Quick Start

```bash
# 1. Clone
git clone git@github.com:GrigoriusPompeus/summer_research_distilled_script_for_gpu.git
cd summer_research_distilled_script_for_gpu

# 2. Install dependencies (on the GPU cluster)
pip install -r requirements_cluster.txt
# Optional but recommended on A100/H100:
pip install flash-attn --no-build-isolation

# 3. Smoke test (100 images, verify GPU + model load)
python ictc_cluster.py \
  --ads_dir  /data/dataset/ads \
  --output_dir /data/results/test \
  --max_images 100 --verbose

# 4. Full run via tmux (edit ADS_DIR / OUTPUT_DIR inside run_ictc.sh first)
chmod +x run_ictc.sh
./run_ictc.sh
```

---

## Input Data Format

The script supports two layouts:

**A. Subdirectory format** (same as your existing dataset):
```
ads/
├── <uuid-1>/
│   ├── full_data.json    # {"observation_id": "...", "observation": {"platform": "...", ...}}
│   └── image.jpg
├── <uuid-2>/
│   └── ...
```

**B. Flat directory of images** (for custom datasets):
```
images/
├── ad_001.jpg
├── ad_002.jpg
└── ...
```

---

## Output Files

All outputs go to `--output_dir`:

| File | Contents |
|------|----------|
| `image_mapping.json` | Step 0: obs_id → image path + metadata |
| `step1_captions.json` | Step 1: VLM captions for valid ads |
| `ui_only_images.json` | Step 1: images classified as UI/interface screens |
| `broken_images.json` | Step 1: broken/unreadable images |
| `step2a_hooks.json` | Step 2a: 2-4 word marketing hook per ad |
| `step2b_dynamic_clusters.json` | Step 2b: K cluster definitions |
| `step3_final_assignment.json` | Step 3: cluster assignment per ad |
| `ictc_final_results.json` | Combined export (all steps, all fields) |
| `categorized_images/` | Images sorted into valid_ads / ui_only / broken |
| `logs/ictc_*.log` | Rotating log files (one per run) |

---

## CLI Reference

### Paths
| Flag | Default | Description |
|------|---------|-------------|
| `--ads_dir` | *(required)* | Root of ad dataset |
| `--output_dir` | *(required)* | Results + checkpoint directory |

### Model Selection
| Flag | Default | Description |
|------|---------|-------------|
| `--vlm_model` | `Qwen/Qwen3-VL-30B-Instruct` | VLM for image captioning (Step 1) |
| `--llm_model` | `meta-llama/Llama-3.1-8B-Instruct` | LLM for clustering steps (2a/2b/3) |

### GPU Configuration
| Flag | Default | Description |
|------|---------|-------------|
| `--num_gpus` | `2` | Total GPUs — sets tensor-parallel for both models |
| `--vlm_tp` | *(from num_gpus)* | Tensor-parallel degree for VLM only |
| `--llm_tp` | `1` | Tensor-parallel degree for LLM only |
| `--gpu_ids` | *(all visible)* | CUDA device IDs, e.g. `"0,1"` or `"2,3"` |
| `--gpu_util` | `0.90` | vLLM memory utilization per GPU (0.0–1.0) |

### Precision & Quantization
| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `bfloat16` | Weight dtype: `bfloat16` / `float16` / `float32` / `auto` |
| `--quantization` | *(none)* | `awq` / `gptq` / `fp8` — applies to both models |
| `--vlm_quantization` | *(none)* | Override quantization for VLM only |
| `--llm_quantization` | *(none)* | Override quantization for LLM only |

### Context & Image Resolution
| Flag | Default | Description |
|------|---------|-------------|
| `--vlm_max_model_len` | `4096` | VLM context window in tokens |
| `--llm_max_model_len` | `4096` | LLM context window in tokens |
| `--max_image_tokens` | `1280` | Image token budget (higher = sharper, more VRAM) |

### vLLM Engine Tuning
| Flag | Default | Description |
|------|---------|-------------|
| `--enforce_eager` | off | Disable CUDA graphs (saves VRAM, use when OOM on load) |
| `--swap_space` | `4` | CPU swap per GPU in GB (increase for large models) |

### Batching
| Flag | Default | Description |
|------|---------|-------------|
| `--batch_vlm` | `8` | Images per VLM call (reduce to 2–4 if OOM) |
| `--batch_llm` | `128` | Prompts per LLM call (can be 256–512 for 8B models) |

### Checkpointing
| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt_vlm` | `1000` | Save Step 1 checkpoint every N images |
| `--ckpt_llm` | `5000` | Save Step 2a/3 checkpoint every N items |

### ICTC / Clustering
| Flag | Default | Description |
|------|---------|-------------|
| `--num_clusters` | `5` | Number of output clusters (K) |
| `--top_hooks` | `60` | Top-N hooks fed into Step 2b synthesis |
| `--criterion` | `Marketing Strategy` | What dimension to cluster by |

### Pipeline Control
| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | `1,2a,2b,3` | Which steps to run (discovery always runs) |
| `--max_images` | *(all)* | Cap images processed — for smoke testing |
| `--seed` | `42` | Random seed for reproducibility |
| `--verbose` | off | Enable DEBUG logging |

---

## GPU Memory Guide

| Model | VRAM (bf16) | Config |
|-------|------------|--------|
| Qwen3-VL-7B | ~14 GB | `--vlm_tp 1` (single A100-40GB) |
| Qwen3-VL-30B | ~60 GB | `--vlm_tp 2` (two A100-80GB) ← **default** |
| Qwen3-VL-72B | ~144 GB | `--vlm_tp 4` or use AWQ |
| Qwen3-VL-72B-AWQ | ~57 GB | `--vlm_tp 2 --quantization awq` |
| Qwen2.5-VL-7B | ~14 GB | `--vlm_tp 1` |
| Qwen2.5-VL-32B | ~64 GB | `--vlm_tp 2` |
| Llama-3.1-8B | ~16 GB | `--llm_tp 1` |
| Llama-3.1-70B | ~140 GB | `--llm_tp 2` or AWQ |

---

## Common Recipes

```bash
# ── 2x A100-80GB, default Qwen3-VL-30B + Llama-3.1-8B ──────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out

# ── 4-GPU node (auto tensor-parallel) ───────────────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --num_gpus 4

# ── Fit Qwen3-VL-72B on 2x A100-80GB using AWQ quantization ─────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --vlm_model Qwen/Qwen3-VL-72B-Instruct-AWQ \
  --vlm_tp 2 --vlm_quantization awq

# ── Resume after crash (re-run same command) ─────────────────────────────────
# Checkpoints are automatic — just re-run and it picks up where it left off.
python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out

# ── Skip VLM (Step 1 already done, only re-run clustering steps) ─────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --steps 2a,2b,3

# ── Cluster by a different criterion (8 visual style clusters) ───────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --criterion "Visual Design Style" --num_clusters 8

# ── Use only GPUs 2 and 3 on a shared node ───────────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --gpu_ids "2,3" --vlm_tp 2

# ── Old V100 GPUs (no bfloat16 support) ──────────────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --dtype float16
```

---

## Resume & Crash Recovery

Every step saves an atomic checkpoint (write-to-tmp then rename — never corrupts):

- **Step 1** checkpoints every `--ckpt_vlm` images (default 1 000)
- **Steps 2a/3** checkpoint every `--ckpt_llm` items (default 5 000)

To resume after any failure: **just re-run the same command.** Each step detects existing output and skips already-processed items.

---

## SSH-Resilient Execution

Use `run_ictc.sh` which wraps everything in a **tmux session**:

```bash
./run_ictc.sh              # starts or attaches to session "ictc"

# While disconnected, job keeps running.
# Re-connect and check:
tmux attach -t ictc
tail -f /data/out/logs/ictc_*.log
```

---

## Background: ICTC

Based on the paper *"Image Clustering Conditioned on Text Criteria"*.
This implementation uses fully local open-source models (no paid APIs) and extends the original approach with:
- Multi-GPU tensor parallelism via vLLM
- Batch inference for 200k+ scale
- Configurable clustering criterion (not limited to marketing strategy)
- Robust checkpoint/resume for multi-day runs
