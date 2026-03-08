# ICTC VLM Clustering — GPU Scripts

Production-ready scripts implementing **Image Clustering Conditioned on Text Criteria (ICTC)** for large-scale image clustering using a unified **Qwen3.5-27B** model.

Based on **Experiment 3B** findings: a single Qwen 3.5 model handles both vision captioning and text reasoning, producing better clusters than a separate VLM + LLM pipeline (higher coherence, better balance, zero unclassified ads).

---

## Repository Contents

| File | Description |
|------|-------------|
| `ictc_cluster.py` | **Multi-GPU production pipeline** — unified or separate model modes, tensor parallelism, full crash resume |
| `ictc_cluster_single_gpu.py` | **Single-GPU shard script (Track 1)** — Marketing Strategy prompts, horizontal scaling across VMs |
| `ictc_cluster_single_gpu_track2.py` | **Single-GPU shard script (Track 2)** — Algorithmic Identity & Profiling prompts |
| `ictc_cluster_single_gpu_track3.py` | **Single-GPU shard script (Track 3)** — Cultural Representation & Social Values prompts |
| `run_ictc.sh` | tmux wrapper for SSH-resilient execution on cluster nodes |
| `run_ictc_single_gpu.sh` | tmux wrapper for Track 1 single-GPU runs |
| `run_ictc_track2.sh` | tmux wrapper for Track 2 single-GPU runs (session: `ictc_t2`) |
| `run_ictc_track3.sh` | tmux wrapper for Track 3 single-GPU runs (session: `ictc_t3`) |
| `ictc_recluster.py` | **Re-clustering script** — re-run Steps 2a/2b/3 from existing VLM captions with a different K, without overwriting originals |
| `run_recluster_k10.sh` | tmux wrapper to re-cluster all 3 tracks with K=10 (session: `ictc_k10`) |
| `requirements_cluster.txt` | pip dependencies (torch, vLLM, transformers, pillow, tqdm) |

---

## Pipeline

The pipeline has 4 steps, shared across all tracks. Only the **prompts** differ between tracks:

```
Step 0  Discovery   Scan dataset, build image map
Step 1  VLM         Qwen3.5 captions each image → {category, brand, text, summary}
Step 2a Text        Same model extracts a 2-4 word label per ad
Step 2b Text        Same model synthesises top labels → K cluster definitions  (single call)
Step 3  Text        Same model assigns each ad to its best-fitting cluster
```

Default model: **Qwen/Qwen3.5-27B** — natively multimodal (early fusion), handles all steps without model swap.

Legacy mode: pass `--llm_model <model_id>` to use a separate LLM for Steps 2a/2b/3 (multi-GPU script only).

### Prompt Tracks

| Track | Script | Criterion | What Step 1 Extracts | Step 2a Label |
|-------|--------|-----------|----------------------|---------------|
| **Track 1** — Marketing Strategy | `ictc_cluster_single_gpu.py` | `Marketing Strategy` | Generic ad description (brand, text, visual summary) | "marketing hook" (e.g., "scarcity urgency") |
| **Track 2** — Algorithmic Identity | `ictc_cluster_single_gpu_track2.py` | `Algorithmic Identity Profiling` | "Assumed User Identity" — socio-economic status, gender performance, cultural signals, lifestyle the ad assumes the viewer has | "algorithmic persona" (e.g., "hustle-culture tech bro") |
| **Track 3** — Cultural Representation | `ictc_cluster_single_gpu_track3.py` | `Cultural Representation & Social Values` | Cultural content & representation — people shown, settings, lifestyles portrayed, values/aspirations communicated, aesthetic and tone | "cultural theme" (e.g., "aspirational wealth display") |

Track 2 is designed for researching the **"Data Double"** — what kind of demographic buckets the algorithm places individuals into based on the ads they are shown. The Step 2b prompt adopts the role of a *Critical Data Scholar* researching surveillance capitalism and algorithmic identity profiling.

Track 3 is designed for studying how advertising reflects and constructs **cultural narratives and social values**. It enables research on identity, representation, consumer culture, and social norms — what lifestyles and identities ads promote as desirable, what cultural values are embedded, and how ads shape norms around gender, class, health, or success. The Step 2b prompt adopts the role of a *Cultural Studies Researcher*.

All tracks share the same pipeline logic, checkpoint system, and CLI flags. They differ **only in the prompt constants** at the top of each script.

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

## Script 1: `ictc_cluster.py` — Multi-GPU Production Pipeline

Runs the full ICTC pipeline on a multi-GPU cluster. Supports two modes:

- **Unified mode** (default): Single Qwen3.5-27B model stays loaded for all steps — no expensive unload/reload.
- **Separate mode** (legacy): VLM for Step 1, unload, then load a dedicated LLM for Steps 2a/2b/3. Use `--llm_model meta-llama/Llama-3.1-8B-Instruct` to activate.

```bash
# Unified (default) — 2x A100-80GB
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out

# Separate VLM + LLM (legacy)
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --llm_model meta-llama/Llama-3.1-8B-Instruct

# 4-GPU node
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --num_gpus 4
```

---

## Script 2: `ictc_cluster_single_gpu.py` — Single-GPU Sharding

Same pipeline, but designed for **one GPU per VM**. Multiple VMs process different image ranges, then results are merged.

Always uses unified mode (single model for all steps, `tp=1`).

```bash
# VM 1: process images 0–50k
python ictc_cluster_single_gpu.py \
  --ads_dir /data/ads --output_dir /data/out \
  --start_index 0 --end_index 50000 --shard_id shard_0

# VM 2: process images 50k–100k
python ictc_cluster_single_gpu.py \
  --ads_dir /data/ads --output_dir /data/out \
  --start_index 50000 --end_index 100000 --shard_id shard_1

# After all VMs finish — merge on any machine:
python ictc_cluster_single_gpu.py --merge_shards \
  /data/out/shard_0 /data/out/shard_1 \
  --output_dir /data/out/merged
```

Sharding is deterministic: all VMs discover the full image set, sort keys identically, then each takes its `[start_index, end_index)` slice.

---

## Script 3: `ictc_cluster_single_gpu_track2.py` — Track 2: Algorithmic Identity

Same pipeline as Track 1, but with **identity-focused prompts** analyzing who the algorithm thinks is watching each ad.

```bash
# Full run — uses same images as Track 1, separate output directory
python ictc_cluster_single_gpu_track2.py \
  --ads_dir /data/ads --output_dir /data/out \
  --shard_id track2_identity

# Or use the tmux wrapper (creates session "ictc_t2")
chmod +x run_ictc_track2.sh
./run_ictc_track2.sh
```

**Track 2 prompt differences:**

| Stage | Track 1 (Marketing) | Track 2 (Identity) |
|-------|---------------------|---------------------|
| Step 1 | Generic `visual_summary` | Analyses "Assumed User Identity" — socio-economic status, gender, cultural signals |
| Step 2a | Extracts "marketing hook" | Extracts "algorithmic persona" (e.g., "exhausted millennial parent") |
| Step 2b | Expert analyst → marketing strategy clusters | Critical Data Scholar → algorithmic identity clusters |
| Step 3 | Assigns to strategy | Assigns to identity cluster |

**Isolation**: Track 2 results go to `output_dir/track2_identity/`, completely separate from Track 1's `full_run/`. Both can coexist on the same server. The tmux session is named `ictc_t2` (vs `ictc` for Track 1) so both can run simultaneously if GPU memory allows.

---

## Script 4: `ictc_cluster_single_gpu_track3.py` — Track 3: Cultural Representation

Same pipeline as Tracks 1 & 2, but with **culture-focused prompts** analyzing cultural content, representation, and social values embedded in ads.

```bash
# Full run — uses same images as Tracks 1 & 2, separate output directory
python ictc_cluster_single_gpu_track3.py \
  --ads_dir /data/ads --output_dir /data/out \
  --shard_id track3_cultural

# Or use the tmux wrapper (creates session "ictc_t3")
chmod +x run_ictc_track3.sh
./run_ictc_track3.sh
```

**Track 3 prompt differences:**

| Stage | Track 1 (Marketing) | Track 3 (Cultural) |
|-------|---------------------|---------------------|
| Step 1 | Generic `visual_summary` | Cultural content & representation — people, settings, lifestyles, values, aesthetic tone |
| Step 2a | Extracts "marketing hook" | Extracts "cultural theme" (e.g., "aspirational wealth display") |
| Step 2b | Expert analyst → marketing strategy clusters | Cultural Studies Researcher → cultural value categories |
| Step 3 | Assigns to strategy | Assigns to cultural category |

**Isolation**: Track 3 results go to `output_dir/track3_cultural/`, completely separate from Tracks 1 and 2. The tmux session is named `ictc_t3` so all three tracks can run sequentially on the same GPU.

---

## Re-clustering Script: `ictc_recluster.py`

A lightweight script that **re-runs Steps 2a/2b/3 from existing VLM captions** with a different number of clusters (K). This avoids re-running the expensive Step 1 VLM captioning (~24 hours) and writes output to a **new directory** so original results are never overwritten.

Supports all 3 prompt tracks via the `--track` flag.

```bash
# Re-cluster Track 1 with K=10 (reuses captions + hooks from full_run/)
python ictc_recluster.py \
  --source_dir /data/out/full_run \
  --output_dir /data/out/track1_k10 \
  --track 1 --num_clusters 10

# Re-cluster Track 2 with K=10
python ictc_recluster.py \
  --source_dir /data/out/track2_identity \
  --output_dir /data/out/track2_k10 \
  --track 2 --num_clusters 10

# Re-cluster Track 3 with K=10, also re-extract themes from scratch
python ictc_recluster.py \
  --source_dir /data/out/track3_cultural \
  --output_dir /data/out/track3_k10 \
  --track 3 --num_clusters 10 --redo_2a

# Or run all 3 tracks sequentially via tmux wrapper (session: ictc_k10)
chmod +x run_recluster_k10.sh
./run_recluster_k10.sh
```

**How it works:**

1. Reads `step1_captions.json` from the source track directory
2. Copies `step2a_hooks.json` from source (or re-extracts with `--redo_2a`)
3. Runs Step 2b (synthesise K clusters) — single LLM call, takes seconds
4. Runs Step 3 (assign all ads) — batched text inference, takes ~15-30 minutes
5. Exports `ictc_final_results.json` to the new output directory

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source_dir` | *(required)* | Directory with existing `step1_captions.json` |
| `--output_dir` | *(required)* | NEW directory for re-clustered output |
| `--track` | *(required)* | Prompt track: `1` (Marketing), `2` (Identity), `3` (Cultural) |
| `--num_clusters` | `10` | Number of output clusters (K) |
| `--redo_2a` | off | Re-extract Step 2a labels instead of copying from source |
| `--vlm_model` | `Qwen/Qwen3.5-9B` | Model for text generation |
| `--top_hooks` | `60` | Top-N hooks fed into Step 2b synthesis |
| `--batch_llm` | `64` | Prompts per text batch |
| `--max_model_len` | `4096` | Context window (text-only needs less than VLM) |

**Note**: The script only loads the model for text-only inference (no image processing), so it uses less VRAM than the full pipeline. On A100-40GB with Qwen3.5-9B, expect ~15-30 minutes per track for Step 3 assignment.

---

## Input Data Format

The scripts support two layouts:

**A. Subdirectory format** (same as the existing dataset):
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

All outputs go to `--output_dir` (or `--output_dir/<shard_id>/` for single-GPU shards):

| File | Contents |
|------|----------|
| `image_mapping.json` | Step 0: obs_id → image path + metadata |
| `step1_captions.json` | Step 1: VLM captions for valid ads |
| `ui_only_images.json` | Step 1: images classified as UI/interface screens |
| `broken_images.json` | Step 1: broken/unreadable images |
| `step2a_hooks.json` | Step 2a: 2-4 word label per ad (marketing hook or algorithmic persona, depending on track) |
| `step2b_dynamic_clusters.json` | Step 2b: K cluster definitions (strategy clusters or identity clusters) |
| `step3_final_assignment.json` | Step 3: cluster assignment per ad |
| `step2b_metadata.json` | Step 2b cache key (num_clusters + criterion) |
| `step3_metadata.json` | Step 3 cache key (cluster names used for assignment) |
| `ictc_final_results.json` | Combined export (all steps, all fields) |
| `categorized_images/` | Images sorted into valid_ads / ui_only / broken |
| `logs/ictc_*.log` | Rotating log files (one per run) |

---

## CLI Reference — `ictc_cluster.py`

### Paths
| Flag | Default | Description |
|------|---------|-------------|
| `--ads_dir` | *(required)* | Root of ad dataset |
| `--output_dir` | *(required)* | Results + checkpoint directory |

### Model Selection
| Flag | Default | Description |
|------|---------|-------------|
| `--vlm_model` | `Qwen/Qwen3.5-27B` | VLM for image captioning. In unified mode, also handles text steps |
| `--llm_model` | `unified` | `unified` = reuse VLM for all steps. Set to a model ID for separate LLM |

### GPU Configuration
| Flag | Default | Description |
|------|---------|-------------|
| `--num_gpus` | *(auto)* | Total GPUs — sets tensor-parallel for both models |
| `--vlm_tp` | *(from num_gpus or 2)* | Tensor-parallel degree for VLM |
| `--llm_tp` | *(from num_gpus or 1)* | Tensor-parallel degree for LLM (separate mode) |
| `--gpu_ids` | *(all visible)* | CUDA device IDs, e.g. `"0,1"` or `"2,3"` |
| `--gpu_util` | `0.90` | vLLM memory utilisation per GPU (0.0–1.0) |

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
| `--force_steps` | *(none)* | Force re-run these steps even if output exists, e.g. `2b,3` |
| `--max_images` | *(all)* | Cap images processed — for smoke testing |
| `--seed` | `42` | Random seed for reproducibility |
| `--verbose` | off | Enable DEBUG logging |

---

## CLI Reference — `ictc_cluster_single_gpu.py`

Key differences from multi-GPU version:

| Flag | Default | Description |
|------|---------|-------------|
| `--start_index` | `0` | Start index into sorted image list (inclusive) |
| `--end_index` | *(end)* | End index into sorted image list (exclusive) |
| `--shard_id` | *(none)* | Shard subdirectory name (e.g. `shard_0`) |
| `--merge_shards` | *(none)* | Merge mode: pass shard directory paths |
| `--gpu_id` | *(auto)* | Single CUDA device ID (e.g. `"0"`) |
| `--max_model_len` | `8192` | Single context window (no separate vlm/llm) |
| `--batch_vlm` | `4` | Lower default for single GPU |
| `--batch_llm` | `64` | Lower default for single GPU |
| `--ckpt_vlm` | `500` | More frequent checkpointing |
| `--ckpt_llm` | `2000` | More frequent checkpointing |

No `--llm_model`, `--num_gpus`, `--vlm_tp`, `--llm_tp` flags — always unified mode, always `tp=1`.

---

## GPU Memory Guide

| Model | VRAM (bf16) | Config |
|-------|------------|--------|
| Qwen3.5-27B | ~54 GB | `--vlm_tp 2` (two A100-80GB) — **default** |
| Qwen3.5-27B-FP8 | ~27 GB | `--vlm_tp 1 --quantization fp8` (single A100-40GB) |
| Qwen2.5-VL-7B | ~14 GB | `--vlm_tp 1` (single L4/A10) |
| Qwen2.5-VL-32B | ~64 GB | `--vlm_tp 2` |
| Qwen3-VL-30B | ~60 GB | `--vlm_tp 2` |

For single-GPU sharding, use FP8 quantization to fit Qwen3.5-27B on an A100-40GB, or use a smaller model like Qwen2.5-VL-7B on L4/A10.

---

## Common Recipes

```bash
# ── Unified Qwen3.5-27B on 2x A100-80GB (default) ──────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out

# ── 4-GPU node (auto tensor-parallel) ───────────────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --num_gpus 4

# ── Separate VLM + LLM (legacy two-model mode) ─────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --llm_model meta-llama/Llama-3.1-8B-Instruct

# ── FP8 quantization to fit on single A100-40GB ────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --vlm_tp 1 --quantization fp8

# ── Resume after crash (re-run same command) ─────────────────────────────────
python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out

# ── Skip VLM (Step 1 already done, only re-run clustering steps) ─────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --steps 2a,2b,3

# ── Iterate on criterion — change what dimension to cluster by ───────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --criterion "Emotional Tone" --steps 2b,3

# ── Force re-run specific steps ─────────────────────────────────────────────
python ictc_cluster.py \
  --ads_dir /data/ads --output_dir /data/out \
  --num_clusters 8 --steps 2b,3 --force_steps 2b,3

# ── Horizontal sharding across 4 single-GPU VMs ─────────────────────────────
# VM 1:
python ictc_cluster_single_gpu.py --ads_dir /data/ads --output_dir /data/out \
  --start_index 0 --end_index 50000 --shard_id shard_0
# VM 2:
python ictc_cluster_single_gpu.py --ads_dir /data/ads --output_dir /data/out \
  --start_index 50000 --end_index 100000 --shard_id shard_1
# Merge:
python ictc_cluster_single_gpu.py --merge_shards \
  /data/out/shard_0 /data/out/shard_1 --output_dir /data/out/merged

# ── Track 2: Algorithmic Identity & Profiling (same images, different prompts)─
python ictc_cluster_single_gpu_track2.py --ads_dir /data/ads --output_dir /data/out \
  --shard_id track2_identity
# Or via tmux wrapper:
./run_ictc_track2.sh

# ── Track 3: Cultural Representation & Social Values ─────────────────────────
python ictc_cluster_single_gpu_track3.py --ads_dir /data/ads --output_dir /data/out \
  --shard_id track3_cultural
# Or via tmux wrapper:
./run_ictc_track3.sh

# ── Re-cluster any track with different K (reuses VLM captions) ──────────────
python ictc_recluster.py \
  --source_dir /data/out/full_run \
  --output_dir /data/out/track1_k10 \
  --track 1 --num_clusters 10
# Or run all 3 tracks with K=10 via tmux wrapper:
./run_recluster_k10.sh

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

## Iterative Criterion Refinement

The ICTC paper's key insight: **only the VLM step (Step 1) is expensive**. Steps 2b and 3 are fast enough to iterate. The script supports this natively:

```
Run 1: full pipeline → inspect cluster names in logs
Run 2: --steps 2b,3 --criterion "Emotional Tone"   → re-clusters in minutes
Run 3: --steps 2b,3 --criterion "Target Audience"  → re-clusters again
```

**What gets reused across runs:**
- `step1_captions.json` — VLM captions (never re-run unless `--force_steps 1`)
- `step2a_hooks.json` — hook extraction (criterion-agnostic, always reusable)

**What auto-invalidates when you change `--criterion` or `--num_clusters`:**
- Step 2b — always re-runs (metadata check includes criterion + K)
- Step 3 — auto-resets when cluster names change (detected from metadata)

**`--force_steps`** bypasses checkpoint detection entirely:

```bash
# Re-extract hooks AND re-cluster (keep only VLM captions)
python ictc_cluster.py ... --steps 2a,2b,3 --force_steps 2a,2b,3

# Just redo assignment with a different K
python ictc_cluster.py ... --num_clusters 8 --steps 3 --force_steps 3
```

---

## Resume & Crash Recovery

Every step saves an atomic checkpoint (`fsync` + `os.replace` — never corrupts even on NFS/Lustre):

- **Step 1** checkpoints every `--ckpt_vlm` images (default 1000 multi-GPU, 500 single-GPU)
- **Steps 2a/3** checkpoint every `--ckpt_llm` items (default 5000 multi-GPU, 2000 single-GPU)

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

## Deployment: UQ AIO Server (A100-40GB)

### Server Details

| Field | Value |
|-------|-------|
| Host | `<USERNAME>@<SERVER_IP>` |
| SSH | `ssh -i <SSH_KEY>.pem <USERNAME>@<SERVER_IP>` |
| GPU | NVIDIA A100 PCIe 40GB |
| RAM | 590 GB |
| Storage | 3.6 TB NVMe at `/mnt/nvme0n1/` |
| Dataset | 86,446 ads at `/mnt/nvme0n1/data/output/exported-data/ads/` |
| Python env | `~/ictc_env/` (venv, Python 3.10) |
| Model cache | `/mnt/nvme0n1/cache/huggingface/` (root disk too small at 30GB) |

### What Was Set Up (2026-03-05)

1. **NVIDIA drivers**: Installed `nvidia-driver-570` (CUDA 12.8 support). Blacklisted `nouveau` (unused on this VM — display via `virtio_gpu`). Disabled MIG mode (`nvidia-smi -mig 0`).

2. **Python environment**: Created `~/ictc_env` venv. Installed vLLM nightly (`0.16.1rc1.dev258`) — required for Qwen 3.5 support (stable vLLM 0.16.0 doesn't recognise `Qwen3_5ForConditionalGeneration`). PyTorch 2.10.0+cu129, transformers 4.57.6.

3. **Model**: Using **Qwen/Qwen3.5-9B** in bf16 (~18GB VRAM). The 27B default doesn't fit on A100-40GB in bf16 (~54GB), and FP8 quantization fails on A100 (compute 8.0 lacks native FP8 — Marlin kernel fallback has tile alignment issues with Qwen 3.5 dimensions).

### Code Changes for Qwen 3.5 Compatibility

**Thinking mode fix** — Qwen 3.5 is a chain-of-thought model that outputs `<think>` reasoning blocks by default. Without suppression, all pipeline steps return the model's internal reasoning instead of the requested output (e.g., Step 2a hooks all come back as "thinking process:" instead of marketing hooks).

Fix: Added `enable_thinking=False` to **all 4** `apply_chat_template()` call sites:

| Location | Line | Purpose |
|----------|------|---------|
| `VLMProcessor._fmt_prompt()` | Step 1 VLM captioning (vLLM path) |
| `VLMProcessor._process_hf()` | Step 1 VLM captioning (HF fallback) |
| `VLMProcessor._format_text_prompt()` | Steps 2a, 2b, 3 (unified mode text generation) |
| `LLMProcessor._format_prompt()` | Steps 2a, 2b, 3 (standalone LLM path) |

**Directory scan optimisation** — `discover_images()` now accepts `start_index`/`end_index` parameters. It enumerates all subdirectory names (fast `iterdir`), slices to the shard range, then only reads `full_data.json` for the selected directories. Previously it read metadata from all 86,446 dirs before slicing (~2.5 min wasted for small test runs).

**`_strip_thinking()` hardening** — Added handling for unclosed `<think>` blocks (truncated model responses), in addition to closed `<think>...</think>` blocks.

**`max_tokens` increases** — Step 2a: 20→200, Step 3: 30→200 (Qwen 3.5 needs more tokens than Qwen 2.5 for the same tasks).

**`max_model_len` increase (4096→8192)** — Image + text prompt tokens for some ads exceed 4096 (observed 4152–4252 tokens). Raised the default to 8192 in both `ictc_cluster_single_gpu.py` and `run_ictc_single_gpu.sh`. The 9B model in bf16 (~18GB) leaves plenty of VRAM headroom on A100-40GB for the larger context.

**Per-image fallback in `_process_vllm()`** — When a vLLM batch fails (e.g., one oversized image exceeds `max_model_len`), the code now retries each image individually instead of marking the entire batch as BROKEN. Only the single problematic image is skipped, so the pipeline keeps making progress.

### Running Production Jobs

```bash
# SSH in
ssh -i <SSH_KEY>.pem <USERNAME>@<SERVER_IP>

# ── Track 1 (Marketing Strategy) ─────────────────────────────────────────────
# Results at:
/mnt/nvme0n1/data/output/ictc_production/full_run/ictc_final_results.json

# ── Track 2 (Algorithmic Identity) ───────────────────────────────────────────
tmux attach -t ictc_t2       # Ctrl+B then D to detach
tail -f /mnt/nvme0n1/data/output/ictc_production/track2_identity/logs/ictc_*.log
# Results at:
/mnt/nvme0n1/data/output/ictc_production/track2_identity/ictc_final_results.json

# ── Track 3 (Cultural Representation) ────────────────────────────────────────
tmux attach -t ictc_t3       # Ctrl+B then D to detach
tail -f /mnt/nvme0n1/data/output/ictc_production/track3_cultural/logs/ictc_*.log
# Results at:
/mnt/nvme0n1/data/output/ictc_production/track3_cultural/ictc_final_results.json

# ── Re-clustered with K=10 ───────────────────────────────────────────────────
tmux attach -t ictc_k10      # Ctrl+B then D to detach
# Results at:
/mnt/nvme0n1/data/output/ictc_production/track1_k10/ictc_final_results.json
/mnt/nvme0n1/data/output/ictc_production/track2_k10/ictc_final_results.json
/mnt/nvme0n1/data/output/ictc_production/track3_k10/ictc_final_results.json
```

Launcher scripts on the server:
- `~/run_ictc_single_gpu.sh` — Track 1 (tmux session: `ictc`)
- `~/run_ictc_track2.sh` — Track 2 (tmux session: `ictc_t2`)
- `~/run_ictc_track3.sh` — Track 3 (tmux session: `ictc_t3`)
- `~/run_recluster_k10.sh` — All 3 tracks re-clustered with K=10 (tmux session: `ictc_k10`)

### If You Need to Re-run or Change Criteria

```bash
source ~/ictc_env/bin/activate

# Resume Track 1 after crash (checkpoints auto-resume)
HF_HOME=/mnt/nvme0n1/cache/huggingface python3 ~/ictc_cluster_single_gpu.py \
  --ads_dir /mnt/nvme0n1/data/output/exported-data/ads \
  --output_dir /mnt/nvme0n1/data/output/ictc_production \
  --shard_id full_run --vlm_model Qwen/Qwen3.5-9B --verbose

# Resume Track 2 after crash
HF_HOME=/mnt/nvme0n1/cache/huggingface python3 ~/ictc_cluster_single_gpu_track2.py \
  --ads_dir /mnt/nvme0n1/data/output/exported-data/ads \
  --output_dir /mnt/nvme0n1/data/output/ictc_production \
  --shard_id track2_identity --vlm_model Qwen/Qwen3.5-9B --verbose

# Resume Track 3 after crash
HF_HOME=/mnt/nvme0n1/cache/huggingface python3 ~/ictc_cluster_single_gpu_track3.py \
  --ads_dir /mnt/nvme0n1/data/output/exported-data/ads \
  --output_dir /mnt/nvme0n1/data/output/ictc_production \
  --shard_id track3_cultural --vlm_model Qwen/Qwen3.5-9B --verbose

# Re-cluster Track 1 with different criterion (reuses Step 1 captions — fast)
HF_HOME=/mnt/nvme0n1/cache/huggingface python3 ~/ictc_cluster_single_gpu.py \
  --ads_dir /mnt/nvme0n1/data/output/exported-data/ads \
  --output_dir /mnt/nvme0n1/data/output/ictc_production \
  --shard_id full_run --vlm_model Qwen/Qwen3.5-9B \
  --criterion "Emotional Tone" --steps 2b,3 --verbose
```

### Known Limitations on This Server

- **FP8 quantization not supported**: A100 is compute capability 8.0, needs 8.9+ for native FP8. The Marlin fallback kernel crashes on Qwen 3.5's layer dimensions.
- **Root disk is 30GB**: All large files (model cache, outputs) must go to `/mnt/nvme0n1/`. Always set `HF_HOME=/mnt/nvme0n1/cache/huggingface`.
- **vLLM nightly required**: Stable vLLM (0.16.0) doesn't support Qwen 3.5. If you reinstall packages, use: `uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly`

---

## Background: ICTC

Based on the paper *"Image Clustering Conditioned on Text Criteria"* (ICLR 2024).
The paper uses LLaVA-1.5 7B + GPT-4o on ~5k images. This implementation extends it with:
- **Unified model** — Qwen3.5-27B for both vision and text (based on Experiment 3B results)
- **Fully open-source** — no paid APIs
- **Multi-GPU tensor parallelism** via vLLM (`--vlm_tp`, `--llm_tp`)
- **Single-GPU horizontal sharding** across VMs with deterministic merge
- **200k+ scale** — batched inference, multi-day checkpoint/resume
- **Iterative refinement** — re-run clustering steps with new criteria in minutes, reusing VLM outputs
- **Quantization** — AWQ/GPTQ/FP8 to fit larger models on fewer GPUs
