#!/usr/bin/env python3
"""
ICTC VLM Clustering — Production Script for GPU Cluster (Unified Model)
=========================================================================
Implements "Image Clustering Conditioned on Text Criteria" at scale.
Based on Experiment 3B findings: a single Qwen 3.5 model handles BOTH
vision captioning AND text reasoning, producing better clusters than a
separate VLM + LLM pipeline (higher coherence, better balance, zero
unclassified ads).

Pipeline (unified mode — default):
  Step 0  — Discover images from dataset structure
  Step 1  — Qwen3.5-27B captioning : image -> JSON {category, brand, text, description}
  Step 2a — Same model (text-only)  : extract 2-4 word marketing hook per ad
  Step 2b — Same model (text-only)  : synthesise K cluster definitions from top hooks
  Step 3  — Same model (text-only)  : assign each ad -> best cluster

  In unified mode the model stays loaded for all steps — no expensive
  unload/reload between VLM and LLM phases.  To use a separate LLM
  (e.g. Llama), pass --llm_model explicitly.

Designed for:
  * Any number of GPUs  (set --num_gpus N, or fine-tune --vlm_tp / --llm_tp separately)
  * 200 k+ images       (batched inference, multi-day runs)
  * SSH-resilient       (all output to rotating log files, atomic JSON saves)
  * Full resume         (every step checkpoints; re-run to continue after crash)
  * Any quantization    (--quantization awq/gptq/fp8 for fitting larger models)

Quick start (unified — recommended):
  python ictc_cluster.py \\
    --ads_dir     /data/dataset/ads \\
    --output_dir  /data/results/run1 \\
    --num_clusters 5

Separate VLM + LLM (legacy):
  python ictc_cluster.py \\
    --ads_dir /data/ads --output_dir /data/out \\
    --llm_model meta-llama/Llama-3.1-8B-Instruct

Smoke test (100 images):
  python ictc_cluster.py --ads_dir ... --output_dir ... --max_images 100 --verbose

GPU memory guide (bfloat16, no quantization):
  Model                  VRAM    Recommended config
  ─────────────────────────────────────────────────────────────
  Qwen3.5-27B           ~54 GB  --vlm_tp 2  (two A100-80GB)  <- default (unified)
  Qwen3.5-27B-FP8       ~27 GB  --vlm_tp 1  --quantization fp8
  Qwen2.5-VL-7B         ~14 GB  --vlm_tp 1  (single A100/L40)
  Qwen2.5-VL-32B        ~64 GB  --vlm_tp 2
  Qwen3-VL-30B          ~60 GB  --vlm_tp 2

  Tip: --num_gpus 4 sets vlm_tp=4 automatically.
       Override individually with --vlm_tp / --llm_tp.
"""

import argparse
import gc
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# PROMPTS  (identical to the notebook so results are comparable)
# ---------------------------------------------------------------------------

STEP1_PROMPT = """Analyze this image for an ad database. Return a valid JSON object.

### STEP 1: CLASSIFY
Determine the image category:
- "ADVERTISEMENT": A clear, valid commercial ad.
- "UI_ONLY": Social media feed, settings menu, or app interface with NO specific ad.
- "BROKEN": Black screen, loading spinner, error message, or system lock screen.

### STEP 2: DESCRIBE
- If "ADVERTISEMENT": Extract brand_name, main_text, and a visual_summary.
- If "UI_ONLY" or "BROKEN": Leave description fields null or empty.

### OUTPUT FORMAT
{
  "category": "ADVERTISEMENT" | "UI_ONLY" | "BROKEN",
  "brand_name": "String or null",
  "main_text": "String or null",
  "visual_summary": "String or null"
}"""

STEP2A_SYSTEM = (
    'Analyze this ad description and identify the core "marketing hook" or psychological '
    "mechanism used. Do NOT categorize it yet. Just describe the specific appeal in 2-4 words.\n"
    'Examples: "scarcity urgency", "social proof testimonial", "luxury status signaling", '
    '"problem-solution utility".\nOutput ONLY the hook phrase.'
)

STEP2B_TEMPLATE = """You are an expert analyst specializing in {criterion}.

I have analyzed a dataset of items and extracted the following specific patterns/hooks:
{hooks_json}

TASK:
Group these patterns into exactly {k} distinct, high-level {criterion} clusters.
The categories must be mutually exclusive and collectively exhaustive for this dataset.

OUTPUT JSON FORMAT ONLY:
{{
    "clusters": [
        {{
            "name": "CATEGORY_NAME (2-3 words)",
            "definition": "A 1-sentence definition of what this cluster entails.",
            "keywords": ["list", "of", "representative", "hooks"]
        }}
    ]
}}"""

STEP3_SYSTEM_TEMPLATE = """You are classifying ads into specific strategies.

AVAILABLE STRATEGIES:
{clusters_context}

TASK:
Assign the advertisement below to the SINGLE best fitting strategy from the list above.

OUTPUT FORMAT:
Return ONLY the exact strategy name from the list. Nothing else."""


# ---------------------------------------------------------------------------
# RESPONSE CLEANING  (Qwen 3.5 may produce <think> blocks and code fences)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and markdown code fences from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    return text.strip("`").strip()


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ictc_{ts}.log"
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    log = logging.getLogger("ictc")
    log.info(f"Log file: {log_file}")
    return log


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGER
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Atomic JSON save/load. Writes to .tmp then renames to avoid corruption."""

    def __init__(self, output_dir: Path):
        self.d = output_dir
        self.d.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict, filename: str) -> None:
        path = self.d / filename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # os.replace is atomic on POSIX and handles pre-existing target on Windows
        os.replace(tmp, path)

    def load(self, filename: str) -> Optional[dict]:
        path = self.d / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def exists(self, filename: str) -> bool:
        return (self.d / filename).exists()


# ---------------------------------------------------------------------------
# DATA DISCOVERY
# ---------------------------------------------------------------------------

def discover_images(
    ads_dir: Path,
    max_images: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, dict]:
    """
    Walk ads_dir looking for the notebook dataset structure:
        ads/<uuid>/full_data.json  +  *.jpg

    Falls back to treating every .jpg/.png directly in ads_dir as an image,
    using the file stem as the observation_id.

    Returns {observation_id: {image_path, platform, ad_format, timestamp}}
    """
    log = log or logging.getLogger("ictc")
    mapping: Dict[str, dict] = {}

    subdirs = sorted(d for d in ads_dir.iterdir() if d.is_dir())
    if subdirs:
        log.info(f"Found {len(subdirs)} subdirectories — using full_data.json structure")
        for ad_dir in subdirs:
            fd = ad_dir / "full_data.json"
            jpg_files = sorted(ad_dir.glob("*.jpg"))
            if not jpg_files:
                jpg_files = sorted(ad_dir.glob("*.png"))
            if not jpg_files:
                continue

            obs_id: str
            platform = "UNKNOWN"
            ad_format = "UNKNOWN"
            timestamp = ""

            if fd.exists():
                try:
                    with open(fd, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    obs_id = data.get("observation_id") or ad_dir.name
                    obs = data.get("observation", {})
                    platform = obs.get("platform", "UNKNOWN")
                    ad_format = obs.get("ad_format", "UNKNOWN")
                    timestamp = data.get("timestamp", "")
                except Exception:
                    obs_id = ad_dir.name
            else:
                obs_id = ad_dir.name

            mapping[obs_id] = {
                "image_path": str(jpg_files[0]),
                "platform": platform,
                "ad_format": ad_format,
                "timestamp": timestamp,
            }
    else:
        log.info("No subdirectories — using flat image directory")
        all_imgs = sorted(ads_dir.glob("*.jpg")) + sorted(ads_dir.glob("*.png"))
        for p in all_imgs:
            mapping[p.stem] = {
                "image_path": str(p),
                "platform": "UNKNOWN",
                "ad_format": "UNKNOWN",
                "timestamp": "",
            }

    if max_images:
        keys = list(mapping.keys())[:max_images]
        mapping = {k: mapping[k] for k in keys}
        log.info(f"Capped at {max_images} images (--max_images)")

    log.info(f"Total images to process: {len(mapping)}")
    return mapping


# ---------------------------------------------------------------------------
# VLM PROCESSOR  (Step 1 — unified Qwen3.5 via vLLM, or any HF VLM)
# ---------------------------------------------------------------------------

class VLMProcessor:
    """
    Wraps a vLLM-served vision-language model for batched image captioning.
    Supports any VLM available on HuggingFace (Qwen3-VL, Qwen2.5-VL, LLaVA, etc.)
    Falls back to HuggingFace transformers if vLLM is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        tp: int = 2,
        max_model_len: int = 4096,
        max_image_tokens: int = 1280,
        gpu_util: float = 0.90,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        swap_space: int = 4,
        max_tokens: int = 300,
        seed: int = 42,
        log: Optional[logging.Logger] = None,
    ):
        self.log = log or logging.getLogger("ictc")
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.dtype = dtype

        q_tag = f"  quantization={quantization}" if quantization else ""
        self.log.info(f"Loading VLM: {model_name}  (tp={tp}, dtype={dtype}{q_tag})")
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor

            self.sampling_params = SamplingParams(
                temperature=0.0,
                top_p=0.001,
                max_tokens=max_tokens,
                repetition_penalty=1.1,
                seed=seed,
            )
            lm_kwargs: dict = dict(
                model=model_name,
                tensor_parallel_size=tp,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_util,
                limit_mm_per_prompt={"image": 1},
                dtype=dtype,
                enforce_eager=enforce_eager,
                swap_space=swap_space,
                seed=seed,
                trust_remote_code=True,
            )
            if quantization:
                lm_kwargs["quantization"] = quantization
            # Qwen3-VL / Qwen2.5-VL: allow larger image token budget
            lm_kwargs["mm_processor_kwargs"] = {"max_pixels": max_image_tokens * 28 * 28}
            self.llm = LLM(**lm_kwargs)
            # Processor only used for chat template formatting, not tokenisation
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.backend = "vllm"
            self.log.info("VLM ready (vLLM backend)")

        except Exception as exc:
            self.log.warning(f"vLLM load failed ({exc}), falling back to HuggingFace transformers")
            self._load_hf(model_name, dtype)

    def _load_hf(self, model_name: str, dtype: str = "bfloat16") -> None:
        """
        HuggingFace transformers fallback.
        Uses device_map='auto' which spreads across all GPUs visible via
        CUDA_VISIBLE_DEVICES — set that env var before launch to control placement.
        """
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            dtype, torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",       # spreads across all CUDA_VISIBLE_DEVICES
            trust_remote_code=True,
        )
        self.model.eval()
        self.backend = "hf"
        self.log.info("VLM ready (HuggingFace transformers fallback)")

    def _fmt_prompt(self) -> str:
        """Return a chat-template-formatted string with an image placeholder."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},   # placeholder; vLLM substitutes multi_modal_data
                    {"type": "text", "text": STEP1_PROMPT},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def process_batch(self, image_paths: List[Path]) -> List[dict]:
        """Caption a batch of images. Returns list aligned 1:1 with image_paths."""
        from PIL import Image as PILImage

        images: list = []
        valid_indices: list = []
        for i, p in enumerate(image_paths):
            try:
                with PILImage.open(p) as raw:
                    images.append(raw.convert("RGB"))
                valid_indices.append(i)
            except Exception as exc:
                self.log.warning(f"Cannot open {p}: {exc}")

        # Build result list aligned with original image_paths (broken for failed opens)
        results = [_broken_entry(p) for p in image_paths]
        if not images:
            return results

        valid_paths = [image_paths[i] for i in valid_indices]
        if self.backend == "vllm":
            valid_results = self._process_vllm(images, valid_paths)
        else:
            valid_results = self._process_hf(images, valid_paths)

        for idx, result in zip(valid_indices, valid_results):
            results[idx] = result
        return results

    def _process_vllm(self, images: list, paths: list) -> List[dict]:
        prompt_text = self._fmt_prompt()
        inputs = [
            {"prompt": prompt_text, "multi_modal_data": {"image": img}}
            for img in images
        ]
        try:
            outputs = self.llm.generate(inputs, self.sampling_params)
            return [self._parse_response(o.outputs[0].text, p) for o, p in zip(outputs, paths)]
        except Exception as exc:
            self.log.error(f"vLLM generate error: {exc}")
            return [_broken_entry(p) for p in paths]

    def _process_hf(self, images: list, paths: list) -> List[dict]:
        """Process one image at a time (HF Qwen batching with variable-size images is fragile)."""
        import torch

        results = []
        for img, p in zip(images, paths):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": STEP1_PROMPT},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                device = next(self.model.parameters()).device
                inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(device)

                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_tokens, do_sample=False
                    )

                generated = gen_ids[0][inputs["input_ids"].shape[-1]:]
                raw = self.processor.decode(generated, skip_special_tokens=True)
                results.append(self._parse_response(raw, p))
            except Exception as exc:
                self.log.warning(f"HF VLM error on {p.name}: {exc}")
                results.append(_broken_entry(p))
        return results

    @staticmethod
    def _parse_response(raw: str, path: Path) -> dict:
        """Extract JSON from model output, with keyword-based fallback."""
        raw = _strip_thinking(raw)
        try:
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, Exception):
            pass
        # Heuristic fallback when JSON extraction fails — use path.name in summary for traceability
        category = "ADVERTISEMENT"
        low = raw.lower()
        if any(w in low for w in ["ui only", "ui_only", "interface", "menu", "settings"]):
            category = "UI_ONLY"
        elif any(w in low for w in ["broken", "black screen", "error", "loading", "spinner"]):
            category = "BROKEN"
        return {
            "category": category,
            "brand_name": None,
            "main_text": raw[:200] if raw else None,
            "visual_summary": f"[parse_fallback:{path.name}] {raw[:160]}" if raw else None,
        }

    # ── Text-only generation (unified mode: reuse VLM for Steps 2a/2b/3) ──

    def _format_text_prompt(self, system: str, user: str) -> str:
        """Format a text-only chat prompt using the VLM's processor."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(
        self,
        system_prompt: str,
        user_texts: List[str],
        temperature: float = 0.3,
        max_tokens: int = 100,
    ) -> List[str]:
        """Batch text-only generation — same interface as LLMProcessor."""
        from vllm import SamplingParams

        if self.backend != "vllm":
            raise RuntimeError("Text generation in unified mode requires vLLM backend")
        prompts = [self._format_text_prompt(system_prompt, u) for u in user_texts]
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sp)
        return [_strip_thinking(o.outputs[0].text.strip()) for o in outputs]

    def generate_single(self, system: str, user: str, max_tokens: int = 800) -> str:
        """Single text-only call — used for Step 2b cluster synthesis."""
        results = self.generate_batch(system, [user], temperature=0.3, max_tokens=max_tokens)
        return results[0] if results else ""

    def unload(self) -> None:
        """Free GPU memory."""
        import torch
        import torch.distributed as dist

        if getattr(self, "backend", None) == "vllm":
            if hasattr(self, "llm"):
                del self.llm
        else:
            if hasattr(self, "model"):
                del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        time.sleep(3)  # let CUDA actually release memory


# ---------------------------------------------------------------------------
# LLM PROCESSOR  (Steps 2a / 2b / 3 — separate LLM via vLLM, legacy mode)
# ---------------------------------------------------------------------------

class LLMProcessor:
    """
    Wraps vLLM for high-throughput text-only inference.
    Supports any causal LLM on HuggingFace (Llama, Mistral, Qwen, etc.)
    All prompts in a step are submitted together in a single generate() call.
    """

    def __init__(
        self,
        model_name: str,
        tp: int = 1,
        max_model_len: int = 4096,
        gpu_util: float = 0.90,
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        swap_space: int = 4,
        seed: int = 42,
        log: Optional[logging.Logger] = None,
    ):
        self.log = log or logging.getLogger("ictc")
        q_tag = f"  quantization={quantization}" if quantization else ""
        self.log.info(f"Loading LLM: {model_name}  (tp={tp}, dtype={dtype}{q_tag})")

        from vllm import LLM
        from transformers import AutoTokenizer

        lm_kwargs: dict = dict(
            model=model_name,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            dtype=dtype,
            enforce_eager=enforce_eager,
            swap_space=swap_space,
            seed=seed,
            trust_remote_code=True,
        )
        if quantization:
            lm_kwargs["quantization"] = quantization
        self.llm = LLM(**lm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.log.info("LLM ready")

    def _format_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(
        self,
        system_prompt: str,
        user_texts: List[str],
        temperature: float = 0.3,
        max_tokens: int = 100,
    ) -> List[str]:
        """Batch-generate for many user inputs sharing the same system prompt."""
        from vllm import SamplingParams

        prompts = [self._format_prompt(system_prompt, u) for u in user_texts]
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_single(self, system: str, user: str, max_tokens: int = 800) -> str:
        """Single call — used for Step 2b cluster synthesis."""
        results = self.generate_batch(system, [user], temperature=0.3, max_tokens=max_tokens)
        return results[0] if results else ""

    def unload(self) -> None:
        import torch
        import torch.distributed as dist

        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        time.sleep(3)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _broken_entry(path: Path) -> dict:
    return {
        "category": "BROKEN",
        "brand_name": None,
        "main_text": None,
        "visual_summary": f"Failed to process {path.name}",
    }


def _safe_copy(src: Path, dst: Path) -> None:
    if not dst.exists():
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def run_step1(
    image_mapping: Dict[str, dict],
    ckpt: CheckpointManager,
    vlm: VLMProcessor,
    batch_size: int,
    checkpoint_interval: int,
    log: logging.Logger,
) -> Dict[str, dict]:
    """
    VLM-caption every image.
    Output: step1_captions.json  (same schema as the notebook)
    """
    CAPTIONS_FILE = "step1_captions.json"
    UI_FILE = "ui_only_images.json"
    BROKEN_FILE = "broken_images.json"

    captions: Dict[str, dict] = ckpt.load(CAPTIONS_FILE) or {}
    ui_only: Dict[str, dict] = ckpt.load(UI_FILE) or {}
    broken_imgs: Dict[str, dict] = ckpt.load(BROKEN_FILE) or {}

    total_done = len(captions) + len(ui_only) + len(broken_imgs)
    if total_done >= len(image_mapping):
        log.info(
            f"Step 1 already complete — "
            f"valid={len(captions)}  ui={len(ui_only)}  broken={len(broken_imgs)}"
        )
        return captions

    done_ids = set(captions) | set(ui_only) | set(broken_imgs)
    todo = [(k, v) for k, v in image_mapping.items() if k not in done_ids]
    log.info(f"Step 1: {len(done_ids)}/{len(image_mapping)} done, {len(todo)} remaining")

    base_out = ckpt.d
    for sub in ["categorized_images/valid_ads", "categorized_images/ui_only", "categorized_images/broken"]:
        (base_out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    processed = len(done_ids)
    last_ckpt = processed
    total = len(image_mapping)

    for batch_start in range(0, len(todo), batch_size):
        chunk = todo[batch_start: batch_start + batch_size]
        obs_ids = [item[0] for item in chunk]
        img_paths = [Path(item[1]["image_path"]) for item in chunk]
        meta_list = [item[1] for item in chunk]

        parsed_list = vlm.process_batch(img_paths)

        for obs_id, meta, src_path, parsed in zip(obs_ids, meta_list, img_paths, parsed_list):
            category = parsed.get("category", "BROKEN")
            base_entry = {
                "image_file": str(src_path),
                "platform": meta["platform"],
                "ad_format": meta["ad_format"],
            }

            if category == "ADVERTISEMENT":
                dest = base_out / "categorized_images" / "valid_ads" / f"{obs_id}.jpg"
                _safe_copy(src_path, dest)
                captions[obs_id] = {
                    **base_entry,
                    "status": "valid",
                    "brand": parsed.get("brand_name"),
                    "text": parsed.get("main_text"),
                    "description": parsed.get("visual_summary"),
                    "categorized_path": str(dest),
                }
            elif category == "UI_ONLY":
                dest = base_out / "categorized_images" / "ui_only" / f"{obs_id}.jpg"
                _safe_copy(src_path, dest)
                ui_only[obs_id] = {**base_entry, "category": "UI_ONLY", "raw_response": parsed}
            else:
                dest = base_out / "categorized_images" / "broken" / f"{obs_id}.jpg"
                _safe_copy(src_path, dest)
                broken_imgs[obs_id] = {**base_entry, "category": "BROKEN", "raw_response": parsed}

        processed += len(chunk)

        if processed - last_ckpt >= checkpoint_interval:
            last_ckpt = processed
            ckpt.save(captions, CAPTIONS_FILE)
            ckpt.save(ui_only, UI_FILE)
            ckpt.save(broken_imgs, BROKEN_FILE)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 1
            eta_secs = int((total - processed) / rate) if rate > 0 else 0
            log.info(
                f"  Step 1 — {processed}/{total} ({100*processed/total:.1f}%)  "
                f"rate={rate:.2f} img/s  ETA={timedelta(seconds=eta_secs)}"
            )

    ckpt.save(captions, CAPTIONS_FILE)
    ckpt.save(ui_only, UI_FILE)
    ckpt.save(broken_imgs, BROKEN_FILE)
    log.info(
        f"Step 1 complete — valid={len(captions)}  ui={len(ui_only)}  broken={len(broken_imgs)}"
    )
    return captions


def run_step2a(
    captions: Dict[str, dict],
    ckpt: CheckpointManager,
    llm,  # LLMProcessor or VLMProcessor (unified mode) — duck-typed via generate_batch()
    batch_size: int,
    checkpoint_interval: int,
    log: logging.Logger,
) -> Dict[str, dict]:
    """Extract a 2-4 word marketing hook from each caption."""
    HOOKS_FILE = "step2a_hooks.json"
    hooks: Dict[str, dict] = ckpt.load(HOOKS_FILE) or {}

    if len(hooks) >= len(captions):
        log.info(f"Step 2a already complete ({len(hooks)} hooks)")
        return hooks

    todo = [(k, v) for k, v in captions.items() if k not in hooks]
    log.info(f"Step 2a: {len(hooks)}/{len(captions)} done, {len(todo)} remaining")

    processed = len(hooks)
    last_ckpt = processed
    for batch_start in range(0, len(todo), batch_size):
        chunk = todo[batch_start: batch_start + batch_size]
        obs_ids = [item[0] for item in chunk]
        user_texts = [
            (item[1].get("text") or item[1].get("description") or "").strip()
            for item in chunk
        ]

        # Items with no text get a placeholder so they aren't retried on resume
        for obs_id, txt in zip(obs_ids, user_texts):
            if not txt and obs_id not in hooks:
                hooks[obs_id] = {"hook": "no description", "description_snippet": ""}

        valid_pairs = [(i, t) for i, t in zip(obs_ids, user_texts) if t]
        if not valid_pairs:
            processed += len(chunk)
            continue

        v_ids, v_texts = zip(*valid_pairs)
        raw_hooks = llm.generate_batch(
            STEP2A_SYSTEM, list(v_texts), temperature=0.3, max_tokens=20
        )

        for obs_id, hook_raw, txt in zip(v_ids, raw_hooks, v_texts):
            clean = _strip_thinking(hook_raw).lower().strip("\"'").split("\n")[0]
            clean = re.sub(r"^(marketing hook:|hook:)\s*", "", clean).strip()
            hooks[obs_id] = {"hook": clean, "description_snippet": txt[:100]}

        processed += len(chunk)
        if processed - last_ckpt >= checkpoint_interval:
            last_ckpt = processed
            ckpt.save(hooks, HOOKS_FILE)
            log.info(f"  Step 2a — {processed}/{len(captions)}")

    ckpt.save(hooks, HOOKS_FILE)
    top5 = Counter(d["hook"] for d in hooks.values()).most_common(5)
    log.info(f"Step 2a complete — {len(hooks)} hooks. Top 5: {top5}")
    return hooks


def run_step2b(
    hooks: Dict[str, dict],
    ckpt: CheckpointManager,
    llm,  # LLMProcessor or VLMProcessor (unified mode) — duck-typed via generate_single()
    num_clusters: int,
    top_n_hooks: int,
    criterion: str,
    log: logging.Logger,
) -> Dict:
    """Synthesise K cluster definitions from the top hooks — single LLM call."""
    CLUSTERS_FILE = "step2b_dynamic_clusters.json"
    META_FILE = "step2b_metadata.json"

    existing = ckpt.load(CLUSTERS_FILE)
    meta = ckpt.load(META_FILE)
    if (
        existing and meta
        and meta.get("num_hooks") == len(hooks)
        and meta.get("num_clusters") == num_clusters
        and meta.get("criterion") == criterion
    ):
        log.info(f"Step 2b already complete ({len(existing.get('clusters', []))} clusters)")
        for c in existing.get("clusters", []):
            log.info(f"  [{c['name']}]: {c['definition']}")
        return existing

    all_hooks = [d["hook"] for d in hooks.values()]
    top_hooks = [h for h, _ in Counter(all_hooks).most_common(top_n_hooks)]
    log.info(f"Step 2b: {len(top_hooks)} top hooks -> {num_clusters} clusters  [criterion: {criterion}]")

    user_msg = "Create the clusters now.\n\n" + STEP2B_TEMPLATE.format(
        hooks_json=json.dumps(top_hooks, indent=2), k=num_clusters, criterion=criterion
    )
    response = llm.generate_single(
        f"You are an expert analyst specializing in {criterion}.", user_msg, max_tokens=800
    )
    response = _strip_thinking(response)

    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        cluster_def = json.loads(m.group()) if m else {}
        assert "clusters" in cluster_def and len(cluster_def["clusters"]) >= 1
    except Exception as exc:
        log.error(f"Step 2b JSON parse failed ({exc}):\n{response[:600]}")
        cluster_def = {
            "clusters": [
                {
                    "name": f"Cluster_{i+1}",
                    "definition": f"Auto-generated cluster {i+1}.",
                    "keywords": [],
                }
                for i in range(num_clusters)
            ]
        }

    ckpt.save(cluster_def, CLUSTERS_FILE)
    ckpt.save({"num_hooks": len(hooks), "num_clusters": num_clusters, "criterion": criterion}, META_FILE)

    for c in cluster_def.get("clusters", []):
        log.info(f"  [{c['name']}]: {c['definition']}")
    return cluster_def


def run_step3(
    captions: Dict[str, dict],
    cluster_def: Dict,
    ckpt: CheckpointManager,
    llm,  # LLMProcessor or VLMProcessor (unified mode) — duck-typed via generate_batch()
    batch_size: int,
    checkpoint_interval: int,
    log: logging.Logger,
) -> Dict[str, dict]:
    """Assign each ad to its best-fitting cluster."""
    ASSIGN_FILE = "step3_final_assignment.json"
    META_FILE = "step3_metadata.json"
    assignments: Dict[str, dict] = ckpt.load(ASSIGN_FILE) or {}
    meta = ckpt.load(META_FILE)

    current_cluster_names = sorted(c["name"] for c in cluster_def.get("clusters", []))
    cached_cluster_names = sorted(meta.get("cluster_names", [])) if meta else []

    if current_cluster_names != cached_cluster_names and assignments:
        log.info(
            "Cluster definitions changed (criterion or K updated) — "
            "re-running Step 3 from scratch with new clusters"
        )
        assignments = {}

    if len(assignments) >= len(captions):
        log.info(f"Step 3 already complete ({len(assignments)} assignments)")
        dist = Counter(v["cluster"] for v in assignments.values())
        for name, count in dist.most_common():
            log.info(f"  {name}: {count}")
        return assignments

    clusters = cluster_def.get("clusters", [])
    valid_names = [c["name"] for c in clusters]
    ctx = "\n".join(f"- {c['name']}: {c['definition']}" for c in clusters)
    system_prompt = STEP3_SYSTEM_TEMPLATE.format(clusters_context=ctx)

    todo = [(k, v) for k, v in captions.items() if k not in assignments]
    log.info(f"Step 3: {len(assignments)}/{len(captions)} done, {len(todo)} remaining")

    processed = len(assignments)
    last_ckpt = processed
    for batch_start in range(0, len(todo), batch_size):
        chunk = todo[batch_start: batch_start + batch_size]
        obs_ids = [item[0] for item in chunk]
        user_texts = [
            (item[1].get("text") or item[1].get("description") or "").strip()
            for item in chunk
        ]

        # Items with no text get "Unclassified" so they aren't retried on resume
        for obs_id, txt in zip(obs_ids, user_texts):
            if not txt and obs_id not in assignments:
                assignments[obs_id] = {"cluster": "Unclassified", "original_description": ""}

        valid_pairs = [(i, t) for i, t in zip(obs_ids, user_texts) if t]
        if not valid_pairs:
            processed += len(chunk)
            continue

        v_ids, v_texts = zip(*valid_pairs)
        responses = llm.generate_batch(
            system_prompt, list(v_texts), temperature=0.2, max_tokens=30
        )

        for obs_id, resp, txt in zip(v_ids, responses, v_texts):
            resp = _strip_thinking(resp)
            matched = "Unclassified"
            # Multi-strategy matching: exact → substring → word-overlap (from Exp 3B)
            resp_lower = resp.lower().strip().strip("\"'")
            for name in valid_names:
                if resp_lower == name.lower():
                    matched = name
                    break
            if matched == "Unclassified":
                for name in valid_names:
                    if name.lower() in resp_lower or resp_lower in name.lower():
                        matched = name
                        break
            if matched == "Unclassified":
                resp_words = set(resp_lower.split())
                best_score, best = 0.0, None
                for name in valid_names:
                    nw = set(name.lower().split())
                    if not nw:
                        continue
                    score = len(resp_words & nw) / len(nw)
                    if score > best_score and score >= 0.5:
                        best_score, best = score, name
                if best:
                    matched = best
            assignments[obs_id] = {"cluster": matched, "original_description": txt[:100]}

        processed += len(chunk)
        if processed - last_ckpt >= checkpoint_interval:
            last_ckpt = processed
            ckpt.save(assignments, ASSIGN_FILE)
            log.info(f"  Step 3 — {processed}/{len(captions)}")

    ckpt.save(assignments, ASSIGN_FILE)
    ckpt.save({"cluster_names": current_cluster_names}, META_FILE)
    dist = Counter(v["cluster"] for v in assignments.values())
    log.info("Step 3 complete. Distribution:")
    for name, count in dist.most_common():
        log.info(f"  {name}: {count}")
    return assignments


def export_results(
    image_mapping: Dict[str, dict],
    captions: Dict[str, dict],
    hooks: Dict[str, dict],
    cluster_def: Dict,
    assignments: Dict[str, dict],
    ckpt: CheckpointManager,
    log: logging.Logger,
) -> None:
    """Write ictc_final_results.json — matches the notebook's export format."""
    cluster_names = [c["name"] for c in cluster_def.get("clusters", [])]
    rows = []
    for obs_id, info in image_mapping.items():
        row = {
            "observation_id": obs_id,
            "image_path": info["image_path"],
            "platform": info["platform"],
            "ad_format": info["ad_format"],
        }
        if obs_id in captions:
            row["caption"] = captions[obs_id].get("text") or captions[obs_id].get("description")
        if obs_id in hooks:
            row["initial_label"] = hooks[obs_id]["hook"]
        if obs_id in assignments:
            row["final_cluster"] = assignments[obs_id]["cluster"]
        rows.append(row)

    summary = {
        "metadata": {
            "total_images": len(image_mapping),
            "valid_ads": len(captions),
            "num_clusters": len(cluster_names),
            "cluster_names": cluster_names,
            "run_date": datetime.now().isoformat(),
        },
        "cluster_definitions": cluster_def,
        "results": rows,
    }
    ckpt.save(summary, "ictc_final_results.json")
    log.info(f"Exported {len(rows)} entries -> ictc_final_results.json")


# ---------------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICTC VLM Clustering — unified Qwen 3.5 pipeline for GPU cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 2x A100-80GB, unified Qwen3.5-27B for all steps (default)\n"
            "  python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out\n\n"
            "  # Separate VLM + LLM (legacy mode)\n"
            "  python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out \\\n"
            "    --llm_model meta-llama/Llama-3.1-8B-Instruct\n\n"
            "  # 4x GPU node, auto tensor-parallel\n"
            "  python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out --num_gpus 4\n\n"
            "  # Different domain: cluster by visual style instead of marketing\n"
            "  python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out \\\n"
            "    --criterion 'Visual Design Style' --num_clusters 8\n\n"
            "  # Resume after crash (just re-run same command)\n"
            "  python ictc_cluster.py --ads_dir /data/ads --output_dir /data/out\n"
        ),
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--ads_dir", type=Path, required=True,
                   help="Root of ad dataset (contains <uuid>/ subdirs with full_data.json, "
                        "OR a flat directory of .jpg/.png files)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write checkpoints, results and logs")

    # ── Model selection ────────────────────────────────────────────────────
    p.add_argument("--vlm_model", default="Qwen/Qwen3.5-27B",
                   help="HuggingFace model ID for the VLM (Step 1 image captioning). "
                        "In unified mode (default) this model also handles Steps 2a/2b/3. "
                        "Qwen3.5-27B, Qwen3-VL-*, Qwen2.5-VL-*, LLaVA-*, etc.")
    p.add_argument("--llm_model", default="unified",
                   help="LLM for Steps 2a/2b/3.  'unified' (default) = reuse the VLM "
                        "for all text steps (no unload/reload, recommended).  Set to a "
                        "separate model ID (e.g. meta-llama/Llama-3.1-8B-Instruct) "
                        "for legacy two-model mode.")

    # ── GPU / parallelism ──────────────────────────────────────────────────
    p.add_argument("--num_gpus", type=int, default=None,
                   help="Total visible GPUs. Sets BOTH --vlm_tp and --llm_tp to this value "
                        "unless those are specified explicitly. Convenience shorthand for "
                        "'use all available GPUs for whichever model is currently loaded.'")
    p.add_argument("--vlm_tp", type=int, default=None,
                   help="Tensor-parallel degree for VLM (overrides --num_gpus for VLM). "
                        "Rule of thumb: model_size_GB / gpu_vram_GB, rounded up.")
    p.add_argument("--llm_tp", type=int, default=None,
                   help="Tensor-parallel degree for LLM (overrides --num_gpus for LLM). "
                        "Usually 1 is fine for 8B models.")
    p.add_argument("--gpu_ids", type=str, default=None,
                   help="Comma-separated CUDA device IDs to use, e.g. '0,1' or '0,1,2,3'. "
                        "Sets CUDA_VISIBLE_DEVICES before any GPU code runs. "
                        "Leave unset to use all available GPUs.")
    p.add_argument("--gpu_util", type=float, default=0.90,
                   help="vLLM GPU memory utilisation per device (0.0–1.0). "
                        "Lower this if you see OOM during model load.")

    # ── Precision / quantization ───────────────────────────────────────────
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"],
                   help="Model weight dtype. bfloat16 is best on A100/H100. "
                        "Use float16 for older GPUs (V100, T4).")
    p.add_argument("--quantization", type=str, default=None,
                   choices=["awq", "gptq", "fp8", "squeezellm", None],
                   help="Quantization format. Use 'awq' or 'gptq' to fit larger models "
                        "(e.g. 72B on 2x A100-80GB). Must match the model's published format.")
    p.add_argument("--vlm_quantization", type=str, default=None,
                   choices=["awq", "gptq", "fp8", "squeezellm", None],
                   help="Override --quantization for VLM only.")
    p.add_argument("--llm_quantization", type=str, default=None,
                   choices=["awq", "gptq", "fp8", "squeezellm", None],
                   help="Override --quantization for LLM only.")

    # ── Context / image resolution ─────────────────────────────────────────
    p.add_argument("--vlm_max_model_len", type=int, default=4096,
                   help="VLM context window (tokens). Increase for longer ad text or "
                        "higher image resolution. Larger = more VRAM.")
    p.add_argument("--llm_max_model_len", type=int, default=4096,
                   help="LLM context window (tokens). 4096 is plenty for hook extraction.")
    p.add_argument("--max_image_tokens", type=int, default=1280,
                   help="Max image token budget per image (Qwen3-VL / Qwen2.5-VL). "
                        "Higher = better quality on dense text ads but more VRAM. "
                        "Default 1280 ≈ 1008x1008 px at 28px/token.")

    # ── vLLM engine tuning ─────────────────────────────────────────────────
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graph capture in vLLM. Slower but uses less VRAM "
                        "and avoids graph-capture OOM on large models.")
    p.add_argument("--swap_space", type=int, default=4,
                   help="CPU swap space per GPU in GB (vLLM). Increase if you see "
                        "KV-cache eviction warnings.")

    # ── Batching ───────────────────────────────────────────────────────────
    p.add_argument("--batch_vlm", type=int, default=8,
                   help="Images submitted per vLLM generate() call. "
                        "Reduce to 2–4 if you get OOM during Step 1.")
    p.add_argument("--batch_llm", type=int, default=128,
                   help="Text prompts per LLM generate() call. Can be very large (256–512) "
                        "for 8B models; vLLM handles continuous batching internally.")

    # ── Checkpointing ──────────────────────────────────────────────────────
    p.add_argument("--ckpt_vlm", type=int, default=1000,
                   help="Save Step 1 checkpoint every N images processed.")
    p.add_argument("--ckpt_llm", type=int, default=5000,
                   help="Save Steps 2a/3 checkpoint every N items processed.")

    # ── ICTC / clustering parameters ──────────────────────────────────────
    p.add_argument("--num_clusters", type=int, default=5,
                   help="Number of clusters K to produce. Higher K = more granular grouping.")
    p.add_argument("--top_hooks", type=int, default=60,
                   help="Top-N most frequent hooks fed to Step 2b cluster synthesis. "
                        "60–100 works well; too many may overflow LLM context.")
    p.add_argument("--criterion", type=str, default="Marketing Strategy",
                   help="Clustering criterion sent to the LLM. Controls what dimension "
                        "the clusters represent. Examples: 'Marketing Strategy', "
                        "'Visual Design Style', 'Target Audience', 'Emotional Tone'.")

    # ── Reproducibility ────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for vLLM sampling (Step 2a/3 temperature>0 calls).")

    # ── Pipeline control ───────────────────────────────────────────────────
    p.add_argument("--steps", default="1,2a,2b,3",
                   help="Comma-separated steps to run. Valid values: 1, 2a, 2b, 3. "
                        "Image discovery always runs. Useful values: "
                        "'1' (VLM only), '2a,2b,3' (skip VLM, resume from captions).")
    p.add_argument("--force_steps", type=str, default=None,
                   help="Comma-separated steps to force re-run even if their output already exists. "
                        "E.g. '2b,3' deletes step2b/3 checkpoints and re-generates from scratch. "
                        "Useful for iterative criterion refinement: keep Step 1+2a, redo clustering. "
                        "Example: --steps 2a,2b,3 --force_steps 2a will re-extract hooks then re-cluster.")
    p.add_argument("--max_images", type=int, default=None,
                   help="Process at most N images total. Useful for smoke-testing "
                        "before committing to a full 200k run.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging (very chatty).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir / "logs", verbose=args.verbose)

    # ── GPU selection: set CUDA_VISIBLE_DEVICES before any torch/vllm import ──
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        visible = args.gpu_ids.split(",")
        log.info(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}  ({len(visible)} GPU(s))")

    # ── Resolve unified vs separate mode ──────────────────────────────────
    unified = args.llm_model == "unified"

    # ── Resolve tensor-parallel degrees ────────────────────────────────────
    # Priority: explicit --vlm_tp / --llm_tp > --num_gpus > defaults
    fallback_tp = args.num_gpus or 2
    vlm_tp = args.vlm_tp if args.vlm_tp is not None else fallback_tp
    llm_tp = args.llm_tp if args.llm_tp is not None else (args.num_gpus or 1)

    # ── Resolve per-model quantization ─────────────────────────────────────
    vlm_quant = args.vlm_quantization or args.quantization
    llm_quant = args.llm_quantization or args.quantization

    # In unified mode, use the larger context window for text steps
    if unified:
        vlm_max_model_len = max(args.vlm_max_model_len, args.llm_max_model_len)
    else:
        vlm_max_model_len = args.vlm_max_model_len

    # ── Force-clear checkpoints for specified steps ─────────────────────────
    # Maps each step to the files that must be deleted to trigger a fresh run.
    _STEP_FILES = {
        "1":  ["step1_captions.json", "ui_only_images.json", "broken_images.json"],
        "2a": ["step2a_hooks.json"],
        "2b": ["step2b_dynamic_clusters.json", "step2b_metadata.json"],
        "3":  ["step3_final_assignment.json", "step3_metadata.json"],
    }
    if args.force_steps:
        for s in (s.strip() for s in args.force_steps.split(",")):
            for fname in _STEP_FILES.get(s, []):
                fp = args.output_dir / fname
                if fp.exists():
                    fp.unlink()
                    log.info(f"  [force_steps] Deleted {fname} — step {s} will re-run")

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(sig, frame):  # noqa: ANN001
        del frame  # unused, required by signal handler signature
        log.warning(f"Signal {sig} received — checkpoints are safe, exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    steps = {s.strip() for s in args.steps.split(",")}
    ckpt = CheckpointManager(args.output_dir)

    log.info("=" * 70)
    log.info("ICTC Production Run")
    log.info(f"  mode              : {'UNIFIED (single model)' if unified else 'SEPARATE (VLM + LLM)'}")
    log.info(f"  ads_dir           : {args.ads_dir}")
    log.info(f"  output_dir        : {args.output_dir}")
    log.info(f"  vlm_model         : {args.vlm_model}")
    log.info(f"    tp={vlm_tp}  dtype={args.dtype}  quant={vlm_quant or 'none'}")
    log.info(f"    max_model_len={vlm_max_model_len}  max_image_tokens={args.max_image_tokens}")
    if unified:
        log.info(f"  llm_model         : (unified — same as VLM)")
    else:
        log.info(f"  llm_model         : {args.llm_model}")
        log.info(f"    tp={llm_tp}  dtype={args.dtype}  quant={llm_quant or 'none'}")
        log.info(f"    max_model_len={args.llm_max_model_len}")
    log.info(f"  gpu_util          : {args.gpu_util}")
    log.info(f"  enforce_eager     : {args.enforce_eager}")
    log.info(f"  swap_space        : {args.swap_space} GB")
    log.info(f"  num_clusters (K)  : {args.num_clusters}")
    log.info(f"  criterion         : {args.criterion}")
    log.info(f"  batch_vlm         : {args.batch_vlm}")
    log.info(f"  batch_llm         : {args.batch_llm}")
    log.info(f"  seed              : {args.seed}")
    log.info(f"  steps             : {args.steps}")
    log.info("=" * 70)

    # -- Step 0: discover images (always runs) --------------------------------
    if ckpt.exists("image_mapping.json"):
        image_mapping: Dict[str, dict] = ckpt.load("image_mapping.json")
        log.info(f"Resumed image_mapping.json ({len(image_mapping)} images)")
        if args.max_images and len(image_mapping) > args.max_images:
            keys = list(image_mapping.keys())[: args.max_images]
            image_mapping = {k: image_mapping[k] for k in keys}
    else:
        image_mapping = discover_images(args.ads_dir, args.max_images, log)
        ckpt.save(image_mapping, "image_mapping.json")

    if not image_mapping:
        log.error("No images found — check --ads_dir")
        sys.exit(1)

    captions: Dict[str, dict] = {}
    hooks: Dict[str, dict] = {}
    cluster_def: Dict = {}
    assignments: Dict[str, dict] = {}

    vlm_needed_for_step1 = "1" in steps
    llm_steps_needed = any(s in steps for s in ["2a", "2b", "3"])

    # ── Unified mode: one model handles everything ─────────────────────────
    if unified:
        # Load the model once, keep it for all steps
        need_model = vlm_needed_for_step1 or llm_steps_needed
        vlm = None

        if need_model:
            # Check if Step 1 still has work
            captions = ckpt.load("step1_captions.json") or {}
            already_done_s1 = (
                len(captions)
                + len(ckpt.load("ui_only_images.json") or {})
                + len(ckpt.load("broken_images.json") or {})
            )
            step1_already_complete = already_done_s1 >= len(image_mapping)

            # Only load model if there's actual work to do
            has_work = (vlm_needed_for_step1 and not step1_already_complete) or llm_steps_needed
            if has_work:
                vlm = VLMProcessor(
                    model_name=args.vlm_model,
                    tp=vlm_tp,
                    max_model_len=vlm_max_model_len,
                    max_image_tokens=args.max_image_tokens,
                    gpu_util=args.gpu_util,
                    dtype=args.dtype,
                    quantization=vlm_quant,
                    enforce_eager=args.enforce_eager,
                    swap_space=args.swap_space,
                    seed=args.seed,
                    log=log,
                )

        # Step 1: VLM captioning (with images)
        captions = ckpt.load("step1_captions.json") or {}
        if vlm_needed_for_step1:
            already_done = (
                len(captions)
                + len(ckpt.load("ui_only_images.json") or {})
                + len(ckpt.load("broken_images.json") or {})
            )
            if already_done < len(image_mapping):
                captions = run_step1(
                    image_mapping, ckpt, vlm,
                    batch_size=args.batch_vlm,
                    checkpoint_interval=args.ckpt_vlm,
                    log=log,
                )
            else:
                log.info(f"Step 1 already fully done ({len(captions)} valid captions loaded)")

        if not captions:
            captions = ckpt.load("step1_captions.json") or {}
        if not captions:
            log.error("No captions available. Run Step 1 first (include '1' in --steps).")
            sys.exit(1)

        # Steps 2a/2b/3: text-only (same model, no reload)
        if llm_steps_needed:
            log.info("Unified mode — reusing VLM for text steps (no model swap)")

            if "2a" in steps:
                hooks = run_step2a(
                    captions, ckpt, vlm,
                    batch_size=args.batch_llm,
                    checkpoint_interval=args.ckpt_llm,
                    log=log,
                )
            else:
                hooks = ckpt.load("step2a_hooks.json") or {}

            if "2b" in steps:
                cluster_def = run_step2b(
                    hooks, ckpt, vlm,
                    num_clusters=args.num_clusters,
                    top_n_hooks=args.top_hooks,
                    criterion=args.criterion,
                    log=log,
                )
            else:
                cluster_def = ckpt.load("step2b_dynamic_clusters.json") or {}

            if "3" in steps:
                assignments = run_step3(
                    captions, cluster_def, ckpt, vlm,
                    batch_size=args.batch_llm,
                    checkpoint_interval=args.ckpt_llm,
                    log=log,
                )
            else:
                assignments = ckpt.load("step3_final_assignment.json") or {}

        if vlm is not None:
            vlm.unload()
            del vlm

    # ── Separate mode: VLM for Step 1, then swap to LLM ────────────────────
    else:
        captions = ckpt.load("step1_captions.json") or {}

        if vlm_needed_for_step1:
            already_done = (
                len(captions)
                + len(ckpt.load("ui_only_images.json") or {})
                + len(ckpt.load("broken_images.json") or {})
            )
            if already_done < len(image_mapping):
                vlm = VLMProcessor(
                    model_name=args.vlm_model,
                    tp=vlm_tp,
                    max_model_len=args.vlm_max_model_len,
                    max_image_tokens=args.max_image_tokens,
                    gpu_util=args.gpu_util,
                    dtype=args.dtype,
                    quantization=vlm_quant,
                    enforce_eager=args.enforce_eager,
                    swap_space=args.swap_space,
                    seed=args.seed,
                    log=log,
                )
                captions = run_step1(
                    image_mapping, ckpt, vlm,
                    batch_size=args.batch_vlm,
                    checkpoint_interval=args.ckpt_vlm,
                    log=log,
                )
                log.info("Unloading VLM to free GPU memory for LLM phase ...")
                vlm.unload()
                del vlm
            else:
                log.info(f"Step 1 already fully done ({len(captions)} valid captions loaded)")

        if not captions:
            captions = ckpt.load("step1_captions.json") or {}
        if not captions:
            log.error("No captions available. Run Step 1 first (include '1' in --steps).")
            sys.exit(1)

        if llm_steps_needed:
            llm = LLMProcessor(
                model_name=args.llm_model,
                tp=llm_tp,
                max_model_len=args.llm_max_model_len,
                gpu_util=args.gpu_util,
                dtype=args.dtype,
                quantization=llm_quant,
                enforce_eager=args.enforce_eager,
                swap_space=args.swap_space,
                seed=args.seed,
                log=log,
            )

            if "2a" in steps:
                hooks = run_step2a(
                    captions, ckpt, llm,
                    batch_size=args.batch_llm,
                    checkpoint_interval=args.ckpt_llm,
                    log=log,
                )
            else:
                hooks = ckpt.load("step2a_hooks.json") or {}

            if "2b" in steps:
                cluster_def = run_step2b(
                    hooks, ckpt, llm,
                    num_clusters=args.num_clusters,
                    top_n_hooks=args.top_hooks,
                    criterion=args.criterion,
                    log=log,
                )
            else:
                cluster_def = ckpt.load("step2b_dynamic_clusters.json") or {}

            if "3" in steps:
                assignments = run_step3(
                    captions, cluster_def, ckpt, llm,
                    batch_size=args.batch_llm,
                    checkpoint_interval=args.ckpt_llm,
                    log=log,
                )
            else:
                assignments = ckpt.load("step3_final_assignment.json") or {}

            llm.unload()
            del llm

    # -- Export ----------------------------------------------------------------
    captions = captions or ckpt.load("step1_captions.json") or {}
    hooks = hooks or ckpt.load("step2a_hooks.json") or {}
    cluster_def = cluster_def or ckpt.load("step2b_dynamic_clusters.json") or {}
    assignments = assignments or ckpt.load("step3_final_assignment.json") or {}

    if captions and assignments:
        export_results(image_mapping, captions, hooks, cluster_def, assignments, ckpt, log)
    else:
        log.info("Skipping export (pipeline not fully complete yet)")

    log.info("=" * 70)
    log.info("Pipeline finished.")
    log.info(f"Results in: {args.output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
