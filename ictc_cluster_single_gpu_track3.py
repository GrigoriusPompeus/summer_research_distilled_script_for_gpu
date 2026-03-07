#!/usr/bin/env python3
"""
ICTC VLM Clustering — Track 3: Cultural Representation & Social Values
===================================================================
Single-GPU shard script (Unified Qwen 3.5) — TRACK 3 PROMPTS

Studies how advertising reflects and constructs cultural narratives,
social values, and visions of "the good life." Enables research on
identity, representation, consumer culture, and social norms.

Pipeline (always unified — single model for all steps):
  Step 0  — Discover images, select shard range [start_index, end_index)
  Step 1  — VLM extraction   : analyse cultural content & representation per ad
  Step 2a — Theme extraction  : extract 2-4 word cultural theme per ad
  Step 2b — Cluster synthesis : synthesise K "Cultural Value Categories"
  Step 3  — Assignment        : assign each ad -> best cultural category

Usage:
  python ictc_cluster_single_gpu_track3.py \
    --ads_dir /data/ads --output_dir /data/out --shard_id track3_cultural

GPU memory guide (single GPU, bfloat16):
  Model                  VRAM    Notes
  —————————————————————————————————————————————————
  Qwen3.5-27B           ~54 GB  A100-80GB / H100-80GB  <- default
  Qwen3.5-27B-FP8       ~27 GB  A100-40GB / A6000-48GB
  Qwen2.5-VL-7B         ~14 GB  L4 / T4 / A10
  Qwen3-VL-7B           ~14 GB  L4 / T4 / A10
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
# PROMPTS  (Track 3: Cultural Representation & Social Values)
# ---------------------------------------------------------------------------

STEP1_PROMPT = """Analyze this image for a cultural studies research project on advertising. Return a valid JSON object.

### STEP 1: CLASSIFY
Determine the image category: "ADVERTISEMENT" | "UI_ONLY" | "BROKEN".

### STEP 2: DESCRIBE
- If "ADVERTISEMENT": Describe the ad with a focus on cultural content and representation:
  - brand_name: The brand or product being advertised.
  - main_text: The primary text or headline shown in the ad.
  - visual_summary: Describe what the ad depicts in terms of people, settings, and cultural signals. Note: who is shown (age, gender presentation, ethnicity if apparent, body type, clothing style), what they are doing (activity, pose, social situation), what setting or lifestyle is portrayed (domestic, professional, outdoors, luxury, casual), and what values or aspirations the ad seems to communicate (health, wealth, beauty, convenience, belonging, independence, etc.). Note the overall aesthetic and tone.
- If "UI_ONLY" or "BROKEN": Leave fields null.

### OUTPUT FORMAT
{
  "category": "ADVERTISEMENT" | "UI_ONLY" | "BROKEN",
  "brand_name": "String or null",
  "main_text": "String or null",
  "visual_summary": "String or null"
}"""

STEP2A_SYSTEM = (
    "You are analyzing an advertisement for a cultural research study on social values "
    "in advertising.\n"
    "Examine this ad description and identify the core cultural theme or social value "
    "being promoted.\n"
    "Do NOT categorize it yet. Just describe the central value or cultural message in "
    "2-4 words.\n"
    'Examples of good outputs: "aspirational wealth display", "domestic comfort ideals", '
    '"youthful beauty standards", "self-improvement imperative", "communal belonging", '
    '"health anxiety", "tech-enabled convenience", "masculine achievement", '
    '"environmental consciousness", "feminine empowerment".\n'
    "Output ONLY the theme phrase, nothing else."
)

STEP2B_TEMPLATE = """You are a Cultural Studies Researcher examining how advertising reflects and constructs social values.

Below are cultural themes identified across a dataset of real advertisements collected from social media users in Australia:
{hooks_json}

TASK:
Group these themes into exactly {k} distinct cultural value categories.

Each category should represent a coherent set of social values, aspirations, or cultural narratives that ads communicate to audiences.

The categories must be mutually exclusive and collectively exhaustive for this dataset.

Think about this from a cultural analysis perspective: what different visions of "the good life" or social identity are being constructed?

OUTPUT JSON FORMAT ONLY:
{{
    "clusters": [
        {{
            "name": "CATEGORY_NAME (2-3 words)",
            "definition": "A plain-language description of the cultural values or social narrative this category represents.",
            "keywords": ["list", "of", "representative", "themes"]
        }}
    ]
}}"""

STEP3_SYSTEM_TEMPLATE = """You are classifying advertisements by their cultural message for a research study.

AVAILABLE CULTURAL CATEGORIES:
{clusters_context}

TASK:
Based on the advertisement description below, assign it to the SINGLE best fitting cultural category from the list above.

Consider: what is the dominant cultural value, lifestyle, or social identity this ad promotes or assumes?

OUTPUT FORMAT:
Return ONLY the exact category name from the list. Nothing else."""


# ---------------------------------------------------------------------------
# RESPONSE CLEANING  (Qwen 3.5 may produce <think> blocks and code fences)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and markdown code fences from model output."""
    # Remove closed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Remove unclosed <think> blocks (truncated responses)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
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
    start_index: int = 0,
    end_index: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, dict]:
    """
    Walk ads_dir looking for the notebook dataset structure:
        ads/<uuid>/full_data.json  +  *.jpg

    Falls back to treating every .jpg/.png directly in ads_dir as an image,
    using the file stem as the observation_id.

    When start_index/end_index are provided, only reads metadata for dirs
    in the shard range (avoids scanning all dirs when only a slice is needed).

    Returns {observation_id: {image_path, platform, ad_format, timestamp}}
    """
    log = log or logging.getLogger("ictc")
    mapping: Dict[str, dict] = {}

    subdirs = sorted(d for d in ads_dir.iterdir() if d.is_dir())
    if subdirs:
        total = len(subdirs)
        log.info(f"Found {total} subdirectories — using full_data.json structure")
        # Slice to shard range BEFORE reading metadata (fast enumeration)
        end = end_index if end_index is not None else total
        selected = subdirs[start_index:end]
        log.info(f"Shard range [{start_index}:{end}] — reading metadata for {len(selected)}/{total} dirs")
        for ad_dir in selected:
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
        end = end_index if end_index is not None else len(all_imgs)
        selected_imgs = all_imgs[start_index:end]
        for p in selected_imgs:
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

    log.info(f"Total images in shard: {len(mapping)}")
    return mapping


# ---------------------------------------------------------------------------
# VLM PROCESSOR  (unified Qwen3.5 — handles all steps via vLLM)
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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
            self.log.warning(f"vLLM batch error ({len(images)} images): {exc}")
            self.log.info("Falling back to per-image processing for this batch")
            # Retry each image individually so only truly broken ones are skipped
            results = []
            for img, p in zip(images, paths):
                try:
                    single_input = [{"prompt": prompt_text, "multi_modal_data": {"image": img}}]
                    out = self.llm.generate(single_input, self.sampling_params)
                    results.append(self._parse_response(out[0].outputs[0].text, p))
                except Exception as e2:
                    self.log.warning(f"Skipping {p.name}: prompt too long or image error: {e2}")
                    results.append(_broken_entry(p))
            return results

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
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
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
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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
# LLM PROCESSOR  (unused in unified mode — kept for compatibility)
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
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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
            STEP2A_SYSTEM, list(v_texts), temperature=0.3, max_tokens=200
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
        "You are a Cultural Studies Researcher examining how advertising reflects and constructs social values.", user_msg, max_tokens=800
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
            system_prompt, list(v_texts), temperature=0.2, max_tokens=200
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
        description="ICTC VLM Clustering — Track 3: Cultural Representation & Social Values (unified Qwen 3.5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Process first 50k images on this VM\n"
            "  python ictc_cluster_single_gpu.py --ads_dir /data/ads --output_dir /data/out \\\n"
            "    --start_index 0 --end_index 50000 --shard_id shard_0\n\n"
            "  # Process next 50k on another VM\n"
            "  python ictc_cluster_single_gpu.py --ads_dir /data/ads --output_dir /data/out \\\n"
            "    --start_index 50000 --end_index 100000 --shard_id shard_1\n\n"
            "  # Merge all shard results\n"
            "  python ictc_cluster_single_gpu.py --merge_shards \\\n"
            "    /data/out/shard_0 /data/out/shard_1 --output_dir /data/out/merged\n\n"
            "  # Small GPU (L4/A10) — use 7B model\n"
            "  python ictc_cluster_single_gpu.py --ads_dir /data/ads --output_dir /data/out \\\n"
            "    --vlm_model Qwen/Qwen2.5-VL-7B-Instruct\n"
        ),
    )

    # ── Merge mode (run this after all shards complete) ────────────────────
    p.add_argument("--merge_shards", nargs="*", type=Path, default=None,
                   help="Merge mode: pass paths to shard output directories. "
                        "Combines step1_captions, step2a_hooks, step3_assignments "
                        "into a single result set in --output_dir.")

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--ads_dir", type=Path, default=None,
                   help="Root of ad dataset (contains <uuid>/ subdirs with full_data.json, "
                        "OR a flat directory of .jpg/.png files). "
                        "Required unless --merge_shards is used.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write checkpoints, results and logs. "
                        "When --shard_id is set, outputs go to output_dir/shard_id/.")

    # ── Sharding ───────────────────────────────────────────────────────────
    p.add_argument("--start_index", type=int, default=0,
                   help="Start index into the sorted image list (inclusive). "
                        "Default 0 = start from the first image.")
    p.add_argument("--end_index", type=int, default=None,
                   help="End index into the sorted image list (exclusive). "
                        "Default None = process until the last image.")
    p.add_argument("--shard_id", type=str, default=None,
                   help="Unique shard identifier (e.g. 'shard_0', 'gpu3'). "
                        "Creates output_dir/shard_id/ subdirectory. "
                        "If not set, results go directly into output_dir.")

    # ── Model selection ────────────────────────────────────────────────────
    p.add_argument("--vlm_model", default="Qwen/Qwen3.5-27B",
                   help="HuggingFace model ID. This single model handles all steps "
                        "(unified mode). Qwen3.5-27B (A100-80GB), Qwen2.5-VL-7B (L4/A10).")

    # ── GPU ────────────────────────────────────────────────────────────────
    p.add_argument("--gpu_id", type=str, default=None,
                   help="CUDA device ID to use, e.g. '0'. "
                        "Sets CUDA_VISIBLE_DEVICES before any GPU code runs.")
    p.add_argument("--gpu_util", type=float, default=0.90,
                   help="vLLM GPU memory utilisation (0.0–1.0). "
                        "Lower this if you see OOM during model load.")

    # ── Precision / quantization ───────────────────────────────────────────
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"],
                   help="Model weight dtype. bfloat16 is best on A100/H100. "
                        "Use float16 for older GPUs (V100, T4).")
    p.add_argument("--quantization", type=str, default=None,
                   choices=["awq", "gptq", "fp8", "squeezellm", None],
                   help="Quantization format. Use 'fp8' to fit Qwen3.5-27B on A100-40GB.")

    # ── Context / image resolution ─────────────────────────────────────────
    p.add_argument("--max_model_len", type=int, default=8192,
                   help="Model context window (tokens). 8192 covers most ad images. "
                        "Increase for very long text. Larger = more VRAM.")
    p.add_argument("--max_image_tokens", type=int, default=1280,
                   help="Max image token budget per image. "
                        "Default 1280 ≈ 1008x1008 px at 28px/token.")

    # ── vLLM engine tuning ─────────────────────────────────────────────────
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graph capture in vLLM. Slower but uses less VRAM.")
    p.add_argument("--swap_space", type=int, default=4,
                   help="CPU swap space in GB (vLLM).")

    # ── Batching ───────────────────────────────────────────────────────────
    p.add_argument("--batch_vlm", type=int, default=4,
                   help="Images per vLLM generate() call. "
                        "Lower than multi-GPU version (4 vs 8) to fit in single GPU.")
    p.add_argument("--batch_llm", type=int, default=64,
                   help="Text prompts per generate() call. "
                        "Lower than multi-GPU version for single-GPU VRAM.")

    # ── Checkpointing ──────────────────────────────────────────────────────
    p.add_argument("--ckpt_vlm", type=int, default=500,
                   help="Save Step 1 checkpoint every N images.")
    p.add_argument("--ckpt_llm", type=int, default=2000,
                   help="Save Steps 2a/3 checkpoint every N items.")

    # ── ICTC / clustering parameters ──────────────────────────────────────
    p.add_argument("--num_clusters", type=int, default=5,
                   help="Number of clusters K to produce.")
    p.add_argument("--top_hooks", type=int, default=60,
                   help="Top-N most frequent hooks fed to Step 2b cluster synthesis.")
    p.add_argument("--criterion", type=str, default="Cultural Representation & Social Values",
                   help="Clustering criterion. Examples: 'Marketing Strategy', "
                        "'Visual Design Style', 'Target Audience'.")

    # ── Reproducibility ────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for vLLM sampling.")

    # ── Pipeline control ───────────────────────────────────────────────────
    p.add_argument("--steps", default="1,2a,2b,3",
                   help="Comma-separated steps to run. Valid: 1, 2a, 2b, 3.")
    p.add_argument("--force_steps", type=str, default=None,
                   help="Comma-separated steps to force re-run. E.g. '2b,3'.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# MERGE SHARDS
# ---------------------------------------------------------------------------

def merge_shards(shard_dirs: List[Path], output_dir: Path, log: logging.Logger) -> None:
    """Merge checkpoint JSONs from multiple shard directories into one."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(output_dir)

    merged_mapping: Dict[str, dict] = {}
    merged_captions: Dict[str, dict] = {}
    merged_ui: Dict[str, dict] = {}
    merged_broken: Dict[str, dict] = {}
    merged_hooks: Dict[str, dict] = {}
    merged_assignments: Dict[str, dict] = {}

    for sd in shard_dirs:
        if not sd.exists():
            log.warning(f"Shard directory not found: {sd}")
            continue
        shard_ckpt = CheckpointManager(sd)

        for src, dest in [
            ("image_mapping.json", merged_mapping),
            ("step1_captions.json", merged_captions),
            ("ui_only_images.json", merged_ui),
            ("broken_images.json", merged_broken),
            ("step2a_hooks.json", merged_hooks),
            ("step3_final_assignment.json", merged_assignments),
        ]:
            data = shard_ckpt.load(src)
            if data:
                dest.update(data)
                log.info(f"  {sd.name}/{src}: +{len(data)} entries")

    ckpt.save(merged_mapping, "image_mapping.json")
    ckpt.save(merged_captions, "step1_captions.json")
    ckpt.save(merged_ui, "ui_only_images.json")
    ckpt.save(merged_broken, "broken_images.json")
    ckpt.save(merged_hooks, "step2a_hooks.json")
    if merged_assignments:
        ckpt.save(merged_assignments, "step3_final_assignment.json")

    # Copy cluster definitions from first shard that has them
    for sd in shard_dirs:
        shard_ckpt = CheckpointManager(sd)
        clusters = shard_ckpt.load("step2b_dynamic_clusters.json")
        if clusters:
            ckpt.save(clusters, "step2b_dynamic_clusters.json")
            break

    log.info("=" * 70)
    log.info("Merge complete")
    log.info(f"  Total images   : {len(merged_mapping)}")
    log.info(f"  Valid captions : {len(merged_captions)}")
    log.info(f"  Hooks          : {len(merged_hooks)}")
    log.info(f"  Assignments    : {len(merged_assignments)}")
    log.info(f"  Output         : {output_dir}")
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    args = parse_args()

    # ── Merge mode: combine shard results and exit ─────────────────────────
    if args.merge_shards is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log = setup_logging(args.output_dir / "logs", verbose=args.verbose)
        merge_shards(args.merge_shards, args.output_dir, log)
        return

    # ── Normal shard processing mode ───────────────────────────────────────
    if args.ads_dir is None:
        print("ERROR: --ads_dir is required (unless using --merge_shards)")
        sys.exit(1)

    # Resolve output directory with shard_id
    actual_output = args.output_dir / args.shard_id if args.shard_id else args.output_dir
    actual_output.mkdir(parents=True, exist_ok=True)
    log = setup_logging(actual_output / "logs", verbose=args.verbose)

    # ── GPU selection ──────────────────────────────────────────────────────
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        log.info(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_id}")

    # ── Force-clear checkpoints for specified steps ────────────────────────
    _STEP_FILES = {
        "1":  ["step1_captions.json", "ui_only_images.json", "broken_images.json"],
        "2a": ["step2a_hooks.json"],
        "2b": ["step2b_dynamic_clusters.json", "step2b_metadata.json"],
        "3":  ["step3_final_assignment.json", "step3_metadata.json"],
    }
    if args.force_steps:
        for s in (s.strip() for s in args.force_steps.split(",")):
            for fname in _STEP_FILES.get(s, []):
                fp = actual_output / fname
                if fp.exists():
                    fp.unlink()
                    log.info(f"  [force_steps] Deleted {fname} — step {s} will re-run")

    # Graceful shutdown
    def _shutdown(sig, frame):  # noqa: ANN001
        del frame
        log.warning(f"Signal {sig} received — checkpoints are safe, exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    steps = {s.strip() for s in args.steps.split(",")}
    ckpt = CheckpointManager(actual_output)

    log.info("=" * 70)
    log.info("ICTC Track 3: Cultural Representation & Social Values — Single-GPU Run")
    log.info(f"  shard_id          : {args.shard_id or '(none)'}")
    log.info(f"  ads_dir           : {args.ads_dir}")
    log.info(f"  output_dir        : {actual_output}")
    log.info(f"  image range       : [{args.start_index}, {args.end_index or 'end'})")
    log.info(f"  vlm_model         : {args.vlm_model}")
    log.info(f"    tp=1  dtype={args.dtype}  quant={args.quantization or 'none'}")
    log.info(f"    max_model_len={args.max_model_len}  max_image_tokens={args.max_image_tokens}")
    log.info(f"  gpu_util          : {args.gpu_util}")
    log.info(f"  num_clusters (K)  : {args.num_clusters}")
    log.info(f"  criterion         : {args.criterion}")
    log.info(f"  batch_vlm         : {args.batch_vlm}")
    log.info(f"  batch_llm         : {args.batch_llm}")
    log.info(f"  seed              : {args.seed}")
    log.info(f"  steps             : {args.steps}")
    log.info("=" * 70)

    # -- Step 0: discover images and apply shard range -------------------------
    if ckpt.exists("image_mapping.json"):
        image_mapping: Dict[str, dict] = ckpt.load("image_mapping.json")
        log.info(f"Resumed image_mapping.json ({len(image_mapping)} images)")
    else:
        image_mapping = discover_images(
            args.ads_dir, max_images=None,
            start_index=args.start_index, end_index=args.end_index, log=log,
        )
        ckpt.save(image_mapping, "image_mapping.json")

    if not image_mapping:
        log.error("No images in this shard range — check --start_index / --end_index")
        sys.exit(1)

    captions: Dict[str, dict] = {}
    hooks: Dict[str, dict] = {}
    cluster_def: Dict = {}
    assignments: Dict[str, dict] = {}

    vlm_needed_for_step1 = "1" in steps
    llm_steps_needed = any(s in steps for s in ["2a", "2b", "3"])

    # ── Always unified: one model, single GPU ──────────────────────────────
    vlm = None
    need_model = vlm_needed_for_step1 or llm_steps_needed

    if need_model:
        captions = ckpt.load("step1_captions.json") or {}
        already_done_s1 = (
            len(captions)
            + len(ckpt.load("ui_only_images.json") or {})
            + len(ckpt.load("broken_images.json") or {})
        )
        step1_already_complete = already_done_s1 >= len(image_mapping)
        has_work = (vlm_needed_for_step1 and not step1_already_complete) or llm_steps_needed
        if has_work:
            vlm = VLMProcessor(
                model_name=args.vlm_model,
                tp=1,  # single GPU
                max_model_len=args.max_model_len,
                max_image_tokens=args.max_image_tokens,
                gpu_util=args.gpu_util,
                dtype=args.dtype,
                quantization=args.quantization,
                enforce_eager=args.enforce_eager,
                swap_space=args.swap_space,
                seed=args.seed,
                log=log,
            )

    # Step 1
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

    # Steps 2a/2b/3 (same model, no reload)
    if llm_steps_needed:
        log.info("Unified mode — reusing model for text steps (no model swap)")

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
    log.info("Shard finished.")
    log.info(f"Results in: {actual_output}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
