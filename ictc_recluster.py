#!/usr/bin/env python3
"""
ICTC Re-clustering — Run Steps 2a/2b/3 from existing VLM captions
===================================================================
Reads Step 1 captions (and optionally Step 2a hooks) from a previous
track run, then re-runs the text-only clustering steps with a new K
or criterion. Writes all output to a NEW directory so nothing is
overwritten.

Supports all 3 prompt tracks:
  Track 1 — Marketing Strategy
  Track 2 — Algorithmic Identity & Profiling
  Track 3 — Cultural Representation & Social Values

Usage:
  # Re-cluster Track 1 with k=10 (reuses captions + hooks from full_run/)
  python ictc_recluster.py \
    --source_dir /data/out/full_run \
    --output_dir /data/out/track1_k10 \
    --track 1 --num_clusters 10

  # Re-cluster Track 2 with k=10
  python ictc_recluster.py \
    --source_dir /data/out/track2_identity \
    --output_dir /data/out/track2_k10 \
    --track 2 --num_clusters 10

  # Re-cluster Track 3 with k=10, also re-extract themes from scratch
  python ictc_recluster.py \
    --source_dir /data/out/track3_cultural \
    --output_dir /data/out/track3_k10 \
    --track 3 --num_clusters 10 --redo_2a

GPU memory guide (text-only steps — much lighter than VLM):
  Qwen3.5-9B   ~18 GB   A100-40GB comfortable
  Qwen3.5-27B  ~54 GB   needs 2x A100-80GB or FP8
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# TRACK PROMPTS — identical to the per-track scripts
# ---------------------------------------------------------------------------

# ── Track 1: Marketing Strategy ───────────────────────────────────────────

TRACK1_STEP2A_SYSTEM = (
    'Analyze this ad description and identify the core "marketing hook" or psychological '
    "mechanism used. Do NOT categorize it yet. Just describe the specific appeal in 2-4 words.\n"
    'Examples: "scarcity urgency", "social proof testimonial", "luxury status signaling", '
    '"fear of missing out", "aspirational lifestyle", "price anchoring".\n'
    "Output ONLY the hook phrase."
)

TRACK1_STEP2B_TEMPLATE = """You are an expert marketing analyst.

I have analyzed a dataset of items and extracted the following specific patterns/hooks:
{hooks_json}

TASK:
Group these hooks into exactly {k} distinct, high-level categories.

OUTPUT JSON FORMAT ONLY:
{{
    "clusters": [
        {{
            "name": "CATEGORY_NAME (2-3 words)",
            "definition": "A 1-sentence definition of what this marketing strategy entails.",
            "keywords": ["list", "of", "representative", "hooks"]
        }}
    ]
}}"""

TRACK1_STEP2B_ROLE = "You are an expert marketing analyst."

TRACK1_STEP3_SYSTEM_TEMPLATE = """You are classifying advertisements by marketing strategy.

AVAILABLE STRATEGIES:
{clusters_context}

TASK:
Assign the advertisement below to the SINGLE best fitting marketing strategy from the list above. Return ONLY the exact strategy name. Nothing else."""

# ── Track 2: Algorithmic Identity & Profiling ─────────────────────────────

TRACK2_STEP2A_SYSTEM = (
    'Analyze this ad description and identify the "algorithmic persona" or specific '
    "socio-cultural demographic the platform believes the user belongs to. "
    "Do NOT categorize it yet. Just describe this niche data-identity in 2-4 words.\n"
    'Examples of good outputs: "hustle-culture tech bro", "exhausted millennial parent", '
    '"status-conscious elite", "health-anxious optimizer".\n'
    "Output ONLY the persona phrase."
)

TRACK2_STEP2B_TEMPLATE = """You are a Critical Data Scholar researching surveillance capitalism and algorithmic identity profiling.

I have analyzed a dataset of social media ads and extracted the following implied algorithmic personas (the target demographic the algorithm assumes the user fits):
{hooks_json}

TASK:
Group these personas into exactly {k} distinct, high-level "Algorithmic Identity Clusters" representing the major demographic boxes platform algorithms put users into.

OUTPUT JSON FORMAT ONLY:
{{
    "clusters": [
        {{
            "name": "CATEGORY_NAME (2-3 words)",
            "definition": "A 1-sentence definition of what this identity cluster entails.",
            "keywords": ["list", "of", "representative", "personas"]
        }}
    ]
}}"""

TRACK2_STEP2B_ROLE = "You are a Critical Data Scholar researching surveillance capitalism and algorithmic identity profiling."

TRACK2_STEP3_SYSTEM_TEMPLATE = """You are classifying targeted ads by the demographic identity the algorithm has assigned to the user.

AVAILABLE IDENTITY CLUSTERS:
{clusters_context}

TASK:
Assign the advertisement below to the SINGLE best fitting identity cluster from the platform's perspective. Return ONLY the exact cluster name. Nothing else."""

# ── Track 3: Cultural Representation & Social Values ──────────────────────

TRACK3_STEP2A_SYSTEM = (
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

TRACK3_STEP2B_TEMPLATE = """You are a Cultural Studies Researcher examining how advertising reflects and constructs social values.

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

TRACK3_STEP2B_ROLE = "You are a Cultural Studies Researcher examining how advertising reflects and constructs social values."

TRACK3_STEP3_SYSTEM_TEMPLATE = """You are classifying advertisements by their cultural message for a research study.

AVAILABLE CULTURAL CATEGORIES:
{clusters_context}

TASK:
Based on the advertisement description below, assign it to the SINGLE best fitting cultural category from the list above.

Consider: what is the dominant cultural value, lifestyle, or social identity this ad promotes or assumes?

OUTPUT FORMAT:
Return ONLY the exact category name from the list. Nothing else."""


# ---------------------------------------------------------------------------
# TRACK PROMPT SELECTOR
# ---------------------------------------------------------------------------

TRACKS = {
    1: {
        "name": "Marketing Strategy",
        "step2a_system": TRACK1_STEP2A_SYSTEM,
        "step2b_template": TRACK1_STEP2B_TEMPLATE,
        "step2b_role": TRACK1_STEP2B_ROLE,
        "step3_system_template": TRACK1_STEP3_SYSTEM_TEMPLATE,
        "criterion": "Marketing Strategy",
    },
    2: {
        "name": "Algorithmic Identity & Profiling",
        "step2a_system": TRACK2_STEP2A_SYSTEM,
        "step2b_template": TRACK2_STEP2B_TEMPLATE,
        "step2b_role": TRACK2_STEP2B_ROLE,
        "step3_system_template": TRACK2_STEP3_SYSTEM_TEMPLATE,
        "criterion": "Algorithmic Identity Profiling",
    },
    3: {
        "name": "Cultural Representation & Social Values",
        "step2a_system": TRACK3_STEP2A_SYSTEM,
        "step2b_template": TRACK3_STEP2B_TEMPLATE,
        "step2b_role": TRACK3_STEP2B_ROLE,
        "step3_system_template": TRACK3_STEP3_SYSTEM_TEMPLATE,
        "criterion": "Cultural Representation & Social Values",
    },
}


# ---------------------------------------------------------------------------
# RESPONSE CLEANING
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and markdown code fences from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    return text.strip("`").strip()


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"recluster_{ts}.log"
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    log = logging.getLogger("recluster")
    log.info(f"Log file: {log_file}")
    return log


# ---------------------------------------------------------------------------
# CHECKPOINT MANAGER
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Atomic JSON save/load."""

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
# LLM LOADER (text-only — no VLM image processing needed)
# ---------------------------------------------------------------------------

class TextLLM:
    """
    Lightweight vLLM wrapper for text-only generation (Steps 2a/2b/3).
    No image processing — just loads the model for text inference.
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
        seed: int = 42,
        log: Optional[logging.Logger] = None,
    ):
        self.log = log or logging.getLogger("recluster")
        q_tag = f"  quantization={quantization}" if quantization else ""
        self.log.info(f"Loading model: {model_name}  (tp={tp}, dtype={dtype}{q_tag})")

        from vllm import LLM
        from transformers import AutoTokenizer

        lm_kwargs: dict = dict(
            model=model_name,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            dtype=dtype,
            enforce_eager=enforce_eager,
            seed=seed,
            trust_remote_code=True,
        )
        if quantization:
            lm_kwargs["quantization"] = quantization
        self.llm = LLM(**lm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.log.info("Model ready")

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
        from vllm import SamplingParams

        prompts = [self._format_prompt(system_prompt, u) for u in user_texts]
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sp)
        return [_strip_thinking(o.outputs[0].text.strip()) for o in outputs]

    def generate_single(self, system: str, user: str, max_tokens: int = 800) -> str:
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


# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def run_step2a(
    captions: Dict[str, dict],
    ckpt: CheckpointManager,
    llm: TextLLM,
    step2a_system: str,
    batch_size: int,
    checkpoint_interval: int,
    log: logging.Logger,
) -> Dict[str, dict]:
    """Extract 2-4 word label per ad from its caption."""
    HOOKS_FILE = "step2a_hooks.json"
    hooks: Dict[str, dict] = ckpt.load(HOOKS_FILE) or {}

    if len(hooks) >= len(captions):
        log.info(f"Step 2a already complete ({len(hooks)} hooks)")
        top5 = Counter(d["hook"] for d in hooks.values()).most_common(5)
        log.info(f"  Top 5: {top5}")
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

        for obs_id, txt in zip(obs_ids, user_texts):
            if not txt and obs_id not in hooks:
                hooks[obs_id] = {"hook": "no description", "description_snippet": ""}

        valid_pairs = [(i, t) for i, t in zip(obs_ids, user_texts) if t]
        if not valid_pairs:
            processed += len(chunk)
            continue

        v_ids, v_texts = zip(*valid_pairs)
        raw_hooks = llm.generate_batch(
            step2a_system, list(v_texts), temperature=0.3, max_tokens=200
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
    llm: TextLLM,
    num_clusters: int,
    top_n_hooks: int,
    criterion: str,
    step2b_template: str,
    step2b_role: str,
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

    user_msg = "Create the clusters now.\n\n" + step2b_template.format(
        hooks_json=json.dumps(top_hooks, indent=2), k=num_clusters, criterion=criterion
    )
    response = llm.generate_single(step2b_role, user_msg, max_tokens=1500)
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
    llm: TextLLM,
    step3_system_template: str,
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
        log.info("Cluster definitions changed — re-running Step 3 from scratch")
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
    system_prompt = step3_system_template.format(clusters_context=ctx)

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
    captions: Dict[str, dict],
    hooks: Dict[str, dict],
    cluster_def: Dict,
    assignments: Dict[str, dict],
    ckpt: CheckpointManager,
    image_mapping: Optional[Dict[str, dict]],
    log: logging.Logger,
) -> None:
    """Write ictc_final_results.json."""
    cluster_names = [c["name"] for c in cluster_def.get("clusters", [])]
    rows = []
    for obs_id in captions:
        row = {"observation_id": obs_id}
        if image_mapping and obs_id in image_mapping:
            row["image_path"] = image_mapping[obs_id].get("image_path", "")
            row["platform"] = image_mapping[obs_id].get("platform", "")
            row["ad_format"] = image_mapping[obs_id].get("ad_format", "")
        if obs_id in captions:
            row["caption"] = captions[obs_id].get("text") or captions[obs_id].get("description")
        if obs_id in hooks:
            row["hook"] = hooks[obs_id].get("hook")
        if obs_id in assignments:
            row["cluster"] = assignments[obs_id].get("cluster")
        rows.append(row)
    ckpt.save(rows, "ictc_final_results.json")
    log.info(f"Exported {len(rows)} entries -> ictc_final_results.json")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICTC Re-clustering — re-run Steps 2a/2b/3 from existing VLM captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--source_dir", type=str, required=True,
                   help="Directory containing step1_captions.json (and optionally step2a_hooks.json) "
                        "from a previous track run")
    p.add_argument("--output_dir", type=str, required=True,
                   help="NEW directory for re-clustered output (will not overwrite source)")
    p.add_argument("--track", type=int, required=True, choices=[1, 2, 3],
                   help="Which prompt track to use (1=Marketing, 2=Identity, 3=Cultural)")
    p.add_argument("--num_clusters", type=int, default=10,
                   help="Number of output clusters (K). Default: 10")

    # Model
    p.add_argument("--vlm_model", type=str, default="Qwen/Qwen3.5-9B",
                   help="Model to load for text generation")
    p.add_argument("--gpu_util", type=float, default=0.90,
                   help="vLLM GPU memory utilisation (0.0–1.0)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"])
    p.add_argument("--quantization", type=str, default=None)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="Context window (text-only steps need less than VLM)")
    p.add_argument("--gpu_id", type=str, default=None,
                   help="CUDA device ID to use, e.g. '0'")

    # Pipeline
    p.add_argument("--redo_2a", action="store_true",
                   help="Re-extract Step 2a hooks instead of copying from source")
    p.add_argument("--top_hooks", type=int, default=60,
                   help="Top-N hooks fed into Step 2b synthesis")
    p.add_argument("--batch_llm", type=int, default=64)
    p.add_argument("--ckpt_llm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # GPU selection
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_dir / "logs", args.verbose)
    track = TRACKS[args.track]

    log.info("=" * 70)
    log.info(f"ICTC Re-clustering — Track {args.track}: {track['name']}")
    log.info(f"  source_dir      : {source_dir}")
    log.info(f"  output_dir      : {output_dir}")
    log.info(f"  num_clusters (K): {args.num_clusters}")
    log.info(f"  model           : {args.vlm_model}")
    log.info(f"  redo_2a         : {args.redo_2a}")
    log.info("=" * 70)

    # ── Load Step 1 captions from source ──────────────────────────────────
    captions_path = source_dir / "step1_captions.json"
    if not captions_path.exists():
        log.error(f"step1_captions.json not found in {source_dir}")
        sys.exit(1)

    log.info(f"Loading captions from {captions_path}")
    with open(captions_path, "r") as f:
        captions = json.load(f)
    log.info(f"  Loaded {len(captions)} captions")

    # ── Optionally load image_mapping for export ──────────────────────────
    mapping_path = source_dir / "image_mapping.json"
    image_mapping = None
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            image_mapping = json.load(f)

    ckpt = CheckpointManager(output_dir)

    # ── Copy or re-run Step 2a ────────────────────────────────────────────
    if not args.redo_2a:
        source_hooks = source_dir / "step2a_hooks.json"
        dest_hooks = output_dir / "step2a_hooks.json"
        if source_hooks.exists() and not dest_hooks.exists():
            log.info(f"Copying step2a_hooks.json from source (not re-extracting)")
            import shutil
            shutil.copy2(source_hooks, dest_hooks)

    # ── Load model ────────────────────────────────────────────────────────
    log.info("Loading model for text generation...")
    llm = TextLLM(
        model_name=args.vlm_model,
        tp=1,
        max_model_len=args.max_model_len,
        gpu_util=args.gpu_util,
        dtype=args.dtype,
        quantization=args.quantization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        log=log,
    )

    # ── Step 2a ───────────────────────────────────────────────────────────
    hooks = run_step2a(
        captions, ckpt, llm,
        step2a_system=track["step2a_system"],
        batch_size=args.batch_llm,
        checkpoint_interval=args.ckpt_llm,
        log=log,
    )

    # ── Step 2b ───────────────────────────────────────────────────────────
    cluster_def = run_step2b(
        hooks, ckpt, llm,
        num_clusters=args.num_clusters,
        top_n_hooks=args.top_hooks,
        criterion=track["criterion"],
        step2b_template=track["step2b_template"],
        step2b_role=track["step2b_role"],
        log=log,
    )

    # ── Step 3 ────────────────────────────────────────────────────────────
    assignments = run_step3(
        captions, cluster_def, ckpt, llm,
        step3_system_template=track["step3_system_template"],
        batch_size=args.batch_llm,
        checkpoint_interval=args.ckpt_llm,
        log=log,
    )

    # ── Export ─────────────────────────────────────────────────────────────
    export_results(captions, hooks, cluster_def, assignments, ckpt, image_mapping, log)

    llm.unload()
    log.info("=" * 70)
    log.info("Re-clustering complete.")
    log.info(f"Results in: {output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
