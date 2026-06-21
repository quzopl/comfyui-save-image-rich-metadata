"""Save Image (Rich Metadata) — ComfyUI v3 node with unlimited IMAGE inputs.

Uses ComfyUI v3 API (`comfy_api.latest`) with `Autogrow` so the user can plug
in as many independent image batches as they want (framework caps at 100).

Each batch is saved with three PNG tEXt chunks:
  - ai_gallery_meta : clean JSON (authoritative; consumed by AI Gallery app)
  - prompt + workflow : standard ComfyUI (round-trip)
  - parameters     : A1111-compatible (CivitAI / webui)

Per-slot filename auto-suffix: slot 1 uses `filename_prefix`, slot N>1 uses
`filename_prefix_N`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io


# ---------- canonical metadata extraction ----------

def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve(graph: dict, value: Any) -> dict | None:
    """If `value` is a link [src_id, out_idx], return the source node dict."""
    if isinstance(value, list) and len(value) == 2:
        node = graph.get(str(value[0]))
        if isinstance(node, dict):
            return node
    return None


def _parse_json_list(s: Any) -> list:
    if isinstance(s, str) and s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _ig4_dumps(v: Any, lvl: int = 0) -> str:
    """Mirror Ideogram4PromptBuilderKJ's serializer: indent=4 but scalar arrays inline."""
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_ig4_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _ig4_dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _ig4_dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _ig4_norm_bbox(box: dict) -> list:
    def c(v):
        return max(0, min(1000, round(v * 1000)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _reconstruct_ideogram4(inputs: dict) -> str:
    """Rebuild the caption JSON that Ideogram4PromptBuilderKJ emits at runtime.

    The builder computes its prompt string from its widget inputs, so it is not
    present as a static `text` value anywhere in the graph. Its inputs *are*
    in the graph, so we reproduce the same assembly to recover the real prompt.
    """
    def s(k: str) -> str:
        v = inputs.get(k, "")
        return v if isinstance(v, str) else ""

    caption: dict = {}
    if s("high_level_description").strip():
        caption["high_level_description"] = s("high_level_description")

    kind = s("style") or "none"
    if kind != "none":
        sd: dict = {"aesthetics": s("aesthetics"), "lighting": s("lighting")}
        if kind == "photo":
            sd["photo"] = s("style.photo")
            sd["medium"] = s("medium")
        else:
            sd["medium"] = s("medium")
            sd["art_style"] = s("style.art_style")
        palette = [c.upper() for c in _parse_json_list(s("style_palette_data")) if c]
        if palette:
            sd["color_palette"] = palette
        caption["style_description"] = sd

    elements = []
    for box in _parse_json_list(s("elements_data")):
        if not isinstance(box, dict):
            continue
        etype = "text" if box.get("type") == "text" else "obj"
        elem: dict = {"type": etype}
        if not box.get("nobbox"):
            elem["bbox"] = _ig4_norm_bbox(box)
        if etype == "text":
            elem["text"] = box.get("text", "")
        elem["desc"] = box.get("desc", "")
        pal = [c.upper() for c in (box.get("palette") or []) if c]
        if pal:
            elem["color_palette"] = pal[:5]
        elements.append(elem)

    caption["compositional_deconstruction"] = {"background": s("background"), "elements": elements}
    return _ig4_dumps(caption)


def _get_text_recursive(graph: dict, value: Any, depth: int = 0) -> str | None:
    """If `value` is a link [src_id, out_idx], walk to a text source and read it.

    Handles plain CLIPTextEncode `text` strings, nested conditioning, and
    runtime prompt builders (Ideogram 4) whose text isn't a static input.
    `ConditioningZeroOut` is treated as empty so a zeroed negative branch
    doesn't echo the positive prompt it wraps.
    """
    if depth > 6:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 2:
        node = graph.get(str(value[0]))
        if not node:
            return None
        ct = node.get("class_type", "")
        if ct == "ConditioningZeroOut":
            return None
        if ct == "Ideogram4PromptBuilderKJ":
            return _reconstruct_ideogram4(node.get("inputs") or {})
        if ct == "Ideogram4BboxEditor":
            # The bbox editor assembles its caption in the frontend and stores
            # it in the `caption_json` widget, so that string *is* the prompt
            # (the node only re-applies the target size at runtime).
            cj = (node.get("inputs") or {}).get("caption_json")
            if isinstance(cj, str) and cj.strip() and cj.strip() != "{}":
                return cj
            return None
        inputs = node.get("inputs") or {}
        # Boolean routers (ComfySwitchNode, ImpactSwitch, etc.): follow the
        # active branch. `switch` may be a literal bool or a link; when it's
        # not a plain bool we can't evaluate it, so try both branches.
        if "on_true" in inputs or "on_false" in inputs:
            sw = inputs.get("switch")
            order = (("on_true",) if sw is True else
                     ("on_false",) if sw is False else
                     ("on_true", "on_false"))
            for b in order:
                if b in inputs:
                    t = _get_text_recursive(graph, inputs[b], depth + 1)
                    if t:
                        return t
            return None
        if "text" in inputs:
            txt = inputs["text"]
            if isinstance(txt, str):
                return txt
            return _get_text_recursive(graph, txt, depth + 1)
        # Primitive string nodes (PrimitiveStringMultiline, PrimitiveString,
        # String Literal, …) carry their text in a `value` field.
        if "value" in inputs:
            val = inputs["value"]
            if isinstance(val, str):
                return val
            t = _get_text_recursive(graph, val, depth + 1)
            if t:
                return t
        for key in ("conditioning", "positive", "negative", "text_g", "text_l"):
            if key in inputs:
                t = _get_text_recursive(graph, inputs[key], depth + 1)
                if t:
                    return t
    return None


def extract_canonical(graph: dict, width: int, height: int) -> dict:
    """Walk the execution graph and pull out clean fields."""
    out: dict = {
        "version": 1,
        "source": "comfyui-save-image-rich-metadata",
        "prompt": None,
        "negative": None,
        "model_name": None,
        "sampler": None,
        "steps": None,
        "cfg": None,
        "seed": None,
        "loras": [],
        "width": width,
        "height": height,
        "generated_at": int(time.time()),
    }
    if not isinstance(graph, dict):
        return out

    sampler_found = False
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.get("inputs") or {}

        # sampler
        if not sampler_found and ("KSampler" in ct or "SamplerCustom" in ct):
            sampler_found = True
            pos = inputs.get("positive")
            neg = inputs.get("negative")
            # Custom samplers (SamplerCustom/Advanced) route conditioning through a
            # guider node instead of exposing positive/negative directly.
            if pos is None and neg is None and "guider" in inputs:
                guider = _resolve(graph, inputs.get("guider"))
                if guider:
                    gin = guider.get("inputs") or {}
                    pos = gin.get("positive")
                    neg = gin.get("negative")
                    out["cfg"] = out["cfg"] or _float_or_none(gin.get("cfg"))
            if pos is not None:
                out["prompt"] = _get_text_recursive(graph, pos) or out["prompt"]
            if neg is not None:
                out["negative"] = _get_text_recursive(graph, neg) or out["negative"]
            if isinstance(inputs.get("sampler_name"), str):
                out["sampler"] = inputs["sampler_name"]
            out["steps"] = out["steps"] or _int_or_none(inputs.get("steps"))
            out["cfg"] = out["cfg"] or _float_or_none(inputs.get("cfg"))
            seed = inputs.get("seed", inputs.get("noise_seed"))
            out["seed"] = out["seed"] or _int_or_none(seed)

        # checkpoint / unet
        if (
            "Checkpoint" in ct or "UNetLoader" in ct or "UNETLoader" in ct
            or "ModelLoader" in ct
        ):
            for k in ("ckpt_name", "unet_name", "model_name", "model"):
                v = inputs.get(k)
                if isinstance(v, str) and not out["model_name"]:
                    out["model_name"] = v
                    break

        # LoRA — stock loaders
        if ("Lora" in ct or "LoRA" in ct) and "Stack" not in ct and "Power" not in ct:
            name = inputs.get("lora_name") or inputs.get("name")
            strength = _float_or_none(inputs.get("strength_model") or inputs.get("strength"))
            if isinstance(name, str):
                out["loras"].append({"name": name, "strength": strength})

        # LoRA — rgthree Power Lora Loader (slot dicts)
        if "Power Lora Loader" in ct or ct == "PowerLoraLoader (rgthree)":
            for k, v in inputs.items():
                if k.startswith("lora_") and isinstance(v, dict):
                    if v.get("on") and isinstance(v.get("lora"), str):
                        out["loras"].append({
                            "name": v["lora"],
                            "strength": _float_or_none(v.get("strength")),
                        })

        # LoRA — stack loaders (lora_name_1, lora_name_2, ...)
        if "Lora" in ct and "Stack" in ct:
            for i in range(1, 50):
                name = inputs.get(f"lora_name_{i}") or inputs.get(f"lora_{i}_name")
                if isinstance(name, str) and name and name.lower() != "none":
                    strength = _float_or_none(
                        inputs.get(f"strength_{i}")
                        or inputs.get(f"lora_wt_{i}")
                        or inputs.get(f"model_str_{i}")
                    )
                    out["loras"].append({"name": name, "strength": strength})

    return out


# ---------- CivitAI resource hashes (AutoV2 = first 12 hex of SHA256) --------

_HASH_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hash_cache.json")
_HASH_CACHE: dict | None = None


def _autov2(sha_hex: Any) -> str:
    return sha_hex[:12] if isinstance(sha_hex, str) else ""


def _lora_key(name: str) -> str:
    """The lora name as used in <lora:NAME:..> / Hashes: basename, no extension."""
    base = os.path.basename(str(name).replace("\\", "/"))
    return os.path.splitext(base)[0]


def _lora_hashes_str(loras: list) -> str:
    parts = [f"{_lora_key(l['name'])}: {l['hash']}" for l in (loras or []) if l.get("hash")]
    return ", ".join(parts)


def _hashes_json(meta: dict) -> dict:
    out: dict = {}
    if meta.get("model_hash"):
        out["model"] = meta["model_hash"]
    for l in meta.get("loras") or []:
        if l.get("hash"):
            out[f"lora:{_lora_key(l['name'])}"] = l["hash"]
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file(path: str, cache: dict) -> str | None:
    """Full SHA256 of `path`, cached by path+size+mtime. None if missing/unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    if key in cache:
        return cache[key]
    try:
        digest = _sha256_file(path)
    except OSError:
        return None
    cache[key] = digest
    return digest


def _load_hash_cache() -> dict:
    global _HASH_CACHE
    if _HASH_CACHE is None:
        try:
            with open(_HASH_CACHE_FILE, "r", encoding="utf-8") as f:
                _HASH_CACHE = json.load(f)
        except (OSError, ValueError):
            _HASH_CACHE = {}
    return _HASH_CACHE


def _save_hash_cache() -> None:
    try:
        with open(_HASH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_HASH_CACHE or {}, f)
    except OSError:
        pass


def _resolve_model_path(name: str, folders: tuple) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    for folder in folders:
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p and os.path.exists(p):
            return p
    return None


def _augment_with_hashes(meta: dict) -> dict:
    """Fill meta['model_hash'] and per-lora 'hash' (AutoV2), using a persistent
    cache so big model files are hashed only once."""
    cache = _load_hash_cache()
    before = len(cache)

    mp = _resolve_model_path(meta.get("model_name"), ("checkpoints", "diffusion_models", "unet"))
    if mp:
        full = _hash_file(mp, cache)
        if full:
            meta["model_hash"] = _autov2(full)

    for lora in meta.get("loras") or []:
        lp = _resolve_model_path(lora.get("name"), ("loras",))
        if lp:
            full = _hash_file(lp, cache)
            if full:
                lora["hash"] = _autov2(full)

    if len(cache) != before:
        _save_hash_cache()
    return meta


# ---------- A1111 parameters formatting ----------

def _format_a1111_parameters(meta: dict) -> str:
    parts: list[str] = []
    pos = meta.get("prompt") or ""
    for lora in meta.get("loras") or []:
        s = lora.get("strength")
        if s is None:
            s = 1.0
        pos = pos.rstrip() + f" <lora:{lora['name']}:{s}>"
    parts.append(pos.strip())
    if meta.get("negative"):
        parts.append(f"Negative prompt: {meta['negative']}")
    kv: list[str] = []
    if meta.get("steps") is not None:
        kv.append(f"Steps: {meta['steps']}")
    if meta.get("sampler"):
        kv.append(f"Sampler: {meta['sampler']}")
    if meta.get("cfg") is not None:
        kv.append(f"CFG scale: {meta['cfg']}")
    if meta.get("seed") is not None:
        kv.append(f"Seed: {meta['seed']}")
    if meta.get("width") and meta.get("height"):
        kv.append(f"Size: {meta['width']}x{meta['height']}")
    if meta.get("model_hash"):
        kv.append(f"Model hash: {meta['model_hash']}")
    if meta.get("model_name"):
        kv.append(f"Model: {meta['model_name']}")
    lora_hashes = _lora_hashes_str(meta.get("loras") or [])
    if lora_hashes:
        kv.append(f'Lora hashes: "{lora_hashes}"')
    # `Hashes` is JSON and must be LAST so its commas aren't read as kv separators.
    hashes = _hashes_json(meta)
    if hashes:
        kv.append("Hashes: " + json.dumps(hashes))
    if kv:
        parts.append(", ".join(kv))
    return "\n".join(parts)


# ---------- ComfyUI v3 node ----------

class SaveImageRichMetadata(io.ComfyNode):
    """Save Image (Rich Metadata) — saves PNG with rich, multi-format metadata.

    Has one Autogrow image input — connect as many image batches as you need
    (framework cap: 100). Each connected batch is saved with the shared
    metadata extracted from the workflow; per-slot filenames auto-suffix
    `_2`, `_3`, ... onto the main `filename_prefix`.
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = _io.Autogrow.TemplatePrefix(
            input=io.Image.Input("img"),
            prefix="img_",
            min=1,
            max=100,
        )
        return io.Schema(
            node_id="SaveImageRichMetadata",
            display_name="Save Image (Rich Metadata)",
            category="image",
            description=(
                "Saves images with clean canonical JSON metadata + standard "
                "ComfyUI prompt/workflow + A1111-compatible parameters "
                "(CivitAI-ready). Unlimited image input slots (up to "
                "framework cap 100)."
            ),
            inputs=[
                io.String.Input(
                    "filename_prefix",
                    default="AIGal",
                    tooltip=(
                        "Filename prefix for the first image batch. Extra "
                        "batches auto-suffix '_2', '_3', ..."
                    ),
                ),
                io.Boolean.Input(
                    "embed_workflow",
                    default=True,
                    tooltip="Also embed standard ComfyUI 'prompt'+'workflow' chunks.",
                    optional=True,
                ),
                io.Boolean.Input(
                    "embed_a1111",
                    default=True,
                    tooltip="Also embed A1111-compatible 'parameters' chunk.",
                    optional=True,
                ),
                _io.Autogrow.Input("images", template=autogrow_template),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        filename_prefix: str,
        embed_workflow: bool,
        embed_a1111: bool,
        images: _io.Autogrow.Type,
    ) -> io.NodeOutput:
        # images is dict {img_0: batch_tensor, img_1: batch_tensor, ...}
        prompt = cls.hidden.prompt if cls.hidden else None
        extra_pnginfo = cls.hidden.extra_pnginfo if cls.hidden else None

        all_results: list[dict] = []
        slot_idx = 0
        for slot_name in sorted(images.keys()):
            batch = images.get(slot_name)
            if batch is None or len(batch) == 0:
                slot_idx += 1
                continue
            slot_prefix = filename_prefix if slot_idx == 0 else f"{filename_prefix}_{slot_idx + 1}"
            results = cls._save_batch(
                batch, slot_prefix,
                embed_workflow=embed_workflow, embed_a1111=embed_a1111,
                prompt=prompt, extra_pnginfo=extra_pnginfo,
            )
            all_results.extend(results)
            slot_idx += 1

        return io.NodeOutput(ui={"images": all_results})

    @classmethod
    def _save_batch(
        cls,
        images,
        filename_prefix: str,
        *,
        embed_workflow: bool,
        embed_a1111: bool,
        prompt: dict | None,
        extra_pnginfo: dict | None,
    ) -> list[dict]:
        output_dir = folder_paths.get_output_directory()
        h, w = images[0].shape[0], images[0].shape[1]
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir, w, h)
        )
        # Same for every image in the batch; hash resource files once (cached).
        meta = extract_canonical(prompt or {}, w, h)
        _augment_with_hashes(meta)

        out: list[dict] = []
        for image in images:
            arr = 255.0 * image.cpu().numpy()
            img = PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            png_info = PngInfo()
            png_info.add_text("ai_gallery_meta", json.dumps(meta, ensure_ascii=False))

            if embed_workflow:
                if prompt is not None:
                    png_info.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        png_info.add_text(k, json.dumps(v))

            if embed_a1111:
                params_str = _format_a1111_parameters(meta)
                if params_str.strip():
                    png_info.add_text("parameters", params_str)

            file = f"{filename}_{counter:05d}_.png"
            path = os.path.join(full_output_folder, file)
            img.save(path, pnginfo=png_info, compress_level=4)
            out.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        return out


# ---------- v3 extension entrypoint ----------

class SaveImageRichMetadataExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImageRichMetadata]


async def comfy_entrypoint() -> SaveImageRichMetadataExtension:
    return SaveImageRichMetadataExtension()


