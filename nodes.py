"""AI Gallery Saver — ComfyUI v3 node with unlimited IMAGE inputs.

Uses ComfyUI v3 API (`comfy_api.latest`) with `Autogrow` so the user can plug
in as many independent image batches as they want (framework caps at 100).

Each batch is saved with three PNG tEXt chunks:
  - ai_gallery_meta : clean JSON for AI Gallery (authoritative)
  - prompt + workflow : standard ComfyUI (round-trip)
  - parameters     : A1111-compatible (CivitAI / webui)

Per-slot filename auto-suffix: slot 1 uses `filename_prefix`, slot N>1 uses
`filename_prefix_N`.
"""
from __future__ import annotations

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


def _get_text_recursive(graph: dict, value: Any, depth: int = 0) -> str | None:
    """If `value` is a link [src_id, out_idx], walk to a CLIPTextEncode and read its text."""
    if depth > 5:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 2:
        node = graph.get(str(value[0]))
        if not node:
            return None
        inputs = node.get("inputs") or {}
        if "text" in inputs:
            txt = inputs["text"]
            if isinstance(txt, str):
                return txt
            return _get_text_recursive(graph, txt, depth + 1)
        for key in ("conditioning", "positive", "negative", "text_g", "text_l"):
            if key in inputs:
                t = _get_text_recursive(graph, inputs[key], depth + 1)
                if t:
                    return t
    return None


def extract_canonical(graph: dict, width: int, height: int) -> dict:
    """Walk the execution graph and pull out clean fields for AI Gallery."""
    out: dict = {
        "version": 1,
        "source": "comfyui-ai-gallery-saver",
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
    if meta.get("model_name"):
        kv.append(f"Model: {meta['model_name']}")
    if kv:
        parts.append(", ".join(kv))
    return "\n".join(parts)


# ---------- ComfyUI v3 node ----------

class AIGallerySaveImage(io.ComfyNode):
    """Save Image (AI Gallery) — saves PNG with rich metadata for AI Gallery.

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
            node_id="AIGallerySaveImage",
            display_name="Save Image (AI Gallery)",
            category="image",
            description=(
                "Saves images with clean AI Gallery JSON metadata + standard "
                "ComfyUI prompt/workflow + A1111 parameters. Unlimited image "
                "input slots (up to framework cap 100)."
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
        out: list[dict] = []
        for image in images:
            arr = 255.0 * image.cpu().numpy()
            img = PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            meta = extract_canonical(prompt or {}, w, h)

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

class AIGallerySaverExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AIGallerySaveImage]


async def comfy_entrypoint() -> AIGallerySaverExtension:
    return AIGallerySaverExtension()


# Backward-compat: keep v1 mappings empty (v3 extension handles registration).
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
