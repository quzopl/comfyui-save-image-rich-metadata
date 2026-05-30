"""ComfyUI custom node: AI Gallery Saver (v3 API with unlimited image inputs).

Saves images with three PNG tEXt chunks:
  - ai_gallery_meta : clean JSON for AI Gallery (authoritative, no heuristics)
  - prompt + workflow : standard ComfyUI (round-trip)
  - parameters     : A1111-compatible (CivitAI / webui)

Single Autogrow `images` input — connect as many batches as needed
(framework cap: 100). Per-slot filename auto-suffix.
"""
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    comfy_entrypoint,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
