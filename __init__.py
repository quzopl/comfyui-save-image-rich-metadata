"""ComfyUI custom node: Save Image (Rich Metadata).

A SaveImage with unlimited image input slots and three PNG tEXt chunks:
  - ai_gallery_meta : clean JSON (authoritative; consumed by AI Gallery app)
  - prompt + workflow : standard ComfyUI (round-trip)
  - parameters     : A1111-compatible (CivitAI / webui)

Single Autogrow `images` input — connect as many batches as needed
(framework cap: 100). Per-slot filename auto-suffix.
"""
from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
