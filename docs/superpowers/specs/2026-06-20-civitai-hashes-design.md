# CivitAI-compatible metadata (resource hashes)

**Date:** 2026-06-20

## Goal

Make `SaveImageRichMetadata` write metadata that CivitAI recognizes and
**auto-links** to the checkpoint and LoRA resource pages. The node already emits
an A1111 `parameters` chunk (prompt, negative, Steps/Sampler/CFG/Seed/Size/Model);
the missing piece is **resource hashes**.

## What CivitAI needs

CivitAI matches resources by **AutoV2** hash = the first **12 hex chars** of the
file's full SHA256. It reads these from the `parameters` text chunk:

```
Model hash: <autov2>, Model: <name>
Lora hashes: "<lora1>: <autov2>, <lora2>: <autov2>"
Hashes: {"model": "<autov2>", "lora:<lora1>": "<autov2>", ...}
```

## Scope

- Hash the **checkpoint/UNet** (search order: `checkpoints` → `diffusion_models`
  → `unet`) and **every LoRA** (`loras`).
- Out of scope: VAE, embeddings, a separate metadata chunk (we extend the
  existing `parameters`), changes to the `prompt`/`workflow` chunks.

## Design

`comfyui-save-image-rich-metadata/nodes.py` only.

### Hashing + cache (IO)
- `_sha256_file(path)` — stream the file in chunks, return full hex digest.
- `_hash_file(path, cache)` — cache layer keyed by `f"{abspath}|{size}|{mtime}"`;
  on hit return cached digest, else compute + store. Pure given a `cache` dict
  and a real file (testable on a temp file).
- Persistent cache: a JSON file `.hash_cache.json` in the plugin directory,
  loaded once and saved after new hashes. Never writes next to model files.
- `_resolve_model_path(name, folders)` — `folder_paths.get_full_path` across the
  candidate folders; returns the first existing path or None.

### Augment metadata (IO)
- `_augment_with_hashes(meta)` — resolves the checkpoint and each LoRA to a path,
  hashes it (cached), and fills:
  - `meta["model_hash"]` = AutoV2 of the checkpoint (or None),
  - `lora["hash"]` = AutoV2 per lora row (or None).
  Missing files → hash stays None (skipped in output).

### Pure helpers
- `_autov2(sha_hex)` → `sha_hex[:12]`.
- `_lora_key(name)` → basename without extension (the name used in `<lora:…>`).
- `_lora_hashes_str(loras)` → `'name: h, name2: h2'` (only rows with a hash).
- `_hashes_json(meta)` → `{"model": h, "lora:name": h, …}` (only present hashes).
- `_format_a1111_parameters(meta)` — extended to append `Model hash`,
  `Lora hashes`, and `Hashes` when the corresponding hashes exist; unchanged when
  they don't.

### Prompt
The positive prompt stays first in `parameters` (A1111/CivitAI requirement),
sourced from `meta["prompt"]` via the existing extraction chain (incl. the
Ideogram bbox editor and Power Lora). LoRAs remain appended as `<lora:name:str>`.

### ai_gallery_meta
Add `model_hash` and per-lora `hash` to the clean JSON for consistency; the rest
unchanged.

## Testing

Pure helpers unit-tested without ComfyUI: `_autov2`, `_lora_key`,
`_lora_hashes_str`, `_hashes_json`, and `_format_a1111_parameters` with and
without hashes. `_hash_file` tested against a small temp file, including a
cache hit and size/mtime-based invalidation. Torch/folder_paths stay lazy.
