# ComfyUI — AI Gallery Saver

Custom SaveImage node for ComfyUI that writes **clean, authoritative metadata**
designed for the [AI Gallery](https://github.com/quzopl/ai-gallery) app.

Built on the ComfyUI v3 API with `Autogrow` — **unlimited image input slots**
(framework cap: 100).

## What it writes

Each PNG gets three `tEXt` chunks:

| Chunk | Format | Purpose |
|---|---|---|
| `ai_gallery_meta` | JSON | Authoritative — AI Gallery uses this first, no heuristics needed |
| `prompt` + `workflow` | JSON | Standard ComfyUI — drag the image back to ComfyUI to restore the workflow |
| `parameters` | A1111 text | CivitAI, stable-diffusion-webui, A1111 ecosystem |

## What it extracts from the workflow

Walks the execution graph (no keyword guessing):

- **prompt** — traces `KSampler.inputs.positive` → `CLIPTextEncode.text`
- **negative** — traces `KSampler.inputs.negative` → `CLIPTextEncode.text`
- **sampler, steps, cfg, seed** — from `KSampler` / `SamplerCustom`
- **model_name** — from `CheckpointLoader*` / `UNETLoader*` / `UnetLoader*`
- **loras** — from all LoRA loaders:
  - stock `LoraLoader`, `LoraLoaderModelOnly`
  - rgthree `Power Lora Loader` (dict slots, respects `on: false`)
  - LoRA Stack loaders (`lora_name_1`, `lora_name_2`, …)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/quzopl/comfyui-ai-gallery-saver.git
```

Restart ComfyUI. The node appears as **Save Image (AI Gallery)** in the
`image` category.

Requires ComfyUI with the v3 API (`comfy_api.latest`) — present in all
recent releases.

## Usage

Replace `SaveImage` in your workflow with **Save Image (AI Gallery)**. Connect
the `images` input, set `filename_prefix` (e.g., `AIGal`).

### Unlimited image inputs

The `images` input is an **autogrow slot** — the node UI adds a new empty
slot whenever you connect one. Plug in as many independent image batches as
you want (e.g., raw + refine + upscale + variations all in one workflow).

Each connected batch is saved separately:

- Slot 1 → `filename_prefix`
- Slot 2 → `filename_prefix_2`
- Slot 3 → `filename_prefix_3`
- … and so on

All slots share the same workflow metadata (same `ai_gallery_meta`,
`prompt`/`workflow`, and `parameters` chunks) since they're produced by the
same execution graph.

### Optional flags

- `embed_workflow` (default ON) — also embed standard ComfyUI chunks
- `embed_a1111` (default ON) — also embed A1111-compatible `parameters`

## Dependencies

None. Uses only Pillow and NumPy (both already shipped with ComfyUI).

## License

MIT
