import sys
import types
import hashlib
import importlib.util
from pathlib import Path

# Stub the ComfyUI-only imports so nodes.py loads standalone; the helpers under
# test only use json/os/hashlib.
for name in ("folder_paths", "numpy"):
    sys.modules.setdefault(name, types.ModuleType(name))
pil = types.ModuleType("PIL"); pil.Image = types.SimpleNamespace(); sys.modules.setdefault("PIL", pil)
pilp = types.ModuleType("PIL.PngImagePlugin"); pilp.PngInfo = object; sys.modules.setdefault("PIL.PngImagePlugin", pilp)
ca = types.ModuleType("comfy_api"); cal = types.ModuleType("comfy_api.latest")
cal.ComfyExtension = object
cal.io = types.SimpleNamespace(ComfyNode=object)
cal._io = types.SimpleNamespace()
sys.modules.setdefault("comfy_api", ca); sys.modules.setdefault("comfy_api.latest", cal)
te = types.ModuleType("typing_extensions"); te.override = lambda f: f; sys.modules.setdefault("typing_extensions", te)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("srm_nodes", ROOT / "nodes.py")
srm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srm)


# ---- AutoV2 ----------------------------------------------------------------
def test_autov2_first_12():
    assert srm._autov2("0123456789abcdef0123") == "0123456789ab"
    assert srm._autov2("") == ""
    assert srm._autov2(None) == ""


# ---- lora key + hashes string ---------------------------------------------
def test_lora_key_basename_no_ext():
    assert srm._lora_key("ideogram/plener.safetensors") == "plener"
    assert srm._lora_key("detail.safetensors") == "detail"
    assert srm._lora_key("sub\\dir\\x.pt") == "x"


def test_lora_hashes_str_skips_unhashed():
    loras = [
        {"name": "ideogram/plener.safetensors", "hash": "aaaaaaaaaaaa"},
        {"name": "no_hash.safetensors"},
        {"name": "b.safetensors", "hash": "bbbbbbbbbbbb"},
    ]
    assert srm._lora_hashes_str(loras) == "plener: aaaaaaaaaaaa, b: bbbbbbbbbbbb"
    assert srm._lora_hashes_str([]) == ""


def test_hashes_json():
    meta = {"model_hash": "deadbeef0000",
            "loras": [{"name": "x/plener.safetensors", "hash": "aaaaaaaaaaaa"},
                      {"name": "no.safetensors"}]}
    assert srm._hashes_json(meta) == {"model": "deadbeef0000", "lora:plener": "aaaaaaaaaaaa"}


def test_hashes_json_empty_when_no_hashes():
    assert srm._hashes_json({"loras": [{"name": "a"}]}) == {}


# ---- A1111 parameters with / without hashes --------------------------------
def _base_meta():
    return {"prompt": "a cat", "negative": "", "steps": 20, "sampler": "euler",
            "cfg": 7.0, "seed": 42, "width": 512, "height": 512, "model_name": "realism",
            "loras": []}


def test_parameters_without_hashes_has_no_hash_fields():
    out = srm._format_a1111_parameters(_base_meta())
    assert out.startswith("a cat")
    assert "Model hash:" not in out and "Hashes:" not in out and "Lora hashes:" not in out


def test_parameters_with_hashes_includes_civitai_fields():
    meta = _base_meta()
    meta["model_hash"] = "deadbeef0000"
    meta["loras"] = [{"name": "ideogram/plener.safetensors", "strength": 0.5, "hash": "aaaaaaaaaaaa"}]
    out = srm._format_a1111_parameters(meta)
    assert out.startswith("a cat")
    assert "<lora:ideogram/plener.safetensors:0.5>" in out
    assert "Model hash: deadbeef0000" in out
    assert 'Lora hashes: "plener: aaaaaaaaaaaa"' in out
    assert '"model": "deadbeef0000"' in out and '"lora:plener": "aaaaaaaaaaaa"' in out
    # Hashes is the last field (its JSON commas must not be parsed as kv separators)
    assert out.rstrip().rfind("Hashes:") > out.rfind("Model hash:")


# ---- file hashing + cache --------------------------------------------------
def test_hash_file_matches_hashlib_and_caches(tmp_path):
    p = tmp_path / "model.bin"
    data = b"hello world" * 1000
    p.write_bytes(data)
    cache = {}
    h1 = srm._hash_file(str(p), cache)
    assert h1 == hashlib.sha256(data).hexdigest()
    assert len(cache) == 1
    # poison the cache to prove the 2nd call is a cache hit, not a recompute
    for k in list(cache):
        cache[k] = "POISONED"
    assert srm._hash_file(str(p), cache) == "POISONED"


def test_hash_file_invalidates_on_change(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"aaaa")
    cache = {}
    srm._hash_file(str(p), cache)
    for k in list(cache):
        cache[k] = "POISONED"
    p.write_bytes(b"aaaabbbb")  # size + mtime change -> new key -> recompute
    h = srm._hash_file(str(p), cache)
    assert h == hashlib.sha256(b"aaaabbbb").hexdigest()


def test_hash_file_missing_returns_none():
    assert srm._hash_file("/no/such/file.bin", {}) is None
