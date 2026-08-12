from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.appearance.hf_access import get_token, verify_hf_access


FLUX_DEPTH_BASE_LITE_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "vae/**",
]

FLUX_DEPTH_BASE_LITE_IGNORE_PATTERNS = [
    "transformer/**",
    "text_encoder_2/**",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "README.md",
    "LICENSE",
]

FLUX_DEPTH_NF4_ALLOW_PATTERNS = [
    "model_index.json",
    "transformer/**",
    "text_encoder_2/**",
]

FLUX_DEPTH_NF4_IGNORE_PATTERNS = [
    "assets/**",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "README.md",
    "LICENSE",
]


def _any_weight_file(path: Path) -> bool:
    if not path.exists():
        return False
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf")
    for pattern in patterns:
        if any(path.rglob(pattern)):
            return True
    return False


def _has_file(path: Path, rel: str) -> bool:
    target = path / rel
    return target.exists() and target.stat().st_size > 0


def _subfolder_complete(path: Path, subfolder: str, require_weights: bool = True) -> bool:
    folder = path / subfolder
    if not _has_file(folder, "config.json"):
        return False
    if require_weights and not _any_weight_file(folder):
        return False
    return True


def _diffusers_pipeline_complete(path: Path, required_subfolders: Iterable[str]) -> bool:
    path = Path(path)
    if not _has_file(path, "model_index.json"):
        return False
    for subfolder in required_subfolders:
        if not _subfolder_complete(path, subfolder, require_weights=True):
            return False
    return True


def _flux_depth_base_lite_complete(path: Path) -> bool:
    path = Path(path)
    # The base-lite directory intentionally excludes the full BF16 transformer
    # and text_encoder_2. They are supplied by the NF4 repo at load time.
    if not _has_file(path, "model_index.json"):
        return False
    for subfolder in ("vae", "text_encoder"):
        if not _subfolder_complete(path, subfolder, require_weights=True):
            return False
    # Tokenizers/scheduler may not contain model weights; config/tokenizer files
    # are enough for the pipeline to reconstruct them locally.
    tokenizer_like = [
        path / "tokenizer",
        path / "tokenizer_2",
        path / "scheduler",
    ]
    return all(folder.exists() and any(folder.rglob("*.json")) for folder in tokenizer_like)


def _flux_depth_nf4_complete(path: Path) -> bool:
    path = Path(path)
    return _subfolder_complete(path, "transformer", True) and _subfolder_complete(path, "text_encoder_2", True)


def _download_repo(
    repo_id: str,
    local_dir: str | Path,
    token: Optional[str],
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model-cache] downloading {repo_id} -> {local_dir}", flush=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return {"repo_id": repo_id, "local_dir": str(local_dir), "resolved_path": str(resolved), "downloaded": True}


def _ensure_flux_depth_nf4(config: Dict[str, Any], auth_config: Dict[str, Any]) -> Dict[str, Any]:
    base_path = Path(config.get("local_base_path", "models/flux1-depth-dev-base-lite"))
    nf4_path = Path(config.get("local_nf4_path", "models/flux1-depth-dev-nf4"))
    base_ready = _flux_depth_base_lite_complete(base_path)
    nf4_ready = _flux_depth_nf4_complete(nf4_path)

    report: Dict[str, Any] = {
        "backend": "flux1_depth_control_inpaint_nf4_16gb",
        "policy": "use local folders if complete; otherwise download required missing repositories",
        "base_model_id": config["base_model_id"],
        "quantized_model_id": config["quantized_model_id"],
        "local_base_path": str(base_path),
        "local_nf4_path": str(nf4_path),
        "base_already_present": base_ready,
        "nf4_already_present": nf4_ready,
        "actions": [],
    }

    if base_ready and nf4_ready:
        report["status"] = "ready_already_local"
        return report

    if config.get("preflight_check", True) or auth_config.get("preflight_check", True):
        verify_hf_access([config["base_model_id"], config["quantized_model_id"]], auth_config)
    token = get_token(auth_config)

    if not base_ready:
        report["actions"].append(
            _download_repo(
                config["base_model_id"],
                base_path,
                token,
                allow_patterns=FLUX_DEPTH_BASE_LITE_ALLOW_PATTERNS,
                ignore_patterns=FLUX_DEPTH_BASE_LITE_IGNORE_PATTERNS,
            )
        )
    else:
        report["actions"].append({"repo_id": config["base_model_id"], "local_dir": str(base_path), "downloaded": False})

    if not nf4_ready:
        report["actions"].append(
            _download_repo(
                config["quantized_model_id"],
                nf4_path,
                token,
                allow_patterns=FLUX_DEPTH_NF4_ALLOW_PATTERNS,
                ignore_patterns=FLUX_DEPTH_NF4_IGNORE_PATTERNS,
            )
        )
    else:
        report["actions"].append({"repo_id": config["quantized_model_id"], "local_dir": str(nf4_path), "downloaded": False})

    base_after = _flux_depth_base_lite_complete(base_path)
    nf4_after = _flux_depth_nf4_complete(nf4_path)
    report["base_ready_after"] = base_after
    report["nf4_ready_after"] = nf4_after
    if not (base_after and nf4_after):
        missing = []
        if not base_after:
            missing.append(str(base_path))
        if not nf4_after:
            missing.append(str(nf4_path))
        raise RuntimeError("Model download finished but local FLUX files are still incomplete: " + ", ".join(missing))

    report["status"] = "ready_after_download"
    return report




def _flux2_klein_complete(path: Path) -> bool:
    path = Path(path)
    if not _has_file(path, "model_index.json"):
        return False
    # Keep the completeness test tolerant to Diffusers snapshot layout changes:
    # the model index plus at least one weight file in each core component is enough.
    for subfolder in ("transformer", "text_encoder", "vae"):
        folder = path / subfolder
        if not folder.exists() or not _any_weight_file(folder):
            return False
    return True


def _ensure_flux2_klein(config: Dict[str, Any], auth_config: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(config.get("model_id", "black-forest-labs/FLUX.2-klein-4B"))
    local_path = Path(config.get("local_model_path", "models/flux2-klein-4b"))
    ready = _flux2_klein_complete(local_path)
    report: Dict[str, Any] = {
        "backend": "flux2_klein_4b_multiref_16gb",
        "model_id": model_id,
        "local_model_path": str(local_path),
        "already_present": ready,
        "actions": [],
    }
    if ready:
        report["status"] = "ready_already_local"
        return report
    if bool(config.get("auto_download", True)) is False:
        raise RuntimeError(f"FLUX.2 model files are incomplete and auto_download=false: {local_path}")
    token = get_token(auth_config)
    report["actions"].append(_download_repo(model_id, local_path, token))
    if not _flux2_klein_complete(local_path):
        raise RuntimeError(f"Model download finished but FLUX.2 local snapshot is incomplete: {local_path}")
    report["status"] = "ready_after_download"
    return report

def ensure_backend_models(
    backend_name: str,
    backend_config: Dict[str, Any],
    auth_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure the active FLUX backend is present locally.

    This is called inside every generation stage. Complete local folders skip
    downloads; missing components are downloaded and verified before inference.
    """
    auth_config = auth_config or {}
    if backend_name == "flux2_klein_4b_multiref_16gb":
        return _ensure_flux2_klein(backend_config, auth_config)
    if backend_name == "flux1_depth_control_inpaint_nf4_16gb":
        if bool(backend_config.get("auto_download", True)) is False:
            base = Path(backend_config.get("local_base_path", "models/flux1-depth-dev-base-lite"))
            nf4 = Path(backend_config.get("local_nf4_path", "models/flux1-depth-dev-nf4"))
            if not (_flux_depth_base_lite_complete(base) and _flux_depth_nf4_complete(nf4)):
                raise RuntimeError("FLUX.1 model files are incomplete and auto_download=false.")
            return {"backend": backend_name, "status": "ready_already_local", "actions": []}
        return _ensure_flux_depth_nf4(backend_config, auth_config)
    raise ValueError(f"No integrated real downloader for unsupported backend: {backend_name}")
