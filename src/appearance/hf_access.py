from __future__ import annotations
import os
from pathlib import Path

class HuggingFaceAccessError(RuntimeError):
    pass

def get_token(config=None):
    config = config or {}
    env_name = config.get("token_env_var", "HF_TOKEN")
    token = os.environ.get(env_name)
    if token:
        return token

    try:
        from huggingface_hub import get_token as hf_get_token
        return hf_get_token()
    except Exception:
        return None

def verify_hf_access(repo_ids, auth_config=None):
    auth_config = auth_config or {}
    token = get_token(auth_config)

    if not token:
        raise HuggingFaceAccessError(
            "No Hugging Face token is available. Run `hf auth login` in the same "
            "environment used for the generation stages, or set HF_TOKEN."
        )

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami(token=token)
    except Exception as exc:
        raise HuggingFaceAccessError(
            "The saved Hugging Face token is invalid or unavailable to this environment. "
            "Run `hf auth login` again."
        ) from exc

    inaccessible = []
    for repo_id in repo_ids:
        try:
            api.model_info(repo_id=repo_id, token=token)
        except Exception as exc:
            inaccessible.append((repo_id, str(exc)))

    if inaccessible:
        lines = [
            "Hugging Face access preflight failed.",
            f"Authenticated account: {user.get('name', 'unknown')}",
            "",
            "The following repositories cannot be read:"
        ]
        for repo_id, message in inaccessible:
            lines.append(f"- {repo_id}: {message}")
        lines.extend([
            "",
            "Fix:",
            "1. Open the FLUX.1-Depth-dev model page while logged in and accept/request access.",
            "2. Create a new Read token, or a fine-grained token with:",
            "   'Read access to contents of all public gated repositories you can access'.",
            "3. In the same conda environment, run: hf auth login",
            "4. Rerun the requested stage, for example: bash run_stage08.sh outputs/my_scene",
        ])
        raise HuggingFaceAccessError("\n".join(lines))

    return {
        "username": user.get("name"),
        "repo_ids": list(repo_ids),
        "token_source": "environment_or_hf_cache",
    }

def resolve_model_source(local_path, repo_id, require_local=False):
    local_path = Path(local_path)
    model_index = local_path / "model_index.json"
    config_file = local_path / "config.json"

    if local_path.exists() and (model_index.exists() or config_file.exists()):
        return str(local_path), True

    # Some lightweight/quantized repos are intentionally downloaded only as
    # component subfolders, e.g. transformer/ and text_encoder_2/ for FLUX NF4.
    component_configs = (
        local_path / "transformer" / "config.json",
        local_path / "text_encoder_2" / "config.json",
        local_path / "controlnet" / "config.json",
    )
    if local_path.exists() and any(path.exists() for path in component_configs):
        return str(local_path), True

    if require_local:
        raise FileNotFoundError(
            f"Local model is incomplete or missing: {local_path}. "
            "Run the corresponding stage script; the active generation stage downloads missing configured FLUX models automatically."
        )

    return repo_id, False
