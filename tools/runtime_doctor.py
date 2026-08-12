#!/usr/bin/env python
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io.json_io import load_json
from tools.audit_blender_dependency_boundary import audit as audit_blender_dependency_boundary

def main():
    config=load_json(ROOT/'configs'/'asset_pipeline.json')['pixal3d']
    repo=Path(os.environ.get('PIXAL3D_REPO', ROOT/config.get('repo_dir','external/Pixal3D'))).expanduser().resolve()
    runtime=os.environ.get('PIXAL3D_PYTHON') or str(config.get('runtime_python','python'))
    if not (repo/'inference.py').exists(): raise SystemExit(f'Pixal3D repo is missing: {repo}')
    probe=subprocess.run([runtime,'-c','import torch; print(torch.__version__); print(torch.cuda.is_available())'],text=True,capture_output=True)
    if probe.returncode!=0: raise SystemExit('Pixal3D runtime probe failed:\n'+probe.stderr)
    blender=os.environ.get('BLENDER_BIN') or shutil.which('blender')
    if not blender: raise SystemExit('Blender executable not found. Set BLENDER_BIN.')
    dep=audit_blender_dependency_boundary(ROOT)
    if dep['status']!='ok': raise SystemExit(json.dumps(dep['findings'],indent=2))
    print(json.dumps({'status':'ok','pixal3d_python':runtime,'pixal3d_repo':str(repo),'runtime_probe':probe.stdout.strip(),'blender':blender,'blender_dependency_entrypoints_checked':dep['entrypoints']},indent=2))
if __name__=='__main__': main()
