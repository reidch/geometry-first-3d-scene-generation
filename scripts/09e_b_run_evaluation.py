#!/usr/bin/env python
from __future__ import annotations
import argparse,subprocess,sys,traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.core.logging_utils import setup_step_logger,status
from src.evaluation.evaluation_sampler import freeze_evaluation_manifest
from src.evaluation.evaluation_metrics import compute_metrics
from src.io.json_io import load_json,save_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--evaluation_config',default='configs/evaluation_pipeline.json');ap.add_argument('--gaussian_config',default='configs/gaussian_pipeline.json');args=ap.parse_args();out=Path(args.out);cfg=load_json(args.evaluation_config);stage=out/str(cfg.get('stage_root','09e_evaluation'));setup_step_logger(Path('logs')/out.name/'09e_b_run_evaluation.log')
    try:
        status('[09E-B] Freezing curated cameras, rendering final 3DGS and computing the 11 locked evaluation metrics...')
        manifest=freeze_evaluation_manifest(stage); config_path=Path((out/'09_gaussian_splat'/'nerfstudio_config_path.txt').read_text().strip()); gcfg=load_json(args.gaussian_config); env=str(dict(gcfg.get('training',{})).get('conda_env','worldmesh-nerfstudio')); renders=stage/'B_renders'; metrics=stage/'B_metrics'; renders.mkdir(parents=True,exist_ok=True);metrics.mkdir(parents=True,exist_ok=True)
        # Guard the Stage09 runtime before evaluation so image-quality dependencies
        # cannot alter the Nerfstudio environment's pinned timm/transformers versions.
        subprocess.run(['conda','run','-n',env,'--no-capture-output','python','scripts/09e_guard_nerfstudio_runtime.py','--repair'],check=True)
        subprocess.run(['conda','run','-n',env,'--no-capture-output','python','scripts/09e_render_nerfstudio_evaluation.py','--load-config',str(config_path),'--manifest',str(stage/'B_frozen_evaluation_manifest.json'),'--output',str(renders)],check=True)

        # Image-quality packages live in an isolated project-local venv. It inherits only
        # the working Stage09 torch/CUDA installation via system-site-packages; package
        # upgrades such as timm/transformers remain local to the venv and cannot mutate
        # worldmesh-nerfstudio.
        qcfg=dict(cfg.get('image_quality',{}))
        quality_python_file=stage/'B_quality_python_path.txt'
        ensure=[sys.executable,'scripts/09e_ensure_quality_runtime.py',
                '--base-conda-env',env,
                '--venv-dir',str(qcfg.get('isolated_venv_dir','.runtime/stage09e_quality')),
                '--requirements',str(qcfg.get('requirements_file','requirements-stage09e-quality.txt')),
                '--write-python-path',str(quality_python_file)]
        if bool(qcfg.get('auto_install_missing_packages',True)): ensure.append('--auto-install')
        subprocess.run(ensure,check=True)
        quality_python=quality_python_file.read_text(encoding='utf-8').strip()
        subprocess.run([quality_python,'scripts/09e_compute_image_quality.py','--manifest',str(stage/'B_frozen_evaluation_manifest.json'),'--renders',str(renders),'--output',str(metrics),'--config',args.evaluation_config],check=True)
        summary=compute_metrics(stage,cfg); save_json({'status':'ok','novel_view_count':manifest['novel_view_count'],'training_replay_count':manifest['training_replay_count'],'reprojection_pair_count':manifest['reprojection_pair_count'],'locked_metric_count':11,'summary':summary},stage/'B_stage_report.json')
        artifact=ArtifactIndex(scene_id=out.name,step='09e_b_evaluation')
        artifact.add('metrics',metrics)
        artifact.add('renders',renders)
        artifact.add('frozen_manifest',stage/'B_frozen_evaluation_manifest.json')
        artifact.add('frozen_manifest_sha256',stage/'B_frozen_evaluation_manifest.sha256')
        artifact.add('stage_report',stage/'B_stage_report.json')
        artifact.save(metrics/'artifact_index.json')
        mark_done(metrics);status(f"[09E-B] Done. {manifest['novel_view_count']} curated novel views, {manifest['training_replay_count']} training replay views; report: {metrics/'summary.json'}")
    except Exception:
        stage.mkdir(parents=True,exist_ok=True);(stage/'.B_failed').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
