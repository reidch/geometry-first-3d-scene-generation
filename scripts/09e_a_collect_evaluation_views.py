#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.core.logging_utils import setup_step_logger, status
from src.evaluation.evaluation_sampler import collect_evaluation_candidates
from src.io.json_io import load_json
from src.pipeline.artifact_index import ArtifactIndex
from src.pipeline.resume import mark_done

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True); ap.add_argument('--scene_json',required=True)
    ap.add_argument('--camera_config',default='configs/cameras.json'); ap.add_argument('--refinement_config',default='configs/refinement_pipeline.json'); ap.add_argument('--evaluation_config',default='configs/evaluation_pipeline.json')
    args=ap.parse_args(); out=Path(args.out); cfg=load_json(args.evaluation_config); stage=out/str(cfg.get('stage_root','09e_evaluation'))
    setup_step_logger(Path('logs')/out.name/'09e_a_collect_evaluation_views.log')
    status('[09E-A] Collecting geometry-only short-trajectory, close-up and object-rotation evaluation views...')
    try:
        report=collect_evaluation_candidates(out,load_json(args.scene_json),load_json(args.camera_config),load_json(args.refinement_config),cfg)
        state=stage/'A_candidates'
        artifact=ArtifactIndex(scene_id=out.name,step='09e_a_collect_evaluation_views')
        artifact.add('candidate_manifest',state/'candidate_manifest.json')
        artifact.add('candidate_shared_buffers',state/'shared_buffers')
        artifact.add('rgb_contact_sheet',stage/'candidate_contact_sheet_rgb.png')
        artifact.add('depth_contact_sheet',stage/'candidate_contact_sheet_depth.png')
        artifact.add('stage_report',stage/'A_stage_report.json')
        artifact.save(state/'artifact_index.json')
        mark_done(state)
        status(f"[09E-A] Done. {report['novel_candidate_count']} novel candidates + {report['training_replay_count']} immutable training-replay views. Edit {stage/'selection.csv'} before Stage09E-B.")
    except Exception:
        stage.mkdir(parents=True,exist_ok=True); (stage/'.A_failed').write_text(traceback.format_exc(),encoding='utf-8'); raise
if __name__=='__main__': main()
