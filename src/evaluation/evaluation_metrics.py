from __future__ import annotations
import csv, json, math
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from src.cameras.reconstruction_view_metrics import decode_metric_depth
from src.io.json_io import load_json, save_json

def _load_mesh_depth(cam):
    report=load_json(Path(cam['mesh_depth_control']).parent/'camera_report.json')
    return np.asarray(decode_metric_depth(report['depth'],report['depth_encoding']),dtype=np.float64)
def _cal(cam): return load_json(cam['rendered_camera'])
def _normal_from_depth(depth,K,abs_t,rel_t):
    d=np.asarray(depth,np.float64); h,w=d.shape; yy,xx=np.indices((h,w),dtype=np.float64); fx,fy=float(K[0,0]),float(K[1,1]); cx,cy=float(K[0,2]),float(K[1,2])
    X=np.stack([(xx+0.5-cx)*d/fx,(yy+0.5-cy)*d/fy,d],axis=-1); vx=X[:,2:]-X[:,:-2]; vy=X[2:]-X[:-2]; n=np.cross(vx[1:-1],vy[:,1:-1]); norm=np.linalg.norm(n,axis=-1); valid=(d[1:-1,1:-1]>0)&np.isfinite(d[1:-1,1:-1])&(norm>1e-9)
    c=d[1:-1,1:-1]; neigh=[d[1:-1,2:],d[1:-1,:-2],d[2:,1:-1],d[:-2,1:-1]]
    tol=np.maximum(abs_t,rel_t*np.maximum(c,1e-9))
    for q in neigh: valid&=np.isfinite(q)&(q>0)&(np.abs(q-c)<=tol)
    out=np.zeros_like(n); out[valid]=n[valid]/norm[valid,None]; return out,valid
def _stats(vals):
    v=np.asarray(vals,np.float64); v=v[np.isfinite(v)]; return {'mean':float(v.mean()) if len(v) else None,'std':float(v.std()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'count':int(len(v))}
def _backproject(depth,K,c2w):
    h,w=depth.shape; yy,xx=np.nonzero(np.isfinite(depth)&(depth>0)); z=depth[yy,xx]; fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
    cam=np.stack([(xx+0.5-cx)*z/fx,(yy+0.5-cy)*z/fy,z],axis=-1); world=cam@c2w[:3,:3].T+c2w[:3,3]; return world
def _directed_reproj(src_depth,src_cal,tgt_depth,tgt_cal,abs_tol,rel_tol):
    Ks=np.asarray(src_cal['K'],np.float64); Kt=np.asarray(tgt_cal['K'],np.float64); cs=np.asarray(src_cal['camera_to_world_opencv'],np.float64); wt=np.asarray(tgt_cal['world_to_camera_opencv'],np.float64)
    world=_backproject(src_depth,Ks,cs); tc=world@wt[:3,:3].T+wt[:3,3]; z=tc[:,2]; valid=np.isfinite(z)&(z>0); tc=tc[valid]; z=z[valid]
    u=Kt[0,0]*tc[:,0]/z+Kt[0,2]-0.5; v=Kt[1,1]*tc[:,1]/z+Kt[1,2]-0.5; xi=np.rint(u).astype(int); yi=np.rint(v).astype(int); h,w=tgt_depth.shape; m=(xi>=0)&(xi<w)&(yi>=0)&(yi<h); xi,yi,z=xi[m],yi[m],z[m]
    flat=yi*w+xi; order=np.lexsort((z,flat)); flat=flat[order]; z=z[order]; first=np.r_[True,flat[1:]!=flat[:-1]]; flat,z=flat[first],z[first]; yi=flat//w; xi=flat%w; gt=tgt_depth[yi,xi]; m=np.isfinite(gt)&(gt>0); z,gt=z[m],gt[m]
    tol=np.maximum(abs_tol,rel_tol*gt); # source point behind an already visible target surface is an expected occlusion.
    m=z<=gt+tol; z,gt=z[m],gt[m]
    if len(z)==0:return {'mae':math.nan,'absrel':math.nan,'count':0}
    e=np.abs(z-gt); return {'mae':float(e.mean()),'absrel':float((e/np.maximum(gt,1e-9)).mean()),'count':int(len(e))}
def compute_metrics(stage:Path,cfg:Mapping[str,Any])->dict:
    manifest=load_json(stage/'B_frozen_evaluation_manifest.json'); render_root=stage/'B_renders'; metrics_root=stage/'B_metrics'; metrics_root.mkdir(parents=True,exist_ok=True); mc=dict(cfg.get('metrics',{})); abs_n=float(mc.get('normal_depth_discontinuity_absolute_threshold_m',0.05)); rel_n=float(mc.get('normal_depth_discontinuity_relative_threshold',0.05)); rows=[]; byid={str(c['camera_id']):c for c in manifest['novel_views']}
    for cam in manifest['novel_views']:
        cid=str(cam['camera_id']); gd=np.load(render_root/'novel'/cid/'gs_depth_m.npy').astype(np.float64); md=_load_mesh_depth(cam); cal=_cal(cam); K=np.asarray(cal['K'],np.float64); valid=np.isfinite(md)&(md>0)&np.isfinite(gd)&(gd>0); diff=gd[valid]-md[valid]
        ng,vg=_normal_from_depth(gd,K,abs_n,rel_n); nm,vm=_normal_from_depth(md,K,abs_n,rel_n); nv=vg&vm; dots=np.sum(ng[nv]*nm[nv],axis=-1); dots=np.clip(dots,-1,1)
        rows.append({'camera_id':cid,'evaluation_type':cam.get('evaluation_type'),'trajectory_id':cam.get('trajectory_id',''),'target_object_id':cam.get('target_object_id',''),'m_mae_m':float(np.mean(np.abs(diff))) if len(diff) else math.nan,'m_rmse_m':float(np.sqrt(np.mean(diff*diff))) if len(diff) else math.nan,'n_cos':float(np.mean(dots)) if len(dots) else math.nan,'n_ang_deg':float(np.degrees(np.arccos(dots)).mean()) if len(dots) else math.nan,'clip_iqa_plus':math.nan,'clip_aesthetic':math.nan})
    qpath=metrics_root/'image_quality_per_view.csv'
    if qpath.exists():
        with qpath.open(newline='',encoding='utf-8') as f:
            qm={r['camera_id']:r for r in csv.DictReader(f)}
        for r in rows:
            if r['camera_id'] in qm:r['clip_iqa_plus']=float(qm[r['camera_id']]['clip_iqa_plus']); r['clip_aesthetic']=float(qm[r['camera_id']]['clip_aesthetic'])
    fields=['camera_id','evaluation_type','trajectory_id','target_object_id','m_mae_m','m_rmse_m','n_cos','n_ang_deg','clip_iqa_plus','clip_aesthetic']
    with (metrics_root/'per_novel_view.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    abs_o=float(mc.get('reprojection_occlusion_absolute_tolerance_m',0.03)); rel_o=float(mc.get('reprojection_occlusion_relative_tolerance',0.01)); prows=[]
    for p in manifest['reprojection_pairs']:
        a,b=byid[p['source_camera_id']],byid[p['target_camera_id']]; da=np.load(render_root/'novel'/a['camera_id']/'gs_depth_m.npy'); db=np.load(render_root/'novel'/b['camera_id']/'gs_depth_m.npy'); ab=_directed_reproj(da,_cal(a),db,_cal(b),abs_o,rel_o); ba=_directed_reproj(db,_cal(b),da,_cal(a),abs_o,rel_o); vals=[x for x in (ab['mae'],ba['mae']) if np.isfinite(x)]; rels=[x for x in (ab['absrel'],ba['absrel']) if np.isfinite(x)]; prows.append({**p,'reproj_mae_m':float(np.mean(vals)) if vals else math.nan,'reproj_absrel':float(np.mean(rels)) if rels else math.nan,'valid_projected_pixels':ab['count']+ba['count']})
    pf=['pair_id','trajectory_id','evaluation_type','source_camera_id','target_camera_id','reproj_mae_m','reproj_absrel','valid_projected_pixels']
    with (metrics_root/'per_pair.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=pf);w.writeheader();w.writerows(prows)
    # copy training replay metrics into metrics root
    import shutil; shutil.copy2(render_root/'training_replay_metrics.csv',metrics_root/'per_training_view.csv')
    with (metrics_root/'per_training_view.csv').open(newline='',encoding='utf-8') as f: tr=list(csv.DictReader(f))
    summary={'schema_version':1,'training_replay':{k:_stats([float(r[k]) for r in tr]) for k in ('psnr','ssim','lpips')},'novel_overall':{},'novel_by_type':{},'pairs_overall':{},'pairs_by_type':{}}
    for k in ('m_mae_m','m_rmse_m','n_cos','n_ang_deg','clip_iqa_plus','clip_aesthetic'):summary['novel_overall'][k]=_stats([r[k] for r in rows])
    for typ in sorted({str(r['evaluation_type']) for r in rows}):summary['novel_by_type'][typ]={k:_stats([r[k] for r in rows if r['evaluation_type']==typ]) for k in ('m_mae_m','m_rmse_m','n_cos','n_ang_deg','clip_iqa_plus','clip_aesthetic')}
    for k in ('reproj_mae_m','reproj_absrel'):summary['pairs_overall'][k]=_stats([r[k] for r in prows])
    for typ in sorted({str(r['evaluation_type']) for r in prows}):summary['pairs_by_type'][typ]={k:_stats([r[k] for r in prows if r['evaluation_type']==typ]) for k in ('reproj_mae_m','reproj_absrel')}
    save_json(summary,metrics_root/'summary.json'); return summary
