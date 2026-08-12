#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

def save_rgb(path:Path,t):
    a=t.detach().float().clamp(0,1).cpu().numpy(); path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.rint(a*255).astype(np.uint8),'RGB').save(path)
def save_gray(path:Path,t):
    a=t.detach().float().clamp(0,1).cpu().numpy(); a=a[...,0] if a.ndim==3 else a; path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.rint(a*255).astype(np.uint8),'L').save(path)
def preview_depth(path:Path,d):
    valid=np.isfinite(d)&(d>0); out=np.zeros(d.shape,np.uint8)
    if np.any(valid):
        lo,hi=np.percentile(d[valid],[2,98]); hi=max(float(hi),float(lo)+1e-6); out[valid]=np.rint((1-np.clip((d[valid]-lo)/(hi-lo),0,1))*255).astype(np.uint8)
    Image.fromarray(out,'L').save(path)

def original_opencv_to_ns(c2w_cv, dp):
    c=np.asarray(c2w_cv,dtype=np.float64).copy(); c[:3,1:3]*=-1.0
    transform=dp.dataparser_transform.detach().cpu().numpy().astype(np.float64)
    T=np.eye(4); T[:3,:4]=transform
    out=T@c; out[:3,3]*=float(dp.dataparser_scale)
    return out[:3,:4]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--load-config',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras,CameraType
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); manifest=json.loads(Path(args.manifest).read_text())
    _,pipeline,checkpoint,_=eval_setup(Path(args.load_config)); pipeline.eval(); dm=pipeline.datamanager
    dp=dm.train_dataparser_outputs; scale=float(dp.dataparser_scale)
    rows=[]; replay=list(manifest.get('training_replay',[])); loader=dm.fixed_indices_eval_dataloader
    if len(loader)!=len(replay): raise RuntimeError(f'Training replay mismatch: Nerfstudio eval cameras={len(loader)} manifest={len(replay)}')
    with torch.no_grad():
        for ordinal,(camera,batch) in enumerate(loader,start=1):
            image_idx_value=batch.get('image_idx',ordinal-1)
            if torch.is_tensor(image_idx_value): image_idx=int(image_idx_value.reshape(-1)[0].item())
            else: image_idx=int(image_idx_value)
            if image_idx<0 or image_idx>=len(replay): raise RuntimeError(f'Unexpected Nerfstudio evaluation image_idx={image_idx}')
            meta=replay[image_idx]
            outputs=pipeline.model.get_outputs_for_camera(camera); metrics,_=pipeline.model.get_image_metrics_and_images(outputs,batch)
            cid=str(meta['camera_id']); d=output/'training_replay'/cid; d.mkdir(parents=True,exist_ok=True); save_rgb(d/'gs_rgb.png',outputs['rgb'])
            rows.append({'camera_id':cid,'evaluation_type':'training_replay','psnr':float(metrics['psnr']),'ssim':float(metrics['ssim']),'lpips':float(metrics['lpips'])})
            print(f"[09E-B][TRAIN] {ordinal}/{len(replay)} {cid} PSNR={rows[-1]['psnr']:.2f}",flush=True)
    with (output/'training_replay_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['camera_id','evaluation_type','psnr','ssim','lpips']); w.writeheader(); w.writerows(rows)
    novel=list(manifest.get('novel_views',[]))
    with torch.no_grad():
        for ordinal,meta in enumerate(novel,start=1):
            cal=json.loads(Path(meta['rendered_camera']).read_text()); K=np.asarray(cal['K'],dtype=np.float64); c2w=np.asarray(cal['camera_to_world_opencv'],dtype=np.float64)
            ns=original_opencv_to_ns(c2w,dp)
            # Nerfstudio Cameras expects camera_to_worlds as a batched tensor
            # shaped [B, 3, 4]. Training/eval dataloaders use B=1; keep novel
            # cameras identical to that contract instead of passing a bare [3,4]
            # matrix (which later makes Splatfacto.get_viewmat index a 2-D tensor
            # as if it were 3-D, and the error is obscured by torch.compile/Dynamo).
            ns_c2w=torch.tensor(ns,dtype=torch.float32).unsqueeze(0)
            if tuple(ns_c2w.shape)!=(1,3,4):
                raise RuntimeError(f'Novel Nerfstudio camera must be [1,3,4], got {tuple(ns_c2w.shape)} for {meta.get("camera_id")}')
            camera=Cameras(camera_to_worlds=ns_c2w,fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),width=int(cal['width']),height=int(cal['height']),camera_type=CameraType.PERSPECTIVE)
            if tuple(camera.camera_to_worlds.shape)!=(1,3,4):
                raise RuntimeError(f'Nerfstudio Cameras changed novel camera batch shape to {tuple(camera.camera_to_worlds.shape)} for {meta.get("camera_id")}')
            outputs=pipeline.model.get_outputs_for_camera(camera)
            cid=str(meta['camera_id']); d=output/'novel'/cid; d.mkdir(parents=True,exist_ok=True); save_rgb(d/'gs_rgb.png',outputs['rgb']); save_gray(d/'gs_accumulation.png',outputs['accumulation'])
            depth=outputs['depth'].detach().float().cpu().numpy().squeeze(-1).astype(np.float32)/scale; np.save(d/'gs_depth_m.npy',depth); preview_depth(d/'gs_depth_preview.png',depth)
            (d/'render_metadata.json').write_text(json.dumps({'camera_id':cid,'nerfstudio_checkpoint':str(checkpoint),'dataparser_scale':scale,'source_camera':meta['rendered_camera'],'depth_units':'original_scene_meters','depth_convention':'camera_z_expected_depth'},indent=2),encoding='utf-8')
            print(f'[09E-B][NOVEL] {ordinal}/{len(novel)} {cid}',flush=True)
    (output/'nerfstudio_render_report.json').write_text(json.dumps({'checkpoint':str(checkpoint),'training_replay_count':len(replay),'novel_view_count':len(novel),'dataparser_scale':scale},indent=2),encoding='utf-8')
if __name__=='__main__': main()
