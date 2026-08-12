#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,json,urllib.request
from pathlib import Path
import torch
from PIL import Image

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--renders',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',required=True); args=ap.parse_args()
    import pyiqa, open_clip
    cfg=json.loads(Path(args.config).read_text()); qcfg=dict(cfg.get('image_quality',{})); manifest=json.loads(Path(args.manifest).read_text()); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iqa=pyiqa.create_metric(str(qcfg.get('pyiqa_metric','clipiqa+')),device=device)
    model_name=str(qcfg.get('clip_model','ViT-L-14')); pretrained=str(qcfg.get('clip_pretrained','openai')); clip,_,preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained); clip=clip.to(device).eval()
    # LAION aesthetic predictor V1: one linear layer over normalized ViT-L/14 CLIP embeddings.
    weights=out/'sa_0_4_vit_l_14_linear.pth'
    if not weights.exists(): urllib.request.urlretrieve(str(qcfg['aesthetic_weights_url']),weights)
    predictor=torch.nn.Linear(768,1).to(device); state=torch.load(weights,map_location=device); predictor.load_state_dict(state); predictor.eval()
    rows=[]
    with torch.no_grad():
        for i,meta in enumerate(manifest.get('novel_views',[]),1):
            cid=str(meta['camera_id']); path=Path(args.renders)/'novel'/cid/'gs_rgb.png'; pil=Image.open(path).convert('RGB')
            # pyiqa accepts image file paths and performs its official preprocessing.
            iq=float(iqa(str(path)).detach().cpu().reshape(-1)[0])
            x=preprocess(pil).unsqueeze(0).to(device); emb=clip.encode_image(x); emb=emb/emb.norm(dim=-1,keepdim=True); aesthetic=float(predictor(emb.float()).detach().cpu().reshape(-1)[0])
            rows.append({'camera_id':cid,'evaluation_type':meta.get('evaluation_type',''),'clip_iqa_plus':iq,'clip_aesthetic':aesthetic}); print(f'[09E-B][QUALITY] {i}/{len(manifest.get("novel_views",[]))} {cid}',flush=True)
    with (out/'image_quality_per_view.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['camera_id','evaluation_type','clip_iqa_plus','clip_aesthetic']);w.writeheader();w.writerows(rows)
if __name__=='__main__': main()
