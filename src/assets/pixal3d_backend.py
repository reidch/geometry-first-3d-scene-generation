from __future__ import annotations
import os, queue, shutil, signal, subprocess, threading, time, math
from collections import deque
from pathlib import Path
from typing import Any, Dict
from src.io.json_io import save_json


class Pixal3DTimeoutError(RuntimeError):
    """Raised when Pixal3D stops producing output for too long or exceeds runtime."""


def ensure_pixal_repo(config):
    repo=Path(os.environ.get("PIXAL3D_REPO") or config.get("repo_dir","external/Pixal3D")).expanduser().resolve()
    if not (repo/"inference.py").exists():
        if not config.get("auto_clone_repo",True): raise FileNotFoundError(f"Pixal3D repo missing: {repo}")
        repo.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["git","clone",config.get("repo_url","https://github.com/TencentARC/Pixal3D.git"),str(repo)],check=True)
    return repo


def _stop_process_tree(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    """Terminate the complete Pixal3D subprocess group, then force-kill if needed."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=max(0.1, float(grace_seconds)))
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()


def _stream(
    cmd,
    cwd,
    env,
    log_path,
    label,
    heartbeat=30,
    stall_timeout_seconds: float | None = None,
    max_runtime_seconds: float | None = None,
):
    q=queue.Queue(); tail=deque(maxlen=1000); start=time.monotonic(); last_real_output=start
    with Path(log_path).open('w',encoding='utf-8',errors='replace') as f:
        p=subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        def reader():
            assert p.stdout is not None
            for line in p.stdout: q.put(line)
            q.put(None)
        t=threading.Thread(target=reader,daemon=True); t.start()
        timed_out_reason=None
        while True:
            now=time.monotonic()
            elapsed=now-start
            silent_for=now-last_real_output
            if p.poll() is None and max_runtime_seconds and elapsed > float(max_runtime_seconds):
                timed_out_reason=f"maximum runtime {float(max_runtime_seconds):.0f}s exceeded"
            elif p.poll() is None and stall_timeout_seconds and silent_for > float(stall_timeout_seconds):
                timed_out_reason=f"no subprocess output for {float(stall_timeout_seconds):.0f}s"
            if timed_out_reason:
                message=(
                    f"[{label}] TIMEOUT: {timed_out_reason}; elapsed={int(elapsed)}s. "
                    "Terminating this seed so Stage04 can retry with the next seed.\n"
                )
                print(message,end='',flush=True); f.write(message); f.flush(); tail.append(message)
                _stop_process_tree(p)
                break
            try:
                x=q.get(timeout=max(1,min(10,int(heartbeat))))
                real_output = x is not None
            except queue.Empty:
                x=None
                real_output=False
                if p.poll() is None and int(time.monotonic()-start) % max(1,int(heartbeat)) < 10:
                    x=f"[{label}] still running; elapsed={int(time.monotonic()-start)}s; silent={int(time.monotonic()-last_real_output)}s\n"
            if x is None:
                if p.poll() is None:
                    continue
                break
            if real_output:
                last_real_output=time.monotonic()
            print(x,end='',flush=True); f.write(x); f.flush(); tail.append(x)
        rc=p.wait()
    if timed_out_reason:
        raise Pixal3DTimeoutError(
            f"{label} timed out ({timed_out_reason}); see {log_path}\n{''.join(tail)[-12000:]}"
        )
    if rc!=0: raise RuntimeError(f"{label} failed; see {log_path}\n{''.join(tail)[-12000:]}")
    return ''.join(tail)


class Pixal3DBackend:
    def __init__(self,config:Dict[str,Any]):
        self.config=dict(config); self.repo=ensure_pixal_repo(config)
        self.python=os.environ.get("PIXAL3D_PYTHON") or str(config.get("runtime_python","python"))
    def generate_asset(self,image_path:Path,output_dir:Path,object_id:str,seed:int,camera_context:Dict[str,Any]|None=None,**_):
        image_path=Path(image_path).resolve(); output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True)
        if not image_path.exists(): raise FileNotFoundError(image_path)
        glb=output_dir/'asset.glb'; log=output_dir/'pixal3d_runtime.log'
        wrapper=Path(__file__).with_name('pixal3d_inference_texture_vram.py').resolve()
        cmd=[self.python,'-u',str(wrapper),'--repo',str(self.repo),'--image',str(image_path),'--output',str(glb),'--seed',str(seed),'--model_path',str(self.config.get('model_path','TencentARC/Pixal3D')),'--resolution',str(int(self.config.get('resolution',1024))),'--texture_naf_target_size',str(int(self.config.get('texture_naf_target_size',64)))]
        camera_context=dict(camera_context or {})
        camera=dict(camera_context.get('camera',{}))
        camera_fov=camera.get('fov_deg')
        fov_source='representative_camera'
        if camera_fov not in (None,''):
            fov=float(camera_fov)
        elif str(camera.get('type','')).upper()=='ORTHO' and camera.get('ortho_scale') and camera.get('location') and camera.get('target'):
            location=[float(v) for v in camera['location']]; target=[float(v) for v in camera['target']]
            distance=math.sqrt(sum((location[i]-target[i])**2 for i in range(3)))
            fov=math.degrees(2.0*math.atan(0.5*float(camera['ortho_scale'])/max(distance,1e-6)))
            fov_source='orthographic_equivalent_from_scale_distance'
        else:
            fov=float(self.config.get('manual_fov_deg',28.0)); fov_source='config_fallback'
        if fov>0:
            cmd += ['--fov',str(math.radians(fov))]
        if self.config.get('low_vram',True): cmd.append('--low_vram')
        env=os.environ.copy(); env['PYTHONPATH']=str(self.repo)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''); env.setdefault('ATTN_BACKEND','sdpa'); env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
        if glb.exists() and glb.stat().st_size>0:
            tail='[PIXAL3D] Reusing existing non-empty asset.glb; inference skipped.\n'
            print(tail,end='',flush=True)
        else:
            # A previous timed-out run may leave a partial file. Never reuse it.
            if glb.exists():
                glb.unlink()
            tail=_stream(
                cmd,self.repo,env,log,'PIXAL3D',self.config.get('heartbeat_seconds',30),
                stall_timeout_seconds=self.config.get('stall_timeout_seconds',600),
                max_runtime_seconds=self.config.get('max_runtime_seconds',3600),
            )
        if not glb.exists() or glb.stat().st_size==0: raise RuntimeError('Pixal3D produced no GLB')
        pose_context=output_dir/'pose_context.json'
        save_json(camera_context,pose_context)
        converter=Path(__file__).with_name('pixal3d_glb_to_obj.py').resolve(); obj=output_dir/'asset.obj'; manifest=output_dir/'asset_bundle.json'
        _stream([self.python,'-u',str(converter),'--input',str(glb),'--output',str(obj),'--manifest',str(manifest)],Path.cwd(),env,output_dir/'pixal3d_convert.log','PIXAL3D-CONVERT',30,max_runtime_seconds=self.config.get('conversion_timeout_seconds',900))
        if not obj.exists() or not manifest.exists(): raise RuntimeError('Pixal3D Blender-safe OBJ export failed')
        report={'status':'ok','backend':'pixal3d','object_id':object_id,'input_image':str(image_path),'seed':seed,'resolution':int(self.config.get('resolution',1024)),'manual_fov_deg':fov,'fov_source':fov_source,'camera_pose_correction_requested':False,'camera_context_saved_for_diagnostics':bool(camera_context),'low_vram':bool(self.config.get('low_vram',True)),'texture_naf_target_size':int(self.config.get('texture_naf_target_size',64)),'texture_memory_policy':'texture_conditioning_naf_only','downstream_atlas_resolution_unchanged':True,'textured':True,'stall_timeout_seconds':float(self.config.get('stall_timeout_seconds',600)),'max_runtime_seconds':float(self.config.get('max_runtime_seconds',3600))}
        save_json(report,output_dir/'pixal3d_runtime_report.json')
        return {'status':'ok','object_id':object_id,'asset_path':str(glb),'blender_asset_path':str(obj),'obj_path':str(obj),'asset_bundle_manifest':str(manifest),'runtime_report':report,'runtime_log':str(log),'runtime_stdout_tail':tail[-3000:],'backend':'pixal3d','geometry_rejection_enabled':False}
