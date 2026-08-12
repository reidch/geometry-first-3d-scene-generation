from __future__ import annotations

"""Standard-library GLB -> OBJ/MTL converter for Pixal3D assets.

No third-party Python packages are used. The converter supports embedded PNG,
JPEG and WebP images, EXT_texture_webp, KHR_texture_basisu, base-color and
metallic/roughness texture extraction, KHR_texture_transform UV transforms,
node transforms, multiple primitives/materials and optional camera-derived
pose correction.
"""

import argparse
import base64
import json
import math
import mimetypes
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_COMPONENT = {5120:("b",1),5121:("B",1),5122:("h",2),5123:("H",2),5125:("I",4),5126:("f",4)}
_TYPE_COUNT = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}


def _args() -> argparse.Namespace:
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--manifest",required=True)
    p.add_argument("--pose-context",default="")
    return p.parse_args(argv)


def _load_glb(path:Path)->tuple[dict[str,Any],bytes]:
    raw=path.read_bytes()
    if len(raw)<20: raise ValueError(f"Invalid GLB (too small): {path}")
    magic,version,total=struct.unpack_from("<4sII",raw,0)
    if magic!=b"glTF" or version!=2 or total>len(raw): raise ValueError(f"Invalid GLB header: {path}")
    off,doc,binary=12,None,b""
    while off+8<=total:
        size,kind=struct.unpack_from("<I4s",raw,off); off+=8; payload=raw[off:off+size]; off+=size
        if kind==b"JSON": doc=json.loads(payload.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif kind in (b"BIN\0",b"BIN "): binary=payload
    if doc is None: raise ValueError("GLB has no JSON chunk")
    return doc,binary


def _buffer_bytes(doc,binary,glb_dir,index):
    item=doc.get("buffers",[])[index]; uri=item.get("uri")
    if not uri: return binary
    if uri.startswith("data:"): return base64.b64decode(uri.split(",",1)[1])
    return (glb_dir/uri).read_bytes()


def _normalized(v,c):
    if c==5120:return max(v/127.0,-1.0)
    if c==5121:return v/255.0
    if c==5122:return max(v/32767.0,-1.0)
    if c==5123:return v/65535.0
    if c==5125:return v/4294967295.0
    return float(v)


def _accessor(doc,binary,glb_dir,index):
    a=doc["accessors"][index]; n=_TYPE_COUNT[a["type"]]
    if "bufferView" not in a:
        z=[0.0]*n; values=[z[0] if n==1 else tuple(z) for _ in range(a["count"])]
    else:
        view=doc["bufferViews"][a["bufferView"]]; buf=_buffer_bytes(doc,binary,glb_dir,view.get("buffer",0))
        component=a["componentType"]; fmt,size=_COMPONENT[component]; stride=view.get("byteStride",size*n)
        start=view.get("byteOffset",0)+a.get("byteOffset",0); unpack=struct.Struct("<"+fmt*n).unpack_from; values=[]
        for i in range(a["count"]):
            v=unpack(buf,start+i*stride)
            if a.get("normalized") and component!=5126: v=tuple(_normalized(x,component) for x in v)
            values.append(v[0] if n==1 else tuple(float(x) for x in v))
    sparse=a.get("sparse")
    if sparse:
        ii,vi=sparse["indices"],sparse["values"]; iv=doc["bufferViews"][ii["bufferView"]]; vv=doc["bufferViews"][vi["bufferView"]]
        ib=_buffer_bytes(doc,binary,glb_dir,iv.get("buffer",0)); vb=_buffer_bytes(doc,binary,glb_dir,vv.get("buffer",0))
        ifmt,isize=_COMPONENT[ii["componentType"]]; component=a["componentType"]; fmt,size=_COMPONENT[component]
        iu=struct.Struct("<"+ifmt).unpack_from; vu=struct.Struct("<"+fmt*n).unpack_from
        io=iv.get("byteOffset",0)+ii.get("byteOffset",0); vo=vv.get("byteOffset",0)+vi.get("byteOffset",0)
        for j in range(sparse["count"]):
            dst=iu(ib,io+j*isize)[0]; v=vu(vb,vo+j*size*n); values[dst]=v[0] if n==1 else tuple(float(x) for x in v)
    return values


def _identity(): return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
def _mul(a,b): return [sum(a[k*4+r]*b[c*4+k] for k in range(4)) for c in range(4) for r in range(4)]


def _trs(node):
    if "matrix" in node:return [float(x) for x in node["matrix"]]
    tx,ty,tz=node.get("translation",[0,0,0]); x,y,z,w=node.get("rotation",[0,0,0,1]); sx,sy,sz=node.get("scale",[1,1,1])
    r=[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),0,2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w),0,2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y),0,0,0,0,1]
    s=[sx,0,0,0,0,sy,0,0,0,0,sz,0,0,0,0,1]; t=[1,0,0,0,0,1,0,0,0,0,1,0,tx,ty,tz,1]
    return _mul(t,_mul(r,s))


def _point(m,p):
    x,y,z=p; return (m[0]*x+m[4]*y+m[8]*z+m[12],m[1]*x+m[5]*y+m[9]*z+m[13],m[2]*x+m[6]*y+m[10]*z+m[14])


def _normal(m,n):
    # Rotation-only correction matrices are orthonormal; node matrices may include scale.
    a,b,c,d,e,f,g,h,i=m[0],m[4],m[8],m[1],m[5],m[9],m[2],m[6],m[10]
    A=e*i-f*h; B=f*g-d*i; C=d*h-e*g; det=a*A+b*B+c*C
    if abs(det)<1e-20:x,y,z=n
    else:
        inv=[A,c*h-b*i,b*f-c*e,B,a*i-c*g,c*d-a*f,C,b*g-a*h,a*e-b*d]
        x,y,z=(inv[0]*n[0]+inv[3]*n[1]+inv[6]*n[2],inv[1]*n[0]+inv[4]*n[1]+inv[7]*n[2],inv[2]*n[0]+inv[5]*n[1]+inv[8]*n[2])
    q=math.sqrt(x*x+y*y+z*z) or 1.0; return x/q,y/q,z/q


def _triangles(indices,mode):
    if mode==4:
        for i in range(0,len(indices)-2,3):yield int(indices[i]),int(indices[i+1]),int(indices[i+2])
    elif mode==5:
        for i in range(len(indices)-2):
            a,b,c=map(int,indices[i:i+3]); yield (a,b,c) if i%2==0 else (b,a,c)
    elif mode==6:
        for i in range(1,len(indices)-1):yield int(indices[0]),int(indices[i]),int(indices[i+1])
    else:raise ValueError(f"Unsupported primitive mode {mode}")


def _image(doc,binary,glb_dir,image_index):
    im=doc["images"][image_index]; mime=im.get("mimeType"); uri=im.get("uri")
    if uri:
        if uri.startswith("data:"):
            header,payload=uri.split(",",1); mime=mime or header[5:].split(";",1)[0]; return base64.b64decode(payload),mime or "application/octet-stream"
        p=glb_dir/uri; return p.read_bytes(),mime or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    view=doc["bufferViews"][im["bufferView"]]; buf=_buffer_bytes(doc,binary,glb_dir,view.get("buffer",0)); start=view.get("byteOffset",0)
    return buf[start:start+view["byteLength"]],mime or "application/octet-stream"


def _texture_source(texture):
    ext=texture.get("extensions",{})
    if "EXT_texture_webp" in ext:return ext["EXT_texture_webp"].get("source")
    if "KHR_texture_basisu" in ext:return ext["KHR_texture_basisu"].get("source")
    return texture.get("source")


def _extract_texture(doc,binary,input_path,output_path,texture_index,label,material_index,texture_paths):
    textures=doc.get("textures",[])
    if texture_index is None or texture_index<0 or texture_index>=len(textures):return None
    source=_texture_source(textures[texture_index])
    if source is None:return None
    payload,mime=_image(doc,binary,input_path.parent,int(source)); ext={"image/png":".png","image/jpeg":".jpg","image/webp":".webp","image/ktx2":".ktx2"}.get(mime,".bin")
    dst=output_path.parent/f"{output_path.stem}_{label}_{material_index:03d}{ext}"; dst.write_bytes(payload); texture_paths.append(str(dst.resolve())); return dst


def _uv_transform(tex_info):
    ext=dict(tex_info.get("extensions",{})).get("KHR_texture_transform",{}) if tex_info else {}
    return {"offset":ext.get("offset",[0.0,0.0]),"scale":ext.get("scale",[1.0,1.0]),"rotation":float(ext.get("rotation",0.0)),"texCoord":int(ext.get("texCoord",tex_info.get("texCoord",0) if tex_info else 0))}


def _apply_uv(uv,tr):
    u,v=float(uv[0]),float(uv[1]); sx,sy=tr["scale"]; ox,oy=tr["offset"]; a=tr["rotation"]; c,s=math.cos(a),math.sin(a)
    u,v=u*float(sx),v*float(sy); return (c*u-s*v+float(ox),s*u+c*v+float(oy))


def _sub(a,b):return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def _cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def _norm(v):
    n=math.sqrt(sum(x*x for x in v)) or 1.0; return tuple(x/n for x in v)


def _pose_matrix(context):
    """Map Pixal camera-aligned local coordinates into the selected render camera frame.

    glTF convention is treated as +X camera-right, +Y camera-up, +Z camera-backward.
    The transformation is rotation-only and recenters geometry later, so the asset
    remains local while its up/front axes inherit the real representative camera.
    """
    camera=dict(context.get("camera",{})); loc=camera.get("location"); target=camera.get("target")
    if not (isinstance(loc,list) and isinstance(target,list) and len(loc)==3 and len(target)==3):return _identity(),False
    forward=_norm(_sub(tuple(target),tuple(loc))); world_up=(0.0,0.0,1.0)
    right=_cross(forward,world_up)
    if sum(x*x for x in right)<1e-10:right=(1.0,0.0,0.0)
    right=_norm(right); up=_norm(_cross(right,forward)); back=tuple(-x for x in forward)
    # Column-major matrix whose columns are right/up/back.
    return [right[0],right[1],right[2],0,up[0],up[1],up[2],0,back[0],back[1],back[2],0,0,0,0,1],True


def convert(input_path:Path,output_path:Path,manifest_path:Path,pose_context_path:Path|None=None)->dict[str,Any]:
    doc,binary=_load_glb(input_path); output_path.parent.mkdir(parents=True,exist_ok=True); mtl_path=output_path.with_suffix(".mtl")
    context={}
    if pose_context_path and pose_context_path.exists():context=json.loads(pose_context_path.read_text(encoding="utf-8"))
    pose,pose_applied=_pose_matrix(context)
    texture_paths=[]; material_names=[]; material_uv=[]; mtl=[]
    for mi,mat in enumerate(doc.get("materials",[])):
        name=f"material_{mi:03d}"; material_names.append(name); pbr=mat.get("pbrMetallicRoughness",{}); color=pbr.get("baseColorFactor",[1,1,1,1])
        mtl += [f"newmtl {name}",f"Kd {color[0]:.9g} {color[1]:.9g} {color[2]:.9g}",f"d {color[3]:.9g}","illum 2"]
        base=pbr.get("baseColorTexture"); material_uv.append(_uv_transform(base))
        if base:
            tex=_extract_texture(doc,binary,input_path,output_path,int(base["index"]),"basecolor",mi,texture_paths)
            if tex:mtl.append(f"map_Kd {tex.name}")
        mr=pbr.get("metallicRoughnessTexture")
        if mr:
            tex=_extract_texture(doc,binary,input_path,output_path,int(mr["index"]),"metallic_roughness",mi,texture_paths)
            if tex:mtl.append(f"# glTF metallic-roughness texture: {tex.name}")
        mtl.append("")
    if not material_names:material_names=["material_default"]; material_uv=[_uv_transform(None)]; mtl=["newmtl material_default","Kd 1 1 1","d 1","illum 2",""]
    mtl_path.write_text("\n".join(mtl),encoding="utf-8")

    # First gather transformed primitives so a single local recenter can be applied.
    nodes=doc.get("nodes",[]); meshes=doc.get("meshes",[]); scene_index=doc.get("scene",0); roots=doc.get("scenes",[{"nodes":list(range(len(nodes)))}])[scene_index].get("nodes",[])
    prims=[]; all_positions=[]
    def visit(node_index,parent_local):
        node=nodes[node_index]; local_world=_mul(parent_local,_trs(node)); world=_mul(pose,local_world)
        if "mesh" in node:
            mesh=meshes[node["mesh"]]
            for pi,prim in enumerate(mesh.get("primitives",[])):
                attrs=prim["attributes"]; positions=[_point(world,p) for p in _accessor(doc,binary,input_path.parent,attrs["POSITION"])]
                mat_i=int(prim.get("material",0)); tr=material_uv[mat_i] if mat_i<len(material_uv) else _uv_transform(None); uv_key=f"TEXCOORD_{tr['texCoord']}"
                uvs=[_apply_uv(uv,tr) for uv in _accessor(doc,binary,input_path.parent,attrs[uv_key])] if uv_key in attrs else []
                normals=[_normal(world,n) for n in _accessor(doc,binary,input_path.parent,attrs["NORMAL"])] if "NORMAL" in attrs else []
                indices=_accessor(doc,binary,input_path.parent,prim["indices"]) if "indices" in prim else list(range(len(positions)))
                prims.append((node_index,pi,positions,uvs,normals,indices,prim.get("mode",4),mat_i)); all_positions.extend(positions)
        for child in node.get("children",[]):visit(child,local_world)
    for root in roots:visit(root,_identity())
    if not all_positions:raise ValueError("GLB contains no exportable triangle mesh")
    center=tuple((min(p[i] for p in all_positions)+max(p[i] for p in all_positions))*0.5 for i in range(3))

    obj=["# Pixal3D GLB conversion (stdlib)",f"mtllib {mtl_path.name}"]; vbase=vtbase=vnbase=0; vc=uc=nc=fc=0
    for node_index,pi,positions,uvs,normals,indices,mode,mat_i in prims:
        obj += [f"o node_{node_index}",f"g primitive_{node_index}_{pi}",f"usemtl {material_names[mat_i] if mat_i<len(material_names) else material_names[0]}"]
        for p in positions:obj.append(f"v {p[0]-center[0]:.9g} {p[1]-center[1]:.9g} {p[2]-center[2]:.9g}")
        for uv in uvs:obj.append(f"vt {uv[0]:.9g} {1.0-uv[1]:.9g}")
        for n in normals:obj.append(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}")
        for a,b,c in _triangles(indices,mode):
            def ref(k):
                vi=vbase+k+1; ti=vtbase+k+1 if uvs else None; ni=vnbase+k+1 if normals else None
                return f"{vi}/{ti}/{ni}" if ti is not None and ni is not None else f"{vi}/{ti}" if ti is not None else f"{vi}//{ni}" if ni is not None else str(vi)
            obj.append(f"f {ref(a)} {ref(b)} {ref(c)}"); fc+=1
        vbase+=len(positions); vtbase+=len(uvs); vnbase+=len(normals); vc+=len(positions); uc+=len(uvs); nc+=len(normals)
    output_path.write_text("\n".join(obj)+"\n",encoding="utf-8")
    report={"status":"ok","converter":"stdlib_glb_to_obj_v33","obj_path":str(output_path.resolve()),"mtl_path":str(mtl_path.resolve()),"texture_paths":texture_paths,"vertex_count":vc,"face_count":fc,"uv_count":uc,"normal_count":nc,"source_glb":str(input_path.resolve()),"pose_context_path":str(pose_context_path.resolve()) if pose_context_path else None,"camera_pose_correction_applied":pose_applied,"camera_pose_correction_method":"camera right/up/back frame; rotation only; local recenter" if pose_applied else "none","materials_in_glb":len(doc.get("materials",[])),"images_in_glb":len(doc.get("images",[])),"textures_in_glb":len(doc.get("textures",[]))}
    manifest_path.write_text(json.dumps(report,indent=2),encoding="utf-8"); return report


def main():
    a=_args(); pose=Path(a.pose_context).resolve() if a.pose_context else None; r=convert(Path(a.input).resolve(),Path(a.output).resolve(),Path(a.manifest).resolve(),pose)
    print(f"[Done] OBJ saved to: {r['obj_path']}"); print(f"[Done] vertices={r['vertex_count']} faces={r['face_count']} textures={len(r['texture_paths'])} pose_corrected={r['camera_pose_correction_applied']}")
if __name__=="__main__":main()
