def select_best_generated_reference(camera_id,generated_ids,graph):
    c=[]
    for e in graph.get('edges',[]):
        other=None
        if e['source']==camera_id:other=e['target']
        elif e['target']==camera_id:other=e['source']
        if other in generated_ids:
            s=.65*float(e.get('approx_overlap',0))+.35*float(e.get('rotation_similarity',0)); c.append((s,other,e))
    if not c:return None
    c.sort(reverse=True,key=lambda x:x[0]); s,o,e=c[0]
    return {'camera_id':o,'score':s,'approx_overlap':e.get('approx_overlap'),'rotation_similarity':e.get('rotation_similarity')}
