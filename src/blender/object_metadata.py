from __future__ import annotations

def set_metadata(obj, **kwargs) -> None:
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            obj[key] = value
        else:
            obj[key] = str(value)

def collect_metadata(obj) -> dict:
    keys = [
        "world_object_id",
        "runtime_object_id",
        "semantic_owner_id",
        "object_id",
        "object_name",
        "render_id",
        "physics_id",
        "binding_id",
        "semantic_class",
        "physical_type",
        "part_id",
        "primitive",
        "material_id",
    ]
    return {k: obj.get(k) for k in keys if k in obj}
