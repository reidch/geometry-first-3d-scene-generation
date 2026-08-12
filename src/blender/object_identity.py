from __future__ import annotations

"""Blender-side identity contract for JSON-driven world objects.

The pipeline carries several identifiers with different jobs.  They must never be
collapsed into one overloaded custom property:

``world_object_id``
    Opaque, stable string copied from ``scene.objects[*].id`` in the input JSON.

``runtime_object_id``
    Integer index allocated by the compiled World IR registry.  It is suitable for
    dense runtime buffers, render/physics bindings, and array lookup only.

``semantic_owner_id``
    Opaque string naming the JSON object that owns semantic masks, UV atlases, and
    coverage state.  All scaffold parts or replacement meshes of one JSON object
    share this value.

Legacy aliases are written for v12 compatibility, but new code must read through
this module instead of interpreting ``object_id`` or Blender object names directly.
"""

from numbers import Integral
from typing import Any, Iterable

WORLD_OBJECT_ID_KEY = "world_object_id"
RUNTIME_OBJECT_ID_KEY = "runtime_object_id"
SEMANTIC_OWNER_ID_KEY = "semantic_owner_id"

# Transitional aliases retained so existing .blend files and external inspection
# scripts remain readable.  Their meanings are fixed here:
#   object_name -> world_object_id (string)
#   object_id   -> runtime_object_id (integer)
LEGACY_WORLD_OBJECT_ID_KEY = "object_name"
LEGACY_RUNTIME_OBJECT_ID_KEY = "object_id"


def _get(obj: Any, key: str, default=None):
    try:
        return obj.get(key, default)
    except Exception:
        try:
            return obj[key]
        except Exception:
            return default


def _set(obj: Any, key: str, value: Any) -> None:
    obj[key] = value


def _delete(obj: Any, key: str) -> None:
    try:
        if key in obj:
            del obj[key]
    except Exception:
        try:
            del obj[key]
        except Exception:
            pass


def _nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_world_object_id(obj: Any, *, required: bool = True) -> str | None:
    """Return the opaque JSON object ID without interpreting it as a number.

    A string-valued legacy ``object_id`` is accepted only as a migration fallback
    for v12 generated assets.  Blender's display name is deliberately *not* used as
    an identity fallback because it can be renamed or suffixed automatically.
    """

    for key in (WORLD_OBJECT_ID_KEY, LEGACY_WORLD_OBJECT_ID_KEY):
        value = _nonempty_string(_get(obj, key))
        if value is not None:
            return value

    legacy = _get(obj, LEGACY_RUNTIME_OBJECT_ID_KEY)
    if isinstance(legacy, str):
        value = _nonempty_string(legacy)
        if value is not None:
            return value

    if required:
        name = getattr(obj, "name", "<unnamed Blender object>")
        raise ValueError(
            f"Mesh {name!r} has no {WORLD_OBJECT_ID_KEY!r}; identity must come "
            "from the JSON/World IR rather than from its Blender name."
        )
    return None


def get_runtime_object_id(obj: Any, *, required: bool = False) -> int | None:
    """Return the dense integer World IR registry ID.

    Arbitrary strings are never passed to ``int``.  This is the invariant that
    prevents JSON IDs such as ``sleeping_unit`` from entering integer-only paths.
    """

    for key in (RUNTIME_OBJECT_ID_KEY, LEGACY_RUNTIME_OBJECT_ID_KEY):
        value = _get(obj, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)

    if required:
        world_id = get_world_object_id(obj, required=False)
        raise ValueError(
            f"World object {world_id!r} has no integer {RUNTIME_OBJECT_ID_KEY!r}."
        )
    return None


def get_semantic_owner_id(obj: Any, *, required: bool = True) -> str | None:
    """Return the object-owned semantic/atlas identity.

    In the current schema each renderable JSON object owns its own atlas, so the
    default is its ``world_object_id``.  The explicit property keeps the ownership
    layer independent from both mesh-part identity and dense runtime IDs.
    """

    value = _nonempty_string(_get(obj, SEMANTIC_OWNER_ID_KEY))
    if value is not None:
        return value
    return get_world_object_id(obj, required=required)


def set_object_identity(
    obj: Any,
    *,
    world_object_id: str,
    runtime_object_id: int | None,
    semantic_owner_id: str | None = None,
    write_legacy_aliases: bool = True,
) -> None:
    world_id = _nonempty_string(world_object_id)
    if world_id is None:
        raise ValueError("world_object_id must be a non-empty opaque string")
    owner_id = _nonempty_string(semantic_owner_id) or world_id

    _set(obj, WORLD_OBJECT_ID_KEY, world_id)
    _set(obj, SEMANTIC_OWNER_ID_KEY, owner_id)

    if runtime_object_id is not None:
        if isinstance(runtime_object_id, bool) or not isinstance(runtime_object_id, Integral):
            raise TypeError("runtime_object_id must be an integer or None")
        runtime_id = int(runtime_object_id)
        _set(obj, RUNTIME_OBJECT_ID_KEY, runtime_id)
    else:
        runtime_id = None
        _delete(obj, RUNTIME_OBJECT_ID_KEY)

    if write_legacy_aliases:
        _set(obj, LEGACY_WORLD_OBJECT_ID_KEY, world_id)
        if runtime_id is not None:
            _set(obj, LEGACY_RUNTIME_OBJECT_ID_KEY, runtime_id)
        else:
            _delete(obj, LEGACY_RUNTIME_OBJECT_ID_KEY)


def copy_object_identity(
    source: Any,
    target: Any,
    *,
    world_object_id: str | None = None,
    semantic_owner_id: str | None = None,
) -> None:
    """Copy the compiled identity from a scaffold part to a replacement mesh."""

    set_object_identity(
        target,
        world_object_id=world_object_id or get_world_object_id(source),
        runtime_object_id=get_runtime_object_id(source),
        semantic_owner_id=semantic_owner_id or get_semantic_owner_id(source),
    )


def assert_consistent_identity(objects: Iterable[Any], expected_world_id: str | None = None) -> dict:
    """Validate that a set of mesh parts belongs to one compiled JSON object."""

    objects = list(objects)
    if not objects:
        raise ValueError("Cannot validate an empty object collection")

    world_ids = {get_world_object_id(obj) for obj in objects}
    owners = {get_semantic_owner_id(obj) for obj in objects}
    runtime_ids = {get_runtime_object_id(obj) for obj in objects}

    if expected_world_id is not None and world_ids != {str(expected_world_id)}:
        raise ValueError(
            f"Scaffold identity mismatch: expected {expected_world_id!r}, got {sorted(world_ids)!r}"
        )
    if len(world_ids) != 1:
        raise ValueError(f"Mesh parts mix world object IDs: {sorted(world_ids)!r}")
    if len(owners) != 1:
        raise ValueError(f"Mesh parts mix semantic owners: {sorted(owners)!r}")
    if len(runtime_ids) != 1:
        raise ValueError(f"Mesh parts mix runtime object IDs: {sorted(runtime_ids, key=str)!r}")

    return {
        "world_object_id": next(iter(world_ids)),
        "semantic_owner_id": next(iter(owners)),
        "runtime_object_id": next(iter(runtime_ids)),
    }
