from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.scene_ir.object_record import ObjectRecord


@dataclass
class ObjectRegistry:
    records: List[ObjectRecord] = field(default_factory=list)
    name_to_id: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        name: str,
        semantic_class: str,
        *,
        display_name: str | None = None,
        parent_id: int | None = None,
        generation_mode: str = "scaffold_only",
    ) -> int:
        if name in self.name_to_id:
            raise ValueError(f"Duplicate object name in registry: {name}")
        object_id = len(self.records)
        self.records.append(
            ObjectRecord(
                object_id=object_id,
                name=name,
                display_name=str(display_name or name),
                semantic_class=semantic_class,
                parent_id=parent_id,
                generation_mode=generation_mode,
            )
        )
        self.name_to_id[name] = object_id
        if parent_id is not None:
            self.records[parent_id].child_ids.append(object_id)
        return object_id

    def get(self, object_id: int) -> ObjectRecord:
        return self.records[object_id]

    def find(self, name: str) -> Optional[ObjectRecord]:
        object_id = self.name_to_id.get(name)
        return None if object_id is None else self.records[object_id]

    def to_list(self):
        return [record.to_dict() for record in self.records]
