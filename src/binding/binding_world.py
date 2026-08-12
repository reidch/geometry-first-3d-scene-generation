from __future__ import annotations
from dataclasses import dataclass, field
from src.binding.binding_record import BindingRecord

@dataclass
class BindingWorld:
    records: list[BindingRecord] = field(default_factory=list)

    def add(self, record: BindingRecord):
        self.records.append(record)
        return record.binding_id

    def to_list(self):
        return [r.to_dict() for r in self.records]
