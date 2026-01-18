from dataclasses import dataclass
from typing import List


@dataclass
class FilterItem:
    name: str
    instance: object
    enabled: bool = True


class FilterPipeline:
    def __init__(self):
        self.items: List[FilterItem] = []

    def add(self, name: str, instance: object, enabled: bool = True):
        item = FilterItem(name, instance, enabled)
        self.items.append(item)

    def set_enabled(self, name: str, enabled: bool):
        for item in self.items:
            if item.name == name:
                item.enabled = enabled
                return

    def apply(self, frame, landmarks):
        for item in self.items:
            if item.enabled:
                frame = item.instance.apply(frame, landmarks)
        return frame

    def move(self, name: str, new_index: int):
        for i, item in enumerate(self.items):
            if item.name == name:
                self.items.pop(i)
                self.items.insert(new_index, item)
                return
