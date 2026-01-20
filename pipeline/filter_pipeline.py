from dataclasses import dataclass
from typing import List, Protocol


# --------------------------------------------------
# Interfaz de filtro (2D o 3D)
# --------------------------------------------------
class FilterProtocol(Protocol):
    def apply(self, frame, landmarks):
        ...


# --------------------------------------------------
# Item del pipeline
# --------------------------------------------------
@dataclass
class FilterItem:
    name: str
    instance: FilterProtocol
    enabled: bool = True
    order: int = 0   # permite ordenar explícitamente


# --------------------------------------------------
# Pipeline
# --------------------------------------------------
class FilterPipeline:
    def __init__(self):
        self.items: List[FilterItem] = []

    # --------------------------------------------------
    # Añadir filtro
    # --------------------------------------------------
    def add(self, name: str, instance: FilterProtocol, enabled: bool = True, order: int = 0):
        if not hasattr(instance, "apply"):
            raise ValueError(f"El filtro '{name}' no tiene método apply()")

        self.items.append(FilterItem(name, instance, enabled, order))
        self._sort()

    # --------------------------------------------------
    # Orden interno
    # --------------------------------------------------
    def _sort(self):
        self.items.sort(key=lambda f: f.order)

    # --------------------------------------------------
    # Activar / desactivar
    # --------------------------------------------------
    def set_enabled(self, name: str, enabled: bool):
        for item in self.items:
            if item.name == name:
                item.enabled = enabled
                return

    def toggle(self, name: str):
        for item in self.items:
            if item.name == name:
                item.enabled = not item.enabled
                return

    def is_enabled(self, name: str) -> bool:
        for item in self.items:
            if item.name == name:
                return item.enabled
        return False

    # --------------------------------------------------
    # Aplicar pipeline
    # --------------------------------------------------
    def apply(self, frame, landmarks):
        for item in self.items:
            if not item.enabled:
                continue

            try:
                frame = item.instance.apply(frame, landmarks)
            except Exception as e:
                print(f"[WARN] Filtro '{item.name}' falló: {e}")

        return frame

    # --------------------------------------------------
    # Reordenar manualmente
    # --------------------------------------------------
    def move(self, name: str, new_index: int):
        for i, item in enumerate(self.items):
            if item.name == name:
                self.items.pop(i)
                self.items.insert(new_index, item)
                return
