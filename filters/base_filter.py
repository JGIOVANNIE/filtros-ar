from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """
    Clase base para filtros faciales.
    Todas las subclases deben implementar el método apply.
    """

    def __init__(self, frame, landmarks):
        self.frame = frame
        self.landmarks = landmarks

    @abstractmethod
    def apply(self):
        """
        Método que aplica el filtro sobre el frame.
        Debe implementarse en cada filtro específico.
        """
        pass