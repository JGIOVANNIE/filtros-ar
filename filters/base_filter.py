from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """
    Clase base para filtros faciales.
    Todas las subclases deben implementar el método apply.
    """

    def __init__(self):
        pass

    @abstractmethod
    def apply(self, frame, landmarks):
        """
        Método que aplica el filtro sobre el frame.
        Debe implementarse en cada filtro específico.
        """
        pass