import cv2

class CameraManager:
    """
    Clase para manejar la cámara en tiempo real.
    Permite abrir la cámara, capturar frames y liberarla correctamente.
    """

    def __init__(self, camera_index=0):
        """
        Inicializa la cámara.
        :param camera_index: ID de la cámara (0 por defecto = webcam principal)
        """
        # Crea un objeto de captura de video con OpenCV
        self.cap = cv2.VideoCapture(camera_index)

    def get_frame(self):
        """
        Captura un frame de la cámara.
        Devuelve el frame espejeado para una experiencia tipo espejo.
        :return: frame (imagen BGR) o None si falla la captura
        """
        ret, frame = self.cap.read()  # Intenta leer un nuevo cuadro
        if not ret:
            return None

        # Espejea horizontalmente para que se vea como un espejo
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        """
        Libera la cámara y destruye todas las ventanas de OpenCV.
        """
        self.cap.release()          # Libera el acceso a la cámara
        cv2.destroyAllWindows()     # Cierra todas las ventanas creadas por OpenCV
