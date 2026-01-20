import cv2


class CameraManager:
    """
    Maneja la cámara de forma robusta para proyectos AR.
    """

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = self._open_camera()

    # --------------------------------------------------
    # Apertura segura de cámara
    # --------------------------------------------------
    def _open_camera(self):
        """
        Intenta abrir la cámara con distintos backends (Windows friendly).
        """
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY
        ]

        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print(f"[INFO] Cámara abierta con backend {backend}")
                return cap

        raise RuntimeError("❌ No se pudo abrir la cámara")

    # --------------------------------------------------
    # Captura de frame
    # --------------------------------------------------
    def get_frame(self):
        """
        Captura un frame espejeado.
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return cv2.flip(frame, 1)

    # --------------------------------------------------
    # Liberar recursos
    # --------------------------------------------------
    def release(self):
        """
        Libera cámara y ventanas.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
