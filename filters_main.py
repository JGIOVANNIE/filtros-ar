# importando BaseFilter desde la carpeta filters
from filters.base_filter import BaseFilter
import cv2
import numpy as np

class MustacheFilter:
    """
    Filtro de mostacho que se coloca sobre la punta de la nariz usando landmarks de MediaPipe.
    """

    def __init__(self, png_path, target_width_px=60, y_offset_px=10):
        """
        Constructor del filtro de mostacho.

        :param png_path: Ruta al PNG del mostacho (con transparencia)
        :param target_width_px: ancho objetivo en píxeles del mostacho
        :param y_offset_px: desplazamiento vertical respecto a la nariz
        """
        # Cargar imagen PNG con canal alfa
        self.overlay_rgba = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_rgba is None:
            raise ValueError(f"No se pudo cargar la imagen: {png_path}")
        if self.overlay_rgba.shape[2] != 4:
            raise ValueError("La imagen PNG debe tener 4 canales (RGBA).")

        # Guardar parámetros
        self.target_width_px = target_width_px
        self.y_offset_px = int(y_offset_px)

    def landmark_to_xy(self, landmark, frame_shape):
        """
        Convierte un landmark normalizado (0-1) a coordenadas de píxeles.
        :param landmark: landmark de MediaPipe
        :param frame_shape: shape del frame (alto, ancho)
        :return: (x, y) en píxeles
        """
        h, w = frame_shape[:2]
        return int(landmark.x * w), int(landmark.y * h)

    def overlay_rgba_on_bgr(self, frame, overlay, x, y):
        """
        Superpone una imagen RGBA sobre un frame BGR con transparencia.
        :param frame: imagen BGR
        :param overlay: imagen RGBA
        :param x: coordenada x superior izquierda
        :param y: coordenada y superior izquierda
        """
        h, w = overlay.shape[:2]

        # Limitar los bordes para no salir de la pantalla
        y1, y2 = max(0, y), min(frame.shape[0], y + h)
        x1, x2 = max(0, x), min(frame.shape[1], x + w)

        overlay_crop = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
        if overlay_crop.size == 0:
            return  # nada que dibujar

        rgb = overlay_crop[..., :3]
        alpha = overlay_crop[..., 3:4] / 255.0

        roi = frame[y1:y2, x1:x2]
        blended = alpha * rgb + (1 - alpha) * roi
        roi[:] = blended.astype(np.uint8)

    def apply(self, frame, face_landmarks):
        """
        Aplica el filtro del mostacho sobre un rostro detectado.
        :param frame: frame BGR
        :param face_landmarks: landmarks de MediaPipe de un rostro
        """
        # Punto de la nariz (landmark 1)
        x_nose, y_nose = self.landmark_to_xy(face_landmarks.landmark[1], frame.shape)

        # Escalar PNG según ancho objetivo
        oh, ow = self.overlay_rgba.shape[:2]
        scale = self.target_width_px / ow
        new_w, new_h = int(ow * scale), int(oh * scale)
        overlay_resized = cv2.resize(self.overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calcular posición superior izquierda centrada bajo la nariz
        x = x_nose - new_w // 2
        y = y_nose + self.y_offset_px - new_h // 2

        # Superponer sobre el frame
        self.overlay_rgba_on_bgr(frame, overlay_resized, x, y)
        return frame


class NoseDotFilter(BaseFilter):
    def apply(self):
        if not self.landmarks:
            return self.frame

        for face_landmarks in self.landmarks:
            # El punto 1 es la punta de la nariz en FaceMesh
            nose_tip = face_landmarks.landmark[1]
            h, w, _ = self.frame.shape
            x, y = int(nose_tip.x * w), int(nose_tip.y * h)

            cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)

        return self.frame
