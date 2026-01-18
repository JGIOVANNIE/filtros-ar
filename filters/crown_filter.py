import cv2
import numpy as np
from filters.base_filter import BaseFilter


class CrownFilter(BaseFilter):
    def __init__(self, asset_path, scale_factor=1.4):
        super().__init__()
        self.crown = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        self.scale_factor = scale_factor

    def apply(self, frame, landmarks):
        h, w, _ = frame.shape

        # Landmarks clave
        left = landmarks.landmark[127]
        right = landmarks.landmark[356]
        top = landmarks.landmark[10]
        chin = landmarks.landmark[152]

        xL, yL = int(left.x * w), int(left.y * h)
        xR, yR = int(right.x * w), int(right.y * h)
        xT, yT = int(top.x * w), int(top.y * h)
        _, yC = int(chin.x * w), int(chin.y * h)

        # Ancho del rostro
        face_width = abs(xR - xL)
        crown_width = int(face_width * self.scale_factor)

        # Escalar corona
        ch, cw = self.crown.shape[:2]
        scale = crown_width / cw
        crown_resized = cv2.resize(
            self.crown,
            (int(cw * scale), int(ch * scale))
        )

        new_h, new_w = crown_resized.shape[:2]

        # 🔑 Altura del rostro
        face_height = abs(yC - yT)

        # 🔑 Margen automático
        margin = int(face_height * 0.15)

        # Posición final
        cx = (xL + xR) // 2
        x1 = cx - new_w // 2
        y1 = yT - new_h + margin

        # Seguridad de bordes
        if x1 < 0 or y1 < 0 or x1 + new_w > w or y1 + new_h > h:
            return frame

        # Mezcla alpha
        alpha = crown_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y1:y1+new_h, x1:x1+new_w, c] = (
                alpha * crown_resized[:, :, c] +
                (1 - alpha) * frame[y1:y1+new_h, x1:x1+new_w, c]
            )

        return frame
