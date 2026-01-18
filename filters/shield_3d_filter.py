import trimesh
import numpy as np
import cv2
from filters.base_filter import BaseFilter

class Shield3DFilter(BaseFilter):
    def __init__(self, model_path, scale=1.0):
        super().__init__()
        self.mesh = trimesh.load(model_path)
        self.scale = scale

        # Vértices del modelo
        self.vertices = np.array(self.mesh.vertices)

    def project(self, verts, cx, cy, scale):
        projected = []
        for x, y, z in verts:
            px = int(cx + x * scale)
            py = int(cy - y * scale)
            projected.append((px, py))
        return projected

    def apply(self, frame, landmarks):
        h, w, _ = frame.shape

        # Punto de anclaje: centro de la cara
        nose = landmarks.landmark[1]
        cx = int(nose.x * w)
        cy = int(nose.y * h)

        verts = self.vertices * self.scale
        projected = self.project(verts, cx, cy, scale=100)

        # Dibujar puntos
        for x, y in projected:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        return frame
