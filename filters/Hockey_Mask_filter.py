import cv2
import numpy as np
import trimesh
from filters.base_3d_filter import Base3DFilter

class HockeyMaskFilter(Base3DFilter):
    def __init__(self, model_path="assets/Hockey_Mask.obj", scale=1.0, alpha=0.95):
        super().__init__
        # Cargar modelo
        self.mesh = trimesh.load(model_path, process=False)
        self.scale = scale
        self.alpha = alpha

        # Vértices, caras y colores
        self.verts = np.array(self.mesh.vertices)
        self.faces = np.array(self.mesh.faces)
        if hasattr(self.mesh.visual, "vertex_colors"):
            self.colors = np.array(self.mesh.visual.vertex_colors[:, :3])
        else:
            self.colors = None

        # Offset para subir casco hacia arriba de la cabeza
        self.offset_3d = np.array([0.0, 0.08, 0.0], dtype=np.float32)  # ajustable

        # Inclinación hacia atrás en radianes
        self.angle_back_deg = 10
        angle_back = np.deg2rad(self.angle_back_deg)
        self.Rx_back = np.array([
            [1, 0, 0],
            [0, np.cos(angle_back), -np.sin(angle_back)],
            [0, np.sin(angle_back), np.cos(angle_back)]
        ])

        # Rotar 180° sobre X si tu obj lo requiere
        Rx_flip = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        self.verts = self.verts @ Rx_flip.T

    def rotate_vertices(self, verts, landmarks):
        """
        Rotación 3D completa: yaw (horizontal) + pitch (inclinación adelante/atrás) + inclinación hacia atrás
        """
        # Rotación horizontal (yaw) según ojos
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        angle_z = np.arctan2(dy, dx)
        cos_z, sin_z = np.cos(-angle_z), np.sin(-angle_z)
        Rz = np.array([[cos_z, -sin_z, 0],
                    [sin_z,  cos_z, 0],
                    [0, 0, 1]])
        verts_rot = verts @ Rz.T

        # Rotación pitch (inclinación adelante/atrás) según landmarks top y mentón
        top_head = landmarks.landmark[10]      # punta superior cabeza
        chin = landmarks.landmark[152]         # mentón
        pitch = np.arctan2(chin.y - top_head.y, chin.z - top_head.z)
        Rx_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])
        verts_rot = verts_rot @ Rx_pitch.T

        # Inclinación ligera hacia atrás
        verts_rot = verts_rot @ self.Rx_back.T
        return verts_rot

    def project_vertices(self, verts, cx, cy, scale_factor):
        projected = np.zeros((verts.shape[0], 2), dtype=np.int32)
        projected[:, 0] = (verts[:, 0] * scale_factor + cx).astype(int)
        projected[:, 1] = (verts[:, 1] * scale_factor + cy).astype(int)
        return projected

    def apply(self, frame, landmarks):
        if landmarks is None:
            return frame

        h, w, _ = frame.shape

        # 1️⃣ Centro del casco: punta superior de la cabeza
        top_head = landmarks.landmark[10]
        cx = int(top_head.x * w)
        cy = int(top_head.y * h)

        # 2️⃣ Anchura cara para escala
        left = landmarks.landmark[234]
        right = landmarks.landmark[454]
        face_width_px = np.linalg.norm(
            np.array([left.x * w, left.y * h]) -
            np.array([right.x * w, right.y * h])
        )

        scale_factor = face_width_px * self.scale * 0.06  # un poco más grande que antes

        # 3️⃣ Offset + profundidad
        verts_offset = self.verts + self.offset_3d
        verts_offset[:, 2] *= 0.8  # reduce ligeramente la profundidad

        # 4️⃣ Rotar vértices según cabeza
        verts_rotated = self.rotate_vertices(verts_offset, landmarks)

        # 5️⃣ Proyección 2D
        pts2d = self.project_vertices(verts_rotated, cx, cy, scale_factor)

        # 6️⃣ Renderizado
        overlay = frame.copy()
        for face in self.faces:
            poly = pts2d[face]
            if np.any(poly < 0):
                continue
            color = (200, 200, 200) if self.colors is None else np.mean(self.colors[face], axis=0).astype(int).tolist()
            cv2.fillConvexPoly(overlay, poly, color)

        # 7️⃣ Transparencia
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        return frame