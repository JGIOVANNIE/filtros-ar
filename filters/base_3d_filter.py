import cv2
import numpy as np
import trimesh

# --------------------------
# Puntos MediaPipe para pose
# --------------------------
MP_IDXS = [
    1,     # nariz
    33,    # ojo izquierdo
    263,   # ojo derecho
    234,   # sien izquierda
    454,   # sien derecha
    152    # mentón (nuevo punto, mínimo 6 para solvePnP)
]

# --------------------------
# Modelo facial canónico 3D
# --------------------------
CANON_FACE_3D = np.array([
    [0.0, 0.0, 0.0],       # nariz
    [-0.03, 0.04, -0.02],  # ojo izq
    [0.03, 0.04, -0.02],   # ojo der
    [-0.06, 0.0, -0.04],   # sien izq
    [0.06, 0.0, -0.04],    # sien der
    [0.0, -0.06, -0.02],   # mentón
], dtype=np.float32)

class Base3DFilter:
    def __init__(self, model_path, alpha=0.9, scale_base=0.08, offset_3d=(0.0, 0.0, -0.02)):
        self.alpha = alpha
        self.scale_base = scale_base
        self.offset_3d = np.array(offset_3d, dtype=np.float32)

        # Cargar modelo 3D
        self.verts, self.faces, self.colors = self.load_mesh_norm(model_path, 1.0)

        # Cámara
        self.K = None
        self.dist = np.zeros((4, 1), dtype=np.float32)

        # Suavizado
        self.smoothing = 0.6
        self._rvec_prev = None
        self._tvec_prev = None

    # --------------------------
    # Cargar y normalizar modelo
    # --------------------------
    def load_mesh_norm(self, model_path, target_size):
        mesh = trimesh.load(model_path, process=False)
        if hasattr(mesh, "geometry"):
            mesh = list(mesh.geometry.values())[0]

        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces
        colors = mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, "vertex_colors") else None

        # Normalizar
        verts -= verts.mean(axis=0)
        max_range = np.max(np.linalg.norm(verts, axis=1))
        verts /= max_range
        verts *= target_size

        return verts, faces, colors

    # --------------------------
    # Extraer puntos 2D
    # --------------------------
    def extract_face_points(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        pts = []
        for idx in MP_IDXS:
            lm = landmarks.landmark[idx]
            pts.append([lm.x * w, lm.y * h])
        return np.array(pts, dtype=np.float32)

    # --------------------------
    # Matriz de cámara aproximada
    # --------------------------
    def K_from_frame(self, frame):
        h, w = frame.shape[:2]
        f = w
        return np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    # --------------------------
    # Suavizado de pose
    # --------------------------
    def smooth_pose(self, rvec, tvec):
        if self._rvec_prev is None:
            self._rvec_prev = rvec
            self._tvec_prev = tvec
            return rvec, tvec

        rvec_s = self.smoothing * self._rvec_prev + (1 - self.smoothing) * rvec
        tvec_s = self.smoothing * self._tvec_prev + (1 - self.smoothing) * tvec

        self._rvec_prev = rvec_s
        self._tvec_prev = tvec_s
        return rvec_s, tvec_s

    # --------------------------
    # Dibujar modelo 3D sobre frame
    # --------------------------
    def draw(self, frame, pts2d, faces, z_cam):
        overlay = frame.copy()
        order = np.argsort(z_cam[faces].mean(axis=1))[::-1]

        for i in order:
            face = faces[i]
            poly = pts2d[face].astype(np.int32)
            if np.any(poly < 0):
                continue
            color = np.mean(self.colors[face], axis=0).astype(int).tolist() if self.colors is not None else (200, 200, 200)
            cv2.fillConvexPoly(overlay, poly, color)

        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

    # --------------------------
    # Aplicar filtro
    # --------------------------
    def apply(self, frame, landmarks):
        if landmarks is None:
            return frame

        img_pts = self.extract_face_points(landmarks, frame.shape)
        if img_pts.shape[0] != len(CANON_FACE_3D):
            return frame

        if self.K is None:
            self.K = self.K_from_frame(frame)

        # Pose estimation
        ok, rvec, tvec = cv2.solvePnP(
            CANON_FACE_3D,
            img_pts,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return frame

        rvec, tvec = self.smooth_pose(rvec, tvec)

        # Escala proporcional a la cabeza
        left = landmarks.landmark[234]
        right = landmarks.landmark[454]
        face_width_px = np.linalg.norm(
            np.array([left.x * frame.shape[1], left.y * frame.shape[0]]) -
            np.array([right.x * frame.shape[1], right.y * frame.shape[0]])
        )

        scale = 2 * (face_width_px / frame.shape[1])  # ajustar scale_base

        verts_scaled = (self.verts + self.offset_3d) * scale

        # Centrar el modelo en la cabeza
        tvec[0] = 0.0
        tvec[1] = 0.0
        tvec[2] = 0.15
        # Proyección 3D -> 2D
        pts2d, _ = cv2.projectPoints(
            verts_scaled,
            rvec,
            tvec,
            self.K,
            self.dist
        )
        pts2d = pts2d.reshape(-1, 2)
        z_cam = verts_scaled[:, 2] + tvec[2]

        # Dibujar modelo
        self.draw(frame, pts2d, self.faces, z_cam)

        return frame
