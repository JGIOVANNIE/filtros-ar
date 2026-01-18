import cv2
import numpy as np
import trimesh
from filters.base_filter import BaseFilter

MP_IDXS = [33, 263, 1, 61, 291, 199]

CANON_FACE_3D = np.array([
    [-0.03,  0.04, -0.02],
    [ 0.03,  0.04, -0.02],
    [ 0.00,  0.00,  0.00],
    [-0.02, -0.03, -0.02],
    [ 0.02, -0.03, -0.02],
    [ 0.00, -0.06, -0.01],
], dtype=np.float32)


class Base3DFilter(BaseFilter):

    def __init__(self, model_path, color=(255,170,80), alpha=0.7, scale=1.0):  # ✅
        super().__init__()

        self.color = color
        self.alpha = alpha

        # Aplicar scale al cargar la malla
        target_size = 0.12 * scale  # ✅
        self.verts, self.faces = self.load_mesh_norm(model_path, target_size=target_size)

        self.K = None
        self.dist = np.zeros((4, 1))
        self.rvec = None
        self.tvec = None
        self.smoothing_factor = 0.5

    # ---------- Mesh ----------
    def load_mesh_norm(self, path, target_size=0.12):
        mesh = trimesh.load(path, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                tuple(mesh.geometry.values())
            )

        V = np.array(mesh.vertices, dtype=np.float32)
        F = np.array(mesh.faces, dtype=np.int32)

        V -= V.mean(axis=0)
        size = np.linalg.norm(V.max(0) - V.min(0))
        V *= target_size / size

        return V, F

    # ---------- Cámara ----------
    def K_from_frame(self, frame):
        h, w = frame.shape[:2]
        f = 1.0 * w
        return np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    # ---------- MediaPipe ----------
    def mp_to_px(self, w, h, lm, idx):
        p = lm.landmark[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    def face_points_2d(self, frame, lm):
        h, w = frame.shape[:2]
        return np.array(
            [self.mp_to_px(w, h, lm, i) for i in MP_IDXS],
            dtype=np.float32
        )

    # ---------- Suavizado ----------
    def smooth_pose(self, rvec, tvec):
        if self.rvec is None or self.tvec is None:
            self.rvec = rvec.copy()
            self.tvec = tvec.copy()
        else:
            self.rvec = self.rvec * self.smoothing_factor + rvec * (1 - self.smoothing_factor)
            self.tvec = self.tvec * self.smoothing_factor + tvec * (1 - self.smoothing_factor)
        
        return self.rvec, self.tvec

    # ---------- Proyección ----------
    def project(self, verts, rvec, tvec, K):
        pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, self.dist)
        R, _ = cv2.Rodrigues(rvec)
        X_cam = (R @ verts.T).T + tvec.T
        z_cam = X_cam[:, 2]
        return pts2d.reshape(-1, 2), z_cam

    # ---------- Dibujo ----------
    def draw(self, frame, pts2d, faces, z_cam):
        overlay = frame.copy()
        face_depths = z_cam[faces].mean(axis=1)
        order = np.argsort(face_depths)

        for idx in order:
            poly = pts2d[faces[idx]].astype(np.int32)
            if np.all((poly >= 0) & (poly < [frame.shape[1], frame.shape[0]])):
                cv2.fillConvexPoly(overlay, poly, self.color)
                cv2.polylines(overlay, [poly], True, (50, 50, 50), 1)

        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

    # ---------- APPLY ----------
    def apply(self, frame, landmarks):
        img_pts = self.face_points_2d(frame, landmarks)

        if self.K is None:
            self.K = self.K_from_frame(frame)

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
        pts2d, z_cam = self.project(self.verts, rvec, tvec, self.K)
        self.draw(frame, pts2d, self.faces, z_cam)

        return frame