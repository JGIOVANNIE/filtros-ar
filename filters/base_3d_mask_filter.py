import numpy as np
import cv2
from filters.base_3d_filter import Base3DFilter


class Base3DMaskFilter(Base3DFilter):
    def __init__(self, model_path, scale=1.0, alpha=0.9):
        super().__init__(
            model_path=model_path,
            scale=scale,
            alpha=alpha
        )

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def apply(self, frame, face_landmarks):
        if face_landmarks is None:
            return frame

        landmarks = face_landmarks.landmark
        h, w = frame.shape[:2]

        # Inicializar cámara virtual
        if self.camera_matrix is None:
            self.camera_matrix = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1]
            ], dtype=np.float32)

        rvec, tvec = self.get_transform(landmarks, frame.shape)

        # Proyección 3D → 2D (OpenCV directo, sin wrappers)
        pts2d, _ = cv2.projectPoints(
            self.verts,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        pts2d = pts2d.reshape(-1, 2).astype(np.int32)

        z_cam = self.verts[:, 2] + tvec[2][0]

        self.draw_mesh(frame, pts2d, z_cam)

        return frame

    def draw_mesh(self, frame, pts2d, z_cam):
        # Dibujar caras ordenadas por profundidad
        face_depth = np.mean(z_cam[self.faces], axis=1)
        order = np.argsort(face_depth)[::-1]

        for i in order:
            face = self.faces[i]
            pts = pts2d[face]

            cv2.fillConvexPoly(
                frame,
                pts,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA
            )

    def get_transform(self, landmarks, image_shape):
        h, w = image_shape[:2]

        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        eye_dist = np.linalg.norm([
            (left_eye.x - right_eye.x) * w,
            (left_eye.y - right_eye.y) * h
        ])

        scale_z = eye_dist * 4.5

        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[0], [0], [scale_z]], dtype=np.float32)

        return rvec, tvec
