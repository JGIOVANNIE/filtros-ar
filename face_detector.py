import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Detector de caras y landmarks usando MediaPipe FaceMesh.
    Preparado para filtros 2D y 3D.
    """

    def __init__(
        self,
        static_image_mode=False,
        max_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        draw_landmarks=False
    ):
        self.draw_landmarks_enabled = draw_landmarks

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.drawing_spec = self.mp_styles.get_default_face_mesh_tesselation_style()

    # --------------------------------------------------
    # Detección principal
    # --------------------------------------------------
    def detect(self, frame):
        """
        Detecta caras y devuelve landmarks.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        return result.multi_face_landmarks or []

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------
    @staticmethod
    def landmark_to_px(landmark, img_w, img_h):
        """
        Convierte un landmark normalizado a coordenadas de píxel.
        """
        return int(landmark.x * img_w), int(landmark.y * img_h)

    def get_points_px(self, frame, landmarks, indices):
        """
        Devuelve puntos específicos del rostro en píxeles.
        Útil para filtros 2D y solvePnP.
        """
        h, w = frame.shape[:2]
        return np.array(
            [self.landmark_to_px(landmarks.landmark[i], w, h) for i in indices],
            dtype=np.float32
        )

    # --------------------------------------------------
    # Debug visual
    # --------------------------------------------------
    def draw(self, frame, face_landmarks_list):
        """
        Dibuja la malla facial (opcional).
        """
        if not self.draw_landmarks_enabled:
            return frame

        for landmarks in face_landmarks_list:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_spec
            )
        return frame
