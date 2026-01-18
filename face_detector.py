import cv2
import mediapipe as mp

class FaceDetector:
    """
    Clase para detectar caras y landmarks faciales usando MediaPipe FaceMesh.
    """

    def __init__(self, static_image_mode=False, max_faces=1,
                 refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        # Guardamos el módulo FaceMesh de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        # Creamos el objeto detector FaceMesh con los parámetros
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Guardamos el módulo para dibujar landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        # Ajuste para la versión actual de MediaPipe
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

    def detect(self, frame):
        """
        Detecta caras y landmarks en un frame.
        :param frame: imagen BGR
        :return: lista de mallas faciales
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        return result.multi_face_landmarks

    def draw(self, frame, face_landmarks_list):
        """
        Dibuja los contornos de la malla facial en el frame.
        :param frame: imagen BGR
        :param face_landmarks_list: lista de landmarks detectados
        :return: frame con landmarks dibujados
        """
        if face_landmarks_list:
            for landmarks in face_landmarks_list:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec
                )
        return frame
