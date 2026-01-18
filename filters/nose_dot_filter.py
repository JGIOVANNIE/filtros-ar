import cv2
from filters.base_filter import BaseFilter

class NoseDotFilter(BaseFilter):
    def __init__(self, radius=5, color=(0, 0, 255)):
        super().__init__()  # 👈 NUNCA pasarle argumentos

        self.radius = radius
        self.color = color

    def apply(self, frame, landmarks):
        h, w, _ = frame.shape

        # Landmark punta de la nariz (MediaPipe)
        nose = landmarks.landmark[1]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.circle(frame, (x, y), self.radius, self.color, -1)
        return frame
