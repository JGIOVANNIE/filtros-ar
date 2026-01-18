import cv2
import math
from .base_filter import BaseFilter

class EyeFilter(BaseFilter):
    def __init__(self, asset_paths, scale_factor=1.0, y_offset_ratio=0.0):
        super().__init__()

        self.assets = [
            cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in asset_paths
        ]
        self.asset_index = 0
        self.scale_factor = scale_factor
        self.y_offset_ratio = y_offset_ratio
        self.offset_x = 0

    def next_asset(self):
        self.asset_index = (self.asset_index + 1) % len(self.assets)

    def add_offset_x(self, dx):
        self.offset_x += dx

    def apply(self, frame, landmarks):
        asset = self.assets[self.asset_index]
        if asset is None:
            return frame

        h, w, _ = frame.shape

        # Landmarks ojos
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]

        xL, yL = int(left_eye.x * w), int(left_eye.y * h)
        xR, yR = int(right_eye.x * w), int(right_eye.y * h)

        span = math.sqrt((xR - xL) ** 2 + (yR - yL) ** 2)

        cx = (xL + xR) // 2
        cy = (yL + yR) // 2

        # Escalado
        filter_width = int(span * self.scale_factor)
        aspect = asset.shape[0] / asset.shape[1]
        filter_height = int(filter_width * aspect)

        resized = cv2.resize(asset, (filter_width, filter_height))

        x = cx - filter_width // 2 + self.offset_x
        y = int(cy - filter_height // 2 + span * self.y_offset_ratio)

        self._overlay_rgba(frame, resized, x, y)
        return frame

    def _overlay_rgba(self, frame, rgba, x, y):
        fh, fw, _ = rgba.shape
        h, w, _ = frame.shape

        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + fw, w)
        y2 = min(y + fh, h)

        if x1 >= x2 or y1 >= y2:
            return

        roi = frame[y1:y2, x1:x2]
        filter_roi = rgba[y1 - y:y2 - y, x1 - x:x2 - x]

        alpha = filter_roi[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (
                alpha * filter_roi[:, :, c] +
                (1 - alpha) * roi[:, :, c]
            )
