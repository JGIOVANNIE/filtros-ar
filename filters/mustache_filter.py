import cv2
from .base_filter import BaseFilter

class MustacheFilter(BaseFilter):
    def __init__(self, asset_path, target_width_px=100, y_offset_px=0):
        super().__init__()

        self.asset = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        self.target_width_px = target_width_px
        self.y_offset_px = y_offset_px

    def apply(self, frame, landmarks):
        if self.asset is None:
            return frame

        h, w, _ = frame.shape

        # ✅ Landmark nariz (MediaPipe)
        nose = landmarks.landmark[1]

        x = int(nose.x * w)
        y = int(nose.y * h) + self.y_offset_px

        # Escalar mostacho
        scale = self.target_width_px / self.asset.shape[1]
        new_w = self.target_width_px
        new_h = int(self.asset.shape[0] * scale)

        resized = cv2.resize(self.asset, (new_w, new_h))

        x -= new_w // 2
        y -= new_h // 2

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
