import cv2
from face_detector import FaceDetector
from pipeline.filter_pipeline import FilterPipeline

from filters.eye_filter import EyeFilter
from filters.mustache_filter import MustacheFilter
from filters.nose_dot_filter import NoseDotFilter
from filters.crown_filter import CrownFilter
from filters.hat_filter import HatFilter
from filters.shield_3d_filter import Shield3DFilter
from filters.Gas_Mask_Filter import GasMaskFilter
from filters.Hockey_Mask_filter import HockeyMaskFilter


class App:
    def __init__(self):
        self.cap = None
        self.detector = None
        self.pipeline = None
        self.running = True

        self.available_cams = self._detect_cameras()
        self.current_cam_index = self._find_camo() or 0
        print(f"📷 Cámara inicial: {self.current_cam_index}")

        self._init_camera(self.current_cam_index)
        self._init_detector()
        self._init_filters()
        self._init_pipeline()

    # -------------------------------------------------
    # Inicializaciones
    # -------------------------------------------------
    def _detect_cameras(self, max_tested=5):
        """Devuelve una lista de cámaras disponibles"""
        cams = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams

    def _find_camo(self):
        """Busca automáticamente una cámara Camo si está conectada"""
        for i in self.available_cams:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ret, _ = cap.read()
            cap.release()
            # OpenCV no siempre permite leer el nombre de la cámara,
            # así que asumimos que si hay más de 1 cámara disponible,
            # la Camo puede ser la segunda (i=1)
            if i != 0:
                return i
        return None

    def _init_camera(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError("❌ No se pudo abrir la cámara")

    def switch_camera(self):
        """Cambia a la siguiente cámara disponible"""
        if not self.available_cams:
            print("⚠️ No hay cámaras disponibles")
            return
        self.cap.release()
        current_pos = self.available_cams.index(self.current_cam_index)
        self.current_cam_index = self.available_cams[(current_pos + 1) % len(self.available_cams)]
        self._init_camera(self.current_cam_index)
        print(f"📷 Cámara cambiada a: {self.current_cam_index}")

    # -------------------------------------------------
    # Detector y filtros
    # -------------------------------------------------
    def _init_detector(self):
        self.detector = FaceDetector(
            static_image_mode=False,
            max_faces=1,
            refine_landmarks=True
        )

    def _init_filters(self):
        self.eye_filter = EyeFilter(
            asset_paths=["assets/lentes.png"],
            scale_factor=1.6,
            y_offset_ratio=0.0
        )
        self.mustache = MustacheFilter("assets/moustache.png", target_width_px=100, y_offset_px=15)
        self.nose = NoseDotFilter(radius=6, color=(0, 0, 255))
        self.crown = CrownFilter("assets/crown.png")
        self.hat = HatFilter("assets/hat.png")
        self.shield = Shield3DFilter("assets/Hybridge.obj", scale=0.9)
        self.gas_mask = GasMaskFilter()
        self.hockey_mask = HockeyMaskFilter()

    def _init_pipeline(self):
        self.pipeline = FilterPipeline()
        self.pipeline.add("glasses", self.eye_filter, enabled=False)
        self.pipeline.add("mustache", self.mustache, enabled=False)
        self.pipeline.add("nose", self.nose, enabled=False)
        self.pipeline.add("crown", self.crown, enabled=False)
        self.pipeline.add("hat", self.hat, enabled=False)
        self.pipeline.add("shield3d", self.shield, enabled=False)
        self.pipeline.add("gasmask", self.gas_mask, enabled=False)
        self.pipeline.add("hockeymask", self.hockey_mask, enabled=False)

    # -------------------------------------------------
    # Loop principal
    # -------------------------------------------------
    def run(self):
        cv2.namedWindow("Filtros AR", cv2.WINDOW_NORMAL)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ No se pudo leer frame")
                break

            faces = self.detector.detect(frame)
            if faces:
                for landmarks in faces:
                    frame = self.pipeline.apply(frame, landmarks)

            self._draw_hud(frame)
            cv2.imshow("Filtros AR", frame)
            self._handle_keys()

            if cv2.getWindowProperty("Filtros AR", cv2.WND_PROP_VISIBLE) < 1:
                break

        self._release()

    # -------------------------------------------------
    # Controles
    # -------------------------------------------------
    def _handle_keys(self):
        key = cv2.waitKey(1) & 0xFF

        key_actions = {
            27: lambda: setattr(self, "running", False),  # ESC
            9: self.switch_camera,                        # TAB
            ord("g"): lambda: self._toggle("glasses"),
            ord("m"): lambda: self._toggle("mustache"),
            ord("n"): lambda: self._toggle("nose"),
            ord("c"): lambda: self._toggle("crown"),
            ord("h"): lambda: self._toggle("hat"),
            ord("s"): lambda: self._toggle("shield3d"),
            ord("o"): lambda: self._toggle("gasmask"),
            ord("l"): lambda: self._toggle("hockeymask"),
            ord(" "): self.eye_filter.next_asset,
            ord("a"): lambda: self.eye_filter.add_offset_x(-5),
            ord("d"): lambda: self.eye_filter.add_offset_x(5)
        }

        action = key_actions.get(key)
        if action:
            action()


    def _toggle(self, name):
        for item in self.pipeline.items:
            if item.name == name:
                item.enabled = not item.enabled
                break

    # -------------------------------------------------
    # HUD
    # -------------------------------------------------
    def _draw_hud(self, frame):
        y = 20
        for item in self.pipeline.items:
            text = f"[{item.name}] {'ON' if item.enabled else 'OFF'}"
            color = (0, 255, 0) if item.enabled else (0, 0, 255)
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 18

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------
    def _release(self):
        self.cap.release()
        cv2.destroyAllWindows()
