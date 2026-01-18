import cv2
from face_detector import FaceDetector
from filters.mustache_filter import MustacheFilter
from filters.eye_filter import EyeFilter
from filters.nose_dot_filter import NoseDotFilter
from pipeline.filter_pipeline import FilterPipeline
from filters.crown_filter import CrownFilter
from filters.hat_filter import HatFilter
from filters.shield_3d_filter import Shield3DFilter
from filters.Gas_Mask_Filter import GasMaskFilter
from filters.Hockey_Mask_filter import HockeyMaskFilter 

# --------------------------
# Detectar cámaras disponibles
# --------------------------
def listar_camaras(max_index=10):
    camaras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras.append(i)
            cap.release()
    return camaras


camaras = listar_camaras()
if not camaras:
    print("No se detectaron cámaras.")
    exit()

cam_index = camaras[0] if len(camaras) == 1 else int(input(f"Elige cámara {camaras}: "))


# --------------------------
# Inicializar cámara
# --------------------------
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --------------------------
# Inicializar detector
# --------------------------
detector = FaceDetector(
    static_image_mode=False,
    max_faces=1,
    refine_landmarks=True
)


# --------------------------
# Inicializar filtros
# --------------------------
mustache = MustacheFilter(
    "./assets/moustache.png",
    target_width_px=100,
    y_offset_px=15
)

eye_filter = EyeFilter(
    asset_paths=["assets/lentes.png"],
    scale_factor=1.6,
    y_offset_ratio=0.0
)

nose_dot = NoseDotFilter(
    radius=6,
    color=(0, 0, 255)
)

crown = CrownFilter("assets/crown.png")

hat = HatFilter("assets/hat.png")

shield = Shield3DFilter("assets/Hybridge.obj", scale=0.9)

gas_mask = GasMaskFilter()

hockey_mask = HockeyMaskFilter()

# --------------------------
# Pipeline de filtros
# --------------------------
pipeline = FilterPipeline()
pipeline.add("glasses", eye_filter, enabled=True)
pipeline.add("mustache", mustache, enabled=True)
pipeline.add("nose", nose_dot, enabled=False)
pipeline.add("crown", crown, enabled=False)
pipeline.add("hat", hat, enabled=False)
pipeline.add("shield3d", shield, enabled=False)
pipeline.add("gasmask", gas_mask, enabled=False)
pipeline.add("hockeymask", hockey_mask, enabled=False)

# --------------------------
# Loop principal
# --------------------------
cv2.namedWindow("Filtros AR")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    if faces:
        for landmarks in faces:
            frame = pipeline.apply(frame, landmarks)

    cv2.imshow("Filtros AR", frame)

    # Detectar cierre con la X
    if cv2.getWindowProperty("Filtros AR", cv2.WND_PROP_VISIBLE) < 1:        
        break

    key = cv2.waitKey(1) & 0xFF

    # --------------------------
    # Controles
    # --------------------------
    if key == 27:  # ESC
        break

    elif key == ord("g"):
        pipeline.set_enabled("glasses", not next(
            f.enabled for f in pipeline.items if f.name == "glasses"
        ))

    elif key == ord("m"):
        pipeline.set_enabled("mustache", not next(
            f.enabled for f in pipeline.items if f.name == "mustache"
        ))

    elif key == ord("n"):
        pipeline.set_enabled("nose", not next(
            f.enabled for f in pipeline.items if f.name == "nose"
        ))
    elif key == ord("c"):
        pipeline.set_enabled("crown", not next(
        f.enabled for f in pipeline.items if f.name == "crown"
        ))

    elif key == ord("h"):
        pipeline.set_enabled("hat", not next(
        f.enabled for f in pipeline.items if f.name == "hat"
        ))
    
    elif key == ord("s"):
        pipeline.set_enabled("shield3d", not next(
        f.enabled for f in pipeline.items if f.name == "shield3d"
        ))
    elif key == ord("o"):
        pipeline.set_enabled("gasmask", not next(
        f.enabled for f in pipeline.items if f.name == "gasmask"
        ))
    elif key == ord("l"):
        pipeline.set_enabled("hockeymask", not next(
        f.enabled for f in pipeline.items if f.name == "hockeymask"
        ))            
    elif key == ord(" "):
        eye_filter.next_asset()

    elif key == ord("a"):
        eye_filter.add_offset_x(-5)

    elif key == ord("d"):
        eye_filter.add_offset_x(5)


# --------------------------
# Liberar recursos
# --------------------------
cap.release()
cv2.destroyAllWindows()
