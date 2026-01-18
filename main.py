import cv2
from face_detector import FaceDetector
from filters_main import MustacheFilter

# --------------------------
# Detectar cámaras disponibles
# --------------------------
def listar_camaras(max_index=10):
    camaras_disponibles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            camaras_disponibles.append(i)
            cap.release()
    return camaras_disponibles

camaras = listar_camaras()
if not camaras:
    print("No se detectaron cámaras. Conecta Camo o tu webcam.")
    exit()

# Elegir cámara automáticamente si solo hay una
if len(camaras) == 1:
    cam_index = camaras[0]
    print(f"Usando cámara {cam_index}")
else:
    cam_index = int(input(f"Elige cámara {camaras}: "))

# --------------------------
# Inicializar cámara
# --------------------------
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --------------------------
# Inicializar FaceDetector y filtro
# --------------------------
detector = FaceDetector(static_image_mode=False, max_faces=1, refine_landmarks=True)
mustache = MustacheFilter("filters/img/moustache.png", target_width_px=100, y_offset_px=15)

# --------------------------
# Loop principal
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la cámara. Revisa Camo Studio o tu webcam.")
        break

    # Detectar rostros y landmarks
    faces = detector.detect(frame)
    if faces:
        for face_landmarks in faces:
            frame = mustache.apply(frame, face_landmarks)

    # Mostrar resultado
    cv2.imshow("Mostacho AR", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --------------------------
# Liberar recursos
# --------------------------
cap.release()
cv2.destroyAllWindows()
