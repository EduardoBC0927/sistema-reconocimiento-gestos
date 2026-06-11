import cv2
import numpy as np
import mediapipe as mp
import os

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Falta hand_landmarker.task en la carpeta del proyecto.")
    exit()

BaseOptions    = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandOptions    = mp.tasks.vision.HandLandmarkerOptions
RunningMode    = mp.tasks.vision.RunningMode

options = HandOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

cap = cv2.VideoCapture(1)

# Estado de la esfera - vector posicion en R2
p_esfera = np.array([400, 300], dtype=float)
r_esfera = 70
agarrada = False
h_prev   = None
DIST_AGARRE = 90

def dibujar_esfera(frame, centro, radio, agarrada):
    cx, cy = int(centro[0]), int(centro[1])
    color_nucleo = (255, 140, 0) if agarrada else (255, 100, 30)
    color_brillo = (180, 80,  0) if agarrada else (200, 60,  10)
    for i in range(6, 0, -1):
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radio + i * 10, color_brillo, -1)
        cv2.addWeighted(overlay, 0.06 * i, frame, 1 - 0.06 * i, 0, frame)
    cv2.circle(frame, (cx, cy), radio, color_nucleo, -1)
    ox = int(-radio * 0.3)
    oy = int(-radio * 0.35)
    rv = max(int(radio * 0.25), 5)
    cv2.circle(frame, (cx + ox, cy + oy), rv, (255, 220, 180), -1)
    cv2.circle(frame, (cx + ox, cy + oy), rv, (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), radio, (255, 200, 100), 2)

def dibujar_panel(frame, p_esfera, h_centroide, distancia, radio, modo):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 215), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    color = (0,200,255) if modo=="AGARRANDO" else (0,255,150) if modo=="ESCALANDO" else (200,200,200)
    cv2.putText(frame, f"MODO: {modo}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"p_esfera = [{p_esfera[0]:.0f}, {p_esfera[1]:.0f}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)
    if h_centroide is not None:
        cv2.putText(frame, f"h_mano   = [{h_centroide[0]:.0f}, {h_centroide[1]:.0f}]",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)
        cv2.putText(frame, f"||p - h|| = {distancia:.1f}px",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,80), 1)
    cv2.putText(frame, "Traslacion: p' = p + d  (d = desplazamiento)",
                (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, f"Escala: S = [[k,0],[0,k]]   r' = {radio}px",
                (10,155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, "Acercar mano=agarrar | Pinch=escalar | Q=salir",
                (10,195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)

print("="*60)
print("  GestOS - Esfera Virtual con Algebra Lineal")
print("  Acerca la mano a la esfera naranja para agarrarla")
print("  Pinch (pulgar+indice juntos) para escalar")
print("  Presiona Q para salir")
print("="*60)

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Sin senal de camara.")
            break

        frame = cv2.flip(frame, 1)
        h_img, w_img = frame.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        modo        = "LIBRE"
        h_centroide = None
        distancia   = 9999

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                puntos = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in hand]

                for a, b in CONEXIONES:
                    cv2.line(frame, puntos[a], puntos[b], (0,200,255), 2)
                for i, (px, py) in enumerate(puntos):
                    cv2.circle(frame, (px,py), 5,
                               (0,255,150) if i==0 else (255,255,255), -1)

                # Centroide: promedio de vectores posicion
                pts = np.array(puntos, dtype=float)
                h_centroide = np.mean(pts, axis=0)

                # Distancia vectorial ||p - h||
                distancia = np.linalg.norm(p_esfera - h_centroide)

                pulgar = np.array(puntos[4], dtype=float)
                indice = np.array(puntos[8], dtype=float)
                dist_pinch = np.linalg.norm(pulgar - indice)

                if dist_pinch < 50:
                    # Escala con matriz S = k * I
                    modo = "ESCALANDO"
                    k = np.clip(dist_pinch / 50, 0.4, 2.0)
                    S = k * np.eye(2)
                    r_esfera = int(np.clip(S[0,0] * 70, 30, 150))

                elif distancia < DIST_AGARRE:
                    # Traslacion: p' = p + d
                    modo     = "AGARRANDO"
                    agarrada = True
                    if h_prev is not None:
                        d        = h_centroide - h_prev
                        p_esfera = p_esfera + d
                        p_esfera[0] = np.clip(p_esfera[0], 0, w_img)
                        p_esfera[1] = np.clip(p_esfera[1], 0, h_img)
                    h_prev = h_centroide.copy()
                else:
                    agarrada = False
                    h_prev   = None

                if distancia < DIST_AGARRE * 2:
                    overlay = frame.copy()
                    cv2.line(overlay,
                             (int(h_centroide[0]), int(h_centroide[1])),
                             (int(p_esfera[0]),    int(p_esfera[1])),
                             (0,255,200), 1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        else:
            agarrada = False
            h_prev   = None

        dibujar_esfera(frame, p_esfera, r_esfera, agarrada)
        dibujar_panel(frame, p_esfera, h_centroide, distancia, r_esfera, modo)

        cv2.imshow("GestOS - Esfera Virtual", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
