"""
GESTOS CON ÁLGEBRA LINEAL - PASO 1
Tema: Espacios Vectoriales

CONCEPTO:
    La mano tiene 21 puntos clave (landmarks).
    Cada punto tiene coordenadas (x, y, z).
    Vector resultante: v ∈ R⁶³
"""

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os

# ── Descargar modelo si no existe ──────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de mano... (solo la primera vez)")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Modelo descargado ✓")

# ── Configurar MediaPipe Tasks API (0.10.x) ────────────────
BaseOptions      = mp.tasks.BaseOptions
HandLandmarker   = mp.tasks.vision.HandLandmarker
HandOptions      = mp.tasks.vision.HandLandmarkerOptions
RunningMode      = mp.tasks.vision.RunningMode

options = HandOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

# Conexiones entre landmarks para dibujar el esqueleto
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),          # pulgar
    (0,5),(5,6),(6,7),(7,8),          # índice
    (0,9),(9,10),(10,11),(11,12),     # medio
    (0,13),(13,14),(14,15),(15,16),   # anular
    (0,17),(17,18),(18,19),(19,20),   # meñique
    (5,9),(9,13),(13,17)              # palma
]

# ── Cámara Android ─────────────────────────────────────────
cap = cv2.VideoCapture("http://192.168.1.104:8080/video")

print("=" * 60)
print("  PASO 1 — Representación vectorial de la mano")
print("  Muestra tu mano | Presiona Q para salir")
print("=" * 60)

frame_count = 0

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo conectar. Verifica que IP Webcam esté activo.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Procesar con MediaPipe
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result   = detector.detect(mp_img)

        # ── ÁLGEBRA LINEAL ─────────────────────────────────
        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                # Construir vector en R⁶³
                vector_mano = []
                puntos = []
                for lm in hand:
                    vector_mano.extend([lm.x, lm.y, lm.z])
                    puntos.append((int(lm.x * w), int(lm.y * h)))

                v     = np.array(vector_mano)   # shape (63,)
                norma = np.linalg.norm(v)
                frame_count += 1

                # Dibujar conexiones (esqueleto)
                for a, b in CONEXIONES:
                    cv2.line(frame, puntos[a], puntos[b], (0, 200, 255), 2)

                # Dibujar puntos
                for i, (px, py) in enumerate(puntos):
                    color = (0, 255, 150) if i == 0 else (255, 255, 255)
                    cv2.circle(frame, (px, py), 5, color, -1)

                # Panel informativo
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (430, 165), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, "ESPACIO VECTORIAL R^63",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 2)
                cv2.putText(frame, "Landmarks: 21 puntos x (x,y,z)",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                cv2.putText(frame, f"Dimension del vector: {len(v)}  (= 21 x 3)",
                            (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                cv2.putText(frame, f"v[0:6] = [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f},",
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)
                cv2.putText(frame, f"          {v[3]:.2f}, {v[4]:.2f}, {v[5]:.2f}, ...]",
                            (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)
                cv2.putText(frame, f"||v|| (norma) = {norma:.4f}",
                            (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 1)

                if frame_count % 30 == 0:
                    print(f"[R⁶³] norma={norma:.4f} | v[:6]={np.round(v[:6], 3)}")

        else:
            cv2.putText(frame, "Muestra tu mano...",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        cv2.imshow("GestOS - Paso 1: Vector de la Mano", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nPaso 1 completado ✓")