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

# ==========================================
# CONFIGURACIÓN DE CÁMARA 
# ==========================================
URL_CAMARA = "http://192.168.1.151:8080/video"
cap = cv2.VideoCapture(URL_CAMARA)
if not cap.isOpened():
    print(f"No se pudo conectar a la IP. Iniciando webcam local...")
    cap = cv2.VideoCapture(0)

print("="*60)
print("  Algebra 02 - Determinantes y Areas")
print("  Usa tu Pulgar, Indice y Medio para formar un triangulo.")
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

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        # Panel de informacion base
        cv2.rectangle(frame, (0, 0), (430, 140), (0, 0, 0), -1)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            
            # Extraemos las coordenadas de los 3 dedos: Pulgar (4), Indice (8), Medio (12)
            p1 = np.array([hand[4].x * w_img, hand[4].y * h_img])
            p2 = np.array([hand[8].x * w_img, hand[8].y * h_img])
            p3 = np.array([hand[12].x * w_img, hand[12].y * h_img])

            # Dibujamos el triangulo
            puntos_triangulo = np.array([p1, p2, p3], np.int32)
            cv2.polylines(frame, [puntos_triangulo], isClosed=True, color=(0, 255, 150), thickness=2)
            
            for pt in [p1, p2, p3]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 150, 255), -1)

            # ----------------------------------------------------
            # APLICACION DE ALGEBRA LINEAL: DETERMINANTE
            # ----------------------------------------------------
            # 1. Creamos los vectores v1 y v2 a partir del punto base p1 (Pulgar)
            v1 = p2 - p1
            v2 = p3 - p1

            # 2. Construimos la matriz 2x2 con los vectores como columnas
            matriz = np.column_stack((v1, v2))

            # 3. Calculamos el determinante
            determinante = np.linalg.det(matriz)

            # 4. El area del triangulo es la mitad del valor absoluto del determinante
            area = abs(determinante) / 2.0

            # ----------------------------------------------------
            # Feedback visual
            # ----------------------------------------------------
            # Cambiamos el color interior si el area supera un umbral (ej. 5000 px)
            color_relleno = (0, 255, 150) if area < 5000 else (0, 100, 255)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [puntos_triangulo], color_relleno)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            cv2.putText(frame, f"Matriz V = [[{v1[0]:.1f}, {v2[0]:.1f}],", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"            [{v1[1]:.1f}, {v2[1]:.1f}]]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Determinante: {determinante:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(frame, f"Area Triangulo: {area:.1f} px", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
            cv2.putText(frame, f"URL IP: Activa", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        else:
            cv2.putText(frame, "Muestra Pulgar, Indice y Medio", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(frame, f"Camara IP: Intentando leer...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        cv2.imshow("Algebra 02 - Determinantes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()