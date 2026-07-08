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

# Cambiamos a 2 manos para controlar las dos rectas por separado
options = HandOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2 
)

# ---------- Configuración de Cámara IP ----------
URL_CAMARA = "http://192.168.1.151:8080/video"
cap = cv2.VideoCapture(URL_CAMARA)

if not cap.isOpened():
    print(f"No se pudo conectar a {URL_CAMARA}. Iniciando webcam local...")
    cap = cv2.VideoCapture(0)

print("="*60)
print("  Algebra 03 - Sistemas de Ecuaciones Lineales (Multimano)")
print("  Usa AMBAS manos para jugar con las ecuaciones.")
print("  Mano 1 (Pulgar -> Indice) controla la Recta 1")
print("  Mano 2 (Pulgar -> Indice) controla la Recta 2")
print("="*60)

def dibujar_linea_infinita(frame, p1, p2, color):
    """Dibuja una linea que atraviesa toda la pantalla basada en dos puntos"""
    # Evitar division por cero (linea perfectamente vertical)
    if p1[0] == p2[0]:
        cv2.line(frame, (int(p1[0]), 0), (int(p1[0]), frame.shape[0]), color, 2)
        return
    
    pendiente = (p2[1] - p1[1]) / (p2[0] - p1[0])
    interseccion_y = p1[1] - pendiente * p1[0]
    
    y_ini = int(interseccion_y)
    y_fin = int(pendiente * frame.shape[1] + interseccion_y)
    
    cv2.line(frame, (0, y_ini), (frame.shape[1], y_fin), color, 2)

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h_img, w_img = frame.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        # Panel de informacion
        cv2.rectangle(frame, (0, 0), (520, 160), (0, 0, 0), -1)
        cv2.putText(frame, "SISTEMA DE ECUACIONES 2x2 (2 MANOS)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Verificamos que al menos 2 manos esten en pantalla
        if result.hand_landmarks and len(result.hand_landmarks) >= 2:
            mano1 = result.hand_landmarks[0]
            mano2 = result.hand_landmarks[1]
            
            # --- MANO 1 (Recta 1) ---
            m1_pulgar = np.array([mano1[4].x * w_img, mano1[4].y * h_img])
            m1_indice = np.array([mano1[8].x * w_img, mano1[8].y * h_img])
            
            # --- MANO 2 (Recta 2) ---
            m2_pulgar = np.array([mano2[4].x * w_img, mano2[4].y * h_img])
            m2_indice = np.array([mano2[8].x * w_img, mano2[8].y * h_img])

            # Dibujar marcadores en los dedos
            cv2.circle(frame, (int(m1_pulgar[0]), int(m1_pulgar[1])), 8, (0,255,150), -1)
            cv2.circle(frame, (int(m1_indice[0]), int(m1_indice[1])), 8, (0,255,150), -1)
            cv2.circle(frame, (int(m2_pulgar[0]), int(m2_pulgar[1])), 8, (255,100,100), -1)
            cv2.circle(frame, (int(m2_indice[0]), int(m2_indice[1])), 8, (255,100,100), -1)

            # Dibujar lineas infinitas
            dibujar_linea_infinita(frame, m1_pulgar, m1_indice, (0, 200, 100)) # Verde
            dibujar_linea_infinita(frame, m2_pulgar, m2_indice, (200, 100, 50))  # Azul/Violeta

            # --- ALGEBRA LINEAL: SISTEMAS DE ECUACIONES ---
            # Recta 1: A1*x + B1*y = C1 (Mano 1)
            A1 = m1_indice[1] - m1_pulgar[1]
            B1 = m1_pulgar[0] - m1_indice[0]
            C1 = A1 * m1_pulgar[0] + B1 * m1_pulgar[1]

            # Recta 2: A2*x + B2*y = C2 (Mano 2)
            A2 = m2_indice[1] - m2_pulgar[1]
            B2 = m2_pulgar[0] - m2_indice[0]
            C2 = A2 * m2_pulgar[0] + B2 * m2_pulgar[1]

            M = np.array([[A1, B1], [A2, B2]])
            K = np.array([C1, C2])

            det = np.linalg.det(M)

            cv2.putText(frame, f"L1 (Mano 1): {A1:.1f}x + {B1:.1f}y = {C1:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)
            cv2.putText(frame, f"L2 (Mano 2): {A2:.1f}x + {B2:.1f}y = {C2:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 100), 1)

            if abs(det) > 1e-5:
                # Sistema tiene solucion unica
                solucion = np.linalg.solve(M, K)
                sol_x, sol_y = int(solucion[0]), int(solucion[1])

                cv2.putText(frame, f"Solucion Matricial [x, y]: [{sol_x}, {sol_y}]", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Mueve tus manos para alterar el sistema", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                if 0 <= sol_x <= w_img and 0 <= sol_y <= h_img:
                    cv2.circle(frame, (sol_x, sol_y), 10, (0, 255, 255), -1)
                    cv2.circle(frame, (sol_x, sol_y), 15, (255, 255, 255), 2)
                    
                    # Pequeño efecto visual extra de "rayos" si se cruzan en pantalla
                    cv2.line(frame, (sol_x, sol_y), (int(m1_indice[0]), int(m1_indice[1])), (255, 255, 255), 1)
                    cv2.line(frame, (sol_x, sol_y), (int(m2_indice[0]), int(m2_indice[1])), (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Lineas Paralelas (Sin Solucion)", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "Por favor, muestra AMBAS manos en camara", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            # Mostrar cuantas manos detecta
            manos_detectadas = len(result.hand_landmarks) if result.hand_landmarks else 0
            cv2.putText(frame, f"Manos detectadas: {manos_detectadas}/2", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

        cv2.imshow("Algebra 03 - Sistemas Lineales", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()