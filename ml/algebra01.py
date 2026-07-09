import cv2
import numpy as np
import mediapipe as mp
import os
import math

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

# Conexiones de MediaPipe
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
    (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

cap = cv2.VideoCapture("http://10.250.180.216:8080/video")
if not cap.isOpened():
    print(f"No se pudo conectar a la IP. Iniciando webcam local...")
    cap = cv2.VideoCapture(0)

#==========================================
# ESTADO DEL CUBO 3D
# ==========================================
p_cubo = np.array([400.0, 300.0]) # Posición X, Y en pantalla
escala_cubo = 80.0                # Tamaño del cubo
ang_x, ang_y, ang_z = 0.5, 0.5, 0.0 # Ángulos de rotación iniciales

# Variables de interacción
estado_interaccion = "LIBRE"
h_prev = None

# Vértices de un cubo perfecto centrado en el origen (de -1 a 1)
VERTICES_BASE = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
])

# Las 12 aristas que conectan los vértices del cubo
ARISTAS = [
    (0,1), (1,2), (2,3), (3,0), # Cara trasera
    (4,5), (5,6), (6,7), (7,4), # Cara frontal
    (0,4), (1,5), (2,6), (3,7)  # Conexiones laterales
]

def obtener_matriz_rotacion(ax, ay, az):
    """Calcula la matriz de rotación 3D combinada R = Rz * Ry * Rx"""
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)],
                   [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def dibujar_cubo_3d(frame, centro, escala, ax, ay, az, estado):
    R = obtener_matriz_rotacion(ax, ay, az)
    
    # 1. Rotar los vértices base
    verts_rotados = np.dot(VERTICES_BASE, R.T)
    
    # 2. Calcular profundidad y proyección 2D
    puntos_2d = []
    para_dibujar = []
    
    for v in verts_rotados:
        # Proyección ortográfica: x' = x*escala + centro_x
        px = int(v[0] * escala + centro[0])
        py = int(v[1] * escala + centro[1])
        puntos_2d.append((px, py))
        
    # 3. Ordenar las líneas (Z-Sorting) para que el frente tape al fondo
    for a, b in ARISTAS:
        profundidad_z = (verts_rotados[a][2] + verts_rotados[b][2]) / 2.0
        para_dibujar.append((profundidad_z, a, b))
        
    # Ordenar de menor Z (más lejos) a mayor Z (más cerca)
    para_dibujar.sort(key=lambda item: item[0])
    
    color_frente = (0, 255, 200) if estado == "ROTANDO" else (0, 160, 255) if estado == "MOVIENDO" else (255, 100, 30)

    # 4. Dibujar aristas
    for prof, a, b in para_dibujar:
        pt1 = puntos_2d[a]
        pt2 = puntos_2d[b]
        
        # Efecto visual: Las líneas del fondo son más oscuras y finas
        if prof < 0: 
            color = (max(0, color_frente[0]-100), max(0, color_frente[1]-100), max(0, color_frente[2]-100))
            grosor = 2
        else:
            color = color_frente
            grosor = 4
            
        cv2.line(frame, pt1, pt2, color, grosor)
        cv2.circle(frame, pt1, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt2, 4, (255, 255, 255), -1)

def dibujar_panel(frame, estado, p_cubo, h_centroide):
    overlay = frame.copy()
    # Hicimos el panel un poco más alto (200px) para que quepa el nuevo texto
    cv2.rectangle(overlay, (0, 0), (520, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Colores por estado (agregamos color rosa para escalar)
    if estado == "ROTANDO": c_texto = (0, 255, 200)
    elif estado == "MOVIENDO": c_texto = (0, 160, 255)
    elif estado == "ESCALANDO": c_texto = (255, 100, 255) 
    else: c_texto = (200, 200, 200)

    cv2.putText(frame, f"ESTADO: {estado}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_texto, 2)
    cv2.putText(frame, f"Rotacion (Rad): [{ang_x:.2f}, {ang_y:.2f}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)

    if h_centroide is not None:
        cv2.putText(frame, f"Centroide Mano: [{h_centroide[0]:.0f}, {h_centroide[1]:.0f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)
    
    cv2.putText(frame, "Pinch INDICE CERCA del cubo = Mover", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.putText(frame, "Pinch INDICE LEJOS del cubo = Rotar", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.putText(frame, "Pinch MEDIO (Pulgar+Medio) + Mover Arriba/Abajo = Escalar", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,150,255), 1)

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h_img, w_img = frame.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        h_centroide = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            puntos = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in hand]

            for a, b in CONEXIONES:
                cv2.line(frame, puntos[a], puntos[b], (0,200,255), 1)
            for px, py in puntos:
                cv2.circle(frame, (px,py), 3, (255,255,255), -1)

            pts = np.array(puntos, dtype=float)
            h_centroide = np.mean(pts, axis=0)

            pulgar = np.array(puntos[4], dtype=float)
            indice = np.array(puntos[8], dtype=float)
            medio  = np.array(puntos[12], dtype=float)
            dist_pinch = np.linalg.norm(pulgar - indice)
            distancia_al_cubo = np.linalg.norm(p_cubo - h_centroide)
            dist_pinch_medio = np.linalg.norm(pulgar - medio)

            # --- NUEVO MODO: ESCALAR ---
            if dist_pinch_medio < 50:
                estado_interaccion = "ESCALANDO"
                
                if h_prev is not None:
                    delta = h_centroide - h_prev
                    # Eje Y en pantalla: mover arriba (negativo) agranda, abajo achica
                    escala_cubo -= delta[1] * 0.5 
                    # Limitamos el tamaño para que no desaparezca ni se vuelva infinito
                    escala_cubo = max(20.0, min(escala_cubo, 300.0))
                    
                h_prev = h_centroide.copy()

            # --- MODOS ACTUALES: MOVER Y ROTAR ---
            elif dist_pinch < 50:
                # Si venimos de LIBRE o de ESCALAR, decidimos el nuevo modo por distancia
                if estado_interaccion == "LIBRE" or estado_interaccion == "ESCALANDO":
                    if distancia_al_cubo < escala_cubo * 1.5:
                        estado_interaccion = "MOVIENDO"
                    else:
                        estado_interaccion = "ROTANDO"

                if h_prev is not None:
                    delta = h_centroide - h_prev
                    
                    if estado_interaccion == "MOVIENDO":
                        p_cubo += delta
                    elif estado_interaccion == "ROTANDO":
                        ang_y += delta[0] * 0.02
                        ang_x -= delta[1] * 0.02

                h_prev = h_centroide.copy()
                
            # --- MANO ABIERTA ---
            else:
                estado_interaccion = "LIBRE"
                h_prev = None
        else:
            estado_interaccion = "LIBRE"
            h_prev = None

        dibujar_cubo_3d(frame, p_cubo, escala_cubo, ang_x, ang_y, ang_z, estado_interaccion)
        dibujar_panel(frame, estado_interaccion, p_cubo, h_centroide)

        cv2.imshow("GestOS - Cubo 3D", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()