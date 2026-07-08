import cv2
import numpy as np
import mediapipe as mp
import os

MODEL_PATH = "ml/hand_landmarker.task"
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

cap = cv2.VideoCapture(0)

VERTICES = np.array([
    [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],   # cara trasera  (z=-1)
    [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1],   # cara delantera (z=+1)
], dtype=float)

ARISTAS = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

CARAS = [
    (0,1,2,3), (4,5,6,7),
    (0,1,5,4), (2,3,7,6),
    (1,2,6,5), (0,3,7,4)
]

LUZ = np.array([0.35, -0.5, -0.8])
LUZ = LUZ / np.linalg.norm(LUZ)

# ---------- Algebra: rotaciones ----------
def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rodrigues(eje, angulo):
    """Formula de Rodrigues: R = I + sin(t)*K + (1-cos(t))*K^2
    (matriz de rotacion a partir de un eje unitario y un angulo)"""
    if angulo < 1e-6:
        return np.eye(3)
    k = eje / (np.linalg.norm(eje) + 1e-9)
    K = np.array([[0,-k[2],k[1]],
                  [k[2],0,-k[0]],
                  [-k[1],k[0],0]])
    return np.eye(3) + np.sin(angulo)*K + (1-np.cos(angulo))*(K @ K)

def matriz_a_eje_angulo(R):
    """Inversa de Rodrigues: obtiene (eje, angulo) a partir de una matriz de rotacion,
    usando la parte antisimetrica R - R^T y la traza de R."""
    cos_a = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angulo = np.arccos(cos_a)
    if angulo < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    eje = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    eje = eje / (2*np.sin(angulo) + 1e-9)
    return eje, angulo

def ortonormalizar(R):
    """Corrige la deriva numerica acumulada para que R siga siendo una rotacion valida."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt

def marco_mano(p3d):
    """
    Construye una base ortonormal (3 vectores perpendiculares entre si) que
    describe la orientacion de la mano en el espacio 3D real (hand_world_landmarks):
      adelante = muneca(0) -> nudillo medio(9)
      lateral  = nudillo indice(5) -> nudillo menique(17)
      normal   = adelante x lateral   (perpendicular a la palma)
    lateral se reortogonaliza (Gram-Schmidt) para que quede una base limpia.
    """
    adelante = p3d[9]  - p3d[0]
    lateral  = p3d[17] - p3d[5]

    adelante = adelante / (np.linalg.norm(adelante) + 1e-9)
    normal   = np.cross(adelante, lateral)
    normal   = normal / (np.linalg.norm(normal) + 1e-9)
    lateral  = np.cross(normal, adelante)
    lateral  = lateral / (np.linalg.norm(lateral) + 1e-9)

    return np.column_stack([adelante, lateral, normal])

# ---------- Dibujo ----------
def dibujar_cubo(frame, centro, escala, R, agarrada):
    v_local = VERTICES * escala
    v_rot   = v_local @ R.T          # v' = R @ v para cada vertice

    base = (255,140,0) if agarrada else (255,100,30)

    info_caras = []
    for cara in CARAS:
        idx = list(cara)
        p_loc = v_local[idx]
        p_rot = v_rot[idx]
        normal = np.cross(p_rot[1]-p_rot[0], p_rot[2]-p_rot[0])
        n = np.linalg.norm(normal)
        normal = normal/n if n > 1e-8 else normal
        centro_local = np.mean(p_loc, axis=0)
        if np.dot(normal, centro_local) < 0:
            normal = -normal                     # forzar normal "hacia afuera"
        z_prom = np.mean(p_rot[:,2])
        info_caras.append((z_prom, idx, normal))

    info_caras.sort(key=lambda t: -t[0])         # pintor: lejos -> cerca

    for z_prom, idx, normal in info_caras:
        if normal[2] > 1e-6:
            continue                              # cara oculta (back-face culling)
        intensidad = np.clip(np.dot(normal, -LUZ), 0.25, 1.0)
        pts2d = np.array([[centro[0]+v_rot[i,0], centro[1]+v_rot[i,1]] for i in idx],
                          dtype=np.int32)
        col = tuple(int(c*intensidad) for c in base)
        cv2.fillConvexPoly(frame, pts2d, col, lineType=cv2.LINE_AA)

    for a, b in ARISTAS:
        pa = (int(centro[0]+v_rot[a,0]), int(centro[1]+v_rot[a,1]))
        pb = (int(centro[0]+v_rot[b,0]), int(centro[1]+v_rot[b,1]))
        cv2.line(frame, pa, pb, (255,225,160), 2, cv2.LINE_AA)

    for i in range(8):
        p = (int(centro[0]+v_rot[i,0]), int(centro[1]+v_rot[i,1]))
        cv2.circle(frame, p, 3, (255,255,255), -1)

def dibujar_panel(frame, p_cubo, h_centroide, distancia, tam_cubo, modo, R_cubo):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 245), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    color = (0,200,255) if modo=="AGARRANDO" else (0,255,150) if modo=="ESCALANDO" else (200,200,200)
    cv2.putText(frame, f"MODO: {modo}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"p_cubo = [{p_cubo[0]:.0f}, {p_cubo[1]:.0f}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)
    if h_centroide is not None:
        cv2.putText(frame, f"h_mano   = [{h_centroide[0]:.0f}, {h_centroide[1]:.0f}]",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,255), 1)
        cv2.putText(frame, f"||p - h|| = {distancia:.1f}px",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,80), 1)

    ang_tot = np.degrees(np.arccos(np.clip((np.trace(R_cubo)-1)/2, -1, 1)))
    cv2.putText(frame, f"angulo(R_cubo) = {ang_tot:5.1f} grados   (t=arccos((tr(R)-1)/2))",
                (10,123), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,220,180), 1)
    cv2.putText(frame, "Traslacion: p' = p + d            (d = desplazamiento)",
                (10,148), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, f"Escala: S = k * I3                 lado' = {int(tam_cubo*2)}px",
                (10,171), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, "Rotacion: R' = Rodrigues(eje,t) . R   (eje = v_prev x v_act)",
                (10,194), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(frame, "Agarrar=mover+girar | Pinch=escalar | Q=salir",
                (10,222), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)

print("="*60)
print("  GestOS - Cubo Virtual con Algebra Lineal")
print("  Acerca la mano al cubo para agarrarlo y moverlo")
print("  Gira la muneca mientras lo tienes agarrado para rotarlo")
print("  Pinch (pulgar+indice juntos) para escalar")
print("  Presiona Q para salir")
print("="*60)

cap = cv2.VideoCapture(0)

# ---------- Estado del cubo ----------
p_cubo   = np.array([400.0, 300.0])                    # vector posicion en R2 (pantalla)
tam_cubo = 90.0                                          # semi-arista en pixeles
R_cubo   = Ry(np.radians(-25)) @ Rx(np.radians(20))      # orientacion inicial (vista 3/4)

agarrada = False
h_prev   = None       # centroide de la mano en el frame anterior (para trasladar)
F_prev   = None       # marco de orientacion de la mano en el frame anterior (para rotar)

DIST_AGARRE      = 90
SENSIBILIDAD_ROT = 1.3     # >1 = giros mas amplios, <1 = giros mas suaves
UMBRAL_JITTER    = np.radians(0.3)   # ignora micro-vibraciones de deteccion

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
            for hand, hand_mundo in zip(result.hand_landmarks, result.hand_world_landmarks):
                puntos   = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in hand]
                puntos3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_mundo], dtype=float)

                for a, b in CONEXIONES:
                    cv2.line(frame, puntos[a], puntos[b], (0,200,255), 2)
                for i, (px, py) in enumerate(puntos):
                    cv2.circle(frame, (px,py), 5,
                               (0,255,150) if i==0 else (255,255,255), -1)

                # Centroide: promedio de vectores posicion (para agarrar/trasladar)
                pts = np.array(puntos, dtype=float)
                h_centroide = np.mean(pts, axis=0)
                distancia = np.linalg.norm(p_cubo - h_centroide)

                pulgar = np.array(puntos[4], dtype=float)
                indice = np.array(puntos[8], dtype=float)
                dist_pinch = np.linalg.norm(pulgar - indice)

                # Marco de orientacion 3D de la mano (para rotar)
                F_actual = marco_mano(puntos3d)

                if dist_pinch < 50:
                    # Escala con matriz S = k * I
                    modo = "ESCALANDO"
                    k = np.clip(dist_pinch / 50, 0.4, 2.0)
                    S = k * np.eye(3)
                    tam_cubo = np.clip(S[0,0] * 90, 30, 200)
                    h_prev, F_prev = None, None

                elif distancia < DIST_AGARRE:
                    # Traslacion: p' = p + d
                    modo     = "AGARRANDO"
                    agarrada = True
                    if h_prev is not None:
                        d = h_centroide - h_prev
                        p_cubo = p_cubo + d
                        p_cubo[0] = np.clip(p_cubo[0], 0, w_img)
                        p_cubo[1] = np.clip(p_cubo[1], 0, h_img)

                    # Rotacion: delta de orientacion entre frames -> eje/angulo -> Rodrigues
                    if F_prev is not None:
                        R_delta_medida = F_actual @ F_prev.T
                        eje, ang = matriz_a_eje_angulo(R_delta_medida)
                        if ang > UMBRAL_JITTER:
                            R_delta = rodrigues(eje, ang * SENSIBILIDAD_ROT)
                            R_cubo = ortonormalizar(R_delta @ R_cubo)

                    h_prev = h_centroide.copy()
                    F_prev = F_actual.copy()
                else:
                    agarrada = False
                    h_prev, F_prev = None, None

                if distancia < DIST_AGARRE * 2:
                    overlay = frame.copy()
                    cv2.line(overlay,
                             (int(h_centroide[0]), int(h_centroide[1])),
                             (int(p_cubo[0]),    int(p_cubo[1])),
                             (0,255,200), 1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        else:
            agarrada = False
            h_prev, F_prev = None, None

        dibujar_cubo(frame, p_cubo, tam_cubo, R_cubo, agarrada)
        dibujar_panel(frame, p_cubo, h_centroide, distancia, tam_cubo, modo, R_cubo)

        cv2.imshow("GestOS - Cubo Virtual", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
