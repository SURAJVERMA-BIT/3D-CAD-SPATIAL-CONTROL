"""
S.T.A.R.K. Holographic 3D Manipulator
=====================================
Render a 3D model as a hologram over your webcam feed and manipulate it in
mid-air with your bare hands -- no mouse, no CAD app, just the laptop camera.

CONTROLS (one hand in front of the camera does everything)
    - Open hand, move it ................ rotate the hologram (it follows)
    - Flick and release ................. spins with momentum (inertia)
    - Thumb + index only (caliper) ...... scale: spread = bigger, pinch = smaller
    - Fist ............................. freeze / hold the orientation
    - No hand .......................... idle holographic auto-spin

KEYS
    n .... next model (cycles built-in meshes / loaded .obj)
    r .... reset orientation & scale
    g .... toggle hand skeleton overlay
    q .... quit

Load your own CAD model:  python stark_holo.py path\\to\\model.obj
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
import os
import sys
import threading
import time

# ===========================================================================
#  MediaPipe HandLandmarker (Tasks API, LIVE_STREAM async => no render stall)
# ===========================================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")
FRAME_W, FRAME_H = 960, 720

_result_lock = threading.Lock()
_latest_hands = []          # list[ list[(x, y)] ]  in pixel space


def _on_result(result, output_image, timestamp_ms):
    global _latest_hands
    hands = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            hands.append([(int(lm.x * FRAME_W), int(lm.y * FRAME_H)) for lm in hand])
    with _result_lock:
        _latest_hands = hands


_hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    result_callback=_on_result,
)
landmarker = vision.HandLandmarker.create_from_options(_hand_options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Hologram palette (BGR)
CYAN   = (255, 230, 90)
CYAN_D = (170, 130, 40)
AMBER  = (40, 200, 255)
RED    = (70, 70, 255)
WHITE  = (255, 255, 255)


# ===========================================================================
#  Mesh library  -> returns (vertices Nx3 float, edges list[(i, j)])
#  All meshes are recentered and normalized to a unit radius.
# ===========================================================================
def _normalize(V):
    V = np.asarray(V, dtype=float)
    V -= V.mean(axis=0)
    r = np.max(np.linalg.norm(V, axis=1))
    return V / (r if r > 1e-9 else 1.0)


def make_uv_sphere(stacks=16, slices=24):
    V, idx = [], {}
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices):
            theta = 2 * math.pi * j / slices
            idx[(i, j)] = len(V)
            V.append([math.sin(phi) * math.cos(theta),
                      math.cos(phi),
                      math.sin(phi) * math.sin(theta)])
    E = set()
    for i in range(stacks + 1):
        for j in range(slices):
            if 0 < i < stacks:
                E.add(tuple(sorted((idx[(i, j)], idx[(i, (j + 1) % slices)]))))
            if i < stacks:
                E.add(tuple(sorted((idx[(i, j)], idx[(i + 1, j)]))))
    return _normalize(V), list(E)


def make_torus(nu=28, nv=16, R=1.0, r=0.42):
    V, idx = [], {}
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            idx[(i, j)] = len(V)
            V.append([(R + r * math.cos(v)) * math.cos(u),
                      r * math.sin(v),
                      (R + r * math.cos(v)) * math.sin(u)])
    E = set()
    for i in range(nu):
        for j in range(nv):
            E.add(tuple(sorted((idx[(i, j)], idx[((i + 1) % nu, j)]))))
            E.add(tuple(sorted((idx[(i, j)], idx[(i, (j + 1) % nv)]))))
    return _normalize(V), list(E)


def make_cube():
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    E = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.count_nonzero(np.abs(V[i] - V[j]) > 1e-9) == 1]
    return _normalize(V), E


def make_icosahedron():
    t = (1 + math.sqrt(5)) / 2
    V = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    E = set()
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            E.add(tuple(sorted(e)))
    return _normalize(V), list(E)


def load_obj(path):
    V, E = [], set()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                V.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                ids = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(len(ids)):
                    E.add(tuple(sorted((ids[k], ids[(k + 1) % len(ids)]))))
    return _normalize(V), list(E)


# Build the model list (a loaded .obj is added first if supplied)
MODELS = []
if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".obj"):
    try:
        MODELS.append((os.path.basename(sys.argv[1]).upper(), load_obj(sys.argv[1])))
    except Exception as exc:
        print(f"[warn] could not load {sys.argv[1]}: {exc}")
MODELS += [
    ("GEODESIC SPHERE", make_uv_sphere()),
    ("TORUS RING", make_torus()),
    ("ICOSAHEDRON", make_icosahedron()),
    ("REACTOR CUBE", make_cube()),
]


# ===========================================================================
#  3D math
# ===========================================================================
def rotation(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rx @ Ry


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """Orientation-agnostic finger-extension test.
    A finger is 'up' when its tip is farther from the wrist than its
    middle joint. Returns [thumb, index, middle, ring, pinky] booleans."""
    def dw(i):
        return math.hypot(pts[i][0] - pts[0][0], pts[i][1] - pts[0][1])
    up = [dw(4) > dw(2)]                     # thumb: tip vs MCP
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        up.append(dw(tip) > dw(pip))
    return up


# ===========================================================================
#  Hologram renderer
# ===========================================================================
FOCAL = 720.0
CAM_DIST = 4.6


def render_hologram(img, V, E, yaw, pitch, scale, t):
    cx, cy = FRAME_W // 2, FRAME_H // 2
    P = (rotation(yaw, pitch) @ V.T).T * scale
    P[:, 2] += CAM_DIST
    z = P[:, 2]
    xs = (cx + FOCAL * P[:, 0] / z).astype(int)
    ys = (cy - FOCAL * P[:, 1] / z).astype(int)
    zmin, zmax = float(z.min()), float(z.max())
    span = max(zmax - zmin, 1e-6)

    # scanning sweep band (holographic projector look)
    sweep_y = int(cy + math.sin(t * 1.5) * scale * FOCAL / CAM_DIST * 0.9)

    for a, b in E:
        depth = 1.0 - ((z[a] + z[b]) * 0.5 - zmin) / span   # near = 1, far = 0
        k = 0.30 + 0.70 * depth
        col = (int(CYAN[0] * k), int(CYAN[1] * k), int(CYAN[2] * k))
        pa, pb = (xs[a], ys[a]), (xs[b], ys[b])
        if depth > 0.75:                       # bloom on the nearest edges
            cv2.line(img, pa, pb, col, 3, cv2.LINE_AA)
        cv2.line(img, pa, pb, col, 1, cv2.LINE_AA)
        # sweep highlight
        if abs((pa[1] + pb[1]) * 0.5 - sweep_y) < 6:
            cv2.line(img, pa, pb, WHITE, 1, cv2.LINE_AA)

    # bright vertices
    for i in range(len(V)):
        depth = 1.0 - (z[i] - zmin) / span
        if depth > 0.55:
            cv2.circle(img, (xs[i], ys[i]), 2, WHITE, -1, cv2.LINE_AA)

    # containment ring + projector base
    radius = int(scale * FOCAL / CAM_DIST * 1.15)
    cv2.circle(img, (cx, cy), radius, CYAN_D, 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy + radius), (radius, radius // 4), 0, 0, 360, CYAN_D, 1, cv2.LINE_AA)


def draw_hands(img, hands):
    for pts in hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], CYAN_D, 1, cv2.LINE_AA)
        for tip in (4, 8, 12, 16, 20):
            cv2.circle(img, pts[tip], 4, AMBER, 1, cv2.LINE_AA)


def panel(img, x1, y1, x2, y2, alpha=0.35):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (35, 18, 0), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN_D, 1, cv2.LINE_AA)


def text(img, s, org, sc=0.5, col=CYAN, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def frame_brackets(img):
    h, w = img.shape[:2]
    L = 38
    for cx, cy, dx, dy in [(14, 14, 1, 1), (w - 14, 14, -1, 1),
                           (14, h - 14, 1, -1), (w - 14, h - 14, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), CYAN_D, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), CYAN_D, 2, cv2.LINE_AA)


# ===========================================================================
#  Main
# ===========================================================================
def main():
    global FRAME_W, FRAME_H

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # interaction state
    model_i = 0
    name, (V, E) = MODELS[model_i]
    yaw = pitch = 0.0
    vyaw = vpitch = 0.0
    scale = 1.0
    last_palm = None
    gesture = "BOOTING"
    show_hands = True

    loop_start = time.time()
    last_ts = -1
    last_time = time.time()
    fps = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        FRAME_H, FRAME_W = frame.shape[0], frame.shape[1]

        # async hand detection
        ts = int((time.time() - loop_start) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts
        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)

        with _result_lock:
            hands = list(_latest_hands)
        t = time.time() - loop_start

        # ---- hand-driven control (one hand does everything) ----
        if len(hands) >= 1:
            hand = hands[0]
            up = fingers_up(hand)
            palm = hand[9]
            handsize = max(dist(hand[0], hand[9]), 1e-3)

            if up[1] and up[0] and not any(up[2:]):
                # SCALE -- thumb+index caliper, normalized by hand size
                gap = dist(hand[4], hand[8]) / handsize
                target = float(np.clip(gap * 2.4, 0.4, 3.0))
                scale += (target - scale) * 0.3
                last_palm = None
                vyaw = vpitch = 0.0
                gesture = "SCALE"
            elif sum(up) >= 3:
                # ROTATE -- move your open hand, hologram follows
                if last_palm is not None:
                    vyaw = (palm[0] - last_palm[0]) * 0.012
                    vpitch = -(palm[1] - last_palm[1]) * 0.012
                last_palm = palm
                gesture = "ROTATE"
            else:
                # FIST -- freeze and hold orientation
                last_palm = None
                vyaw = vpitch = 0.0
                gesture = "HOLD"
        else:
            last_palm = None
            gesture = "IDLE SPIN"
            vyaw += (0.005 - vyaw) * 0.02        # ease toward gentle auto-spin

        # integrate + inertia
        yaw += vyaw
        pitch = float(np.clip(pitch + vpitch, -1.3, 1.3))
        if gesture != "ROTATE":
            vyaw *= 0.94
            vpitch *= 0.94

        # ---- render ----
        frame_brackets(frame)
        if show_hands:
            draw_hands(frame, hands)
        render_hologram(frame, V, E, yaw, pitch, scale, t)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        panel(frame, 14, 14, 312, 104)
        text(frame, "S.T.A.R.K.  HOLO-FORGE", (28, 38), 0.6, CYAN, 1)
        text(frame, f"MODEL : {name}", (28, 62), 0.45, AMBER, 1)
        text(frame, f"FPS {fps:4.0f}   SCALE {scale*100:3.0f}%   HANDS {len(hands)}",
             (28, 88), 0.45, WHITE, 1)

        panel(frame, 14, FRAME_H - 48, 312, FRAME_H - 14)
        text(frame, f"MODE: {gesture}", (28, FRAME_H - 24), 0.55, CYAN, 1)

        text(frame, "[n] model   [r] reset   [g] hands   [q] quit",
             (FRAME_W - 430, FRAME_H - 22), 0.45, CYAN_D, 1)

        cv2.imshow("STARK // Holographic 3D Manipulator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            model_i = (model_i + 1) % len(MODELS)
            name, (V, E) = MODELS[model_i]
        elif key == ord('r'):
            yaw = pitch = vyaw = vpitch = 0.0
            scale = 1.0
        elif key == ord('g'):
            show_hands = not show_hands

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
