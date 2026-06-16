import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pynput.mouse import Button, Controller
import math
import os
import threading
import time

# ===========================================================================
#  MediaPipe HandLandmarker  (Tasks API)
#  LIVE_STREAM running mode => detection runs async on its own thread, so the
#  render loop never blocks on CPU inference. This is the main lag fix.
# ===========================================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "hand_landmarker.task")

# Frame size is filled in once the camera delivers the first frame; the async
# result callback needs it to convert normalized landmarks -> pixels.
FRAME_W, FRAME_H = 640, 480

result_lock = threading.Lock()
latest_hands_px = []          # list[ list[(x, y)] ] for rendering


def _on_result(result, output_image, timestamp_ms):
    """Runs on MediaPipe's worker thread for every async detection."""
    global latest_hands_px
    hands_px = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pts = [(int(lm.x * FRAME_W), int(lm.y * FRAME_H)) for lm in hand]
            hands_px.append(pts)
    with result_lock:
        latest_hands_px = hands_px
    process_hand_gestures(hands_px)


hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    result_callback=_on_result,
)
landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Hand skeleton topology (mp.solutions.hands.HAND_CONNECTIONS isn't shipped on
# Python 3.13, so the 21-landmark connection set is defined inline).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]
FINGERTIPS = [4, 8, 12, 16, 20]

# HUD palette (BGR) — arc-reactor cyan with amber/red accents
CYAN   = (255, 230, 90)
CYAN_D = (170, 130, 40)
AMBER  = (40, 200, 255)
RED    = (70, 70, 255)
WHITE  = (255, 255, 255)

# Initialize pynput for mouse control
mouse = Controller()

# Interaction tuning
zoom_sensitivity = 10.0
move_sensitivity = 3.0
grab_threshold = 32             # pixels for pinch/grab detection
prev_distance = None
grabbed = False
prev_hand_position = None
current_gesture = "STANDBY"     # surfaced on the HUD


# ===========================================================================
#  Gesture logic
# ===========================================================================
def calculate_distance(p1, p2):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def process_hand_gestures(landmarks_list):
    global prev_distance, grabbed, prev_hand_position, current_gesture

    if not landmarks_list:
        current_gesture = "SCANNING"
        prev_distance = None
        if grabbed:
            mouse.release(Button.left)
            grabbed = False
        prev_hand_position = None
        return

    # --- Zoom (two hands: pinch the gap between index tips) ---
    if len(landmarks_list) == 2:
        hand1_index = landmarks_list[0][8]
        hand2_index = landmarks_list[1][8]
        current_distance = calculate_distance(hand1_index, hand2_index)

        if prev_distance is not None:
            zoom_change = (current_distance - prev_distance) * zoom_sensitivity
            if abs(zoom_change) > 5:
                mouse.scroll(0, int(zoom_change / 20))
                current_gesture = "ZOOM +" if zoom_change > 0 else "ZOOM -"
        prev_distance = current_distance

        if grabbed:
            mouse.release(Button.left)
            grabbed = False
            prev_hand_position = None
        return
    else:
        prev_distance = None

    # --- Grab & move (single hand: thumb-index pinch) ---
    if len(landmarks_list) == 1:
        hand_landmarks = landmarks_list[0]
        thumb_tip = hand_landmarks[4]
        index_finger_tip = hand_landmarks[8]
        grab_distance = calculate_distance(thumb_tip, index_finger_tip)

        if grab_distance < grab_threshold:
            if not grabbed:
                mouse.press(Button.left)
                grabbed = True
        else:
            if grabbed:
                mouse.release(Button.left)
                grabbed = False

        if grabbed:
            current_gesture = "GRAB / DRAG"
            index_finger = hand_landmarks[8]
            if prev_hand_position is not None:
                dx = (index_finger[0] - prev_hand_position[0]) * move_sensitivity
                dy = (index_finger[1] - prev_hand_position[1]) * move_sensitivity
                current_pos = mouse.position
                mouse.position = (current_pos[0] + dx, current_pos[1] + dy)
            prev_hand_position = index_finger
        else:
            current_gesture = "TRACKING"
            prev_hand_position = None


# ===========================================================================
#  HUD rendering
# ===========================================================================
def hud_text(img, text, org, scale=0.5, color=CYAN, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def glow_line(img, p1, p2, color=CYAN, thick=2):
    cv2.line(img, p1, p2, color, thick + 2, cv2.LINE_AA)   # outer glow
    cv2.line(img, p1, p2, WHITE, max(1, thick - 1), cv2.LINE_AA)  # bright core


def reticle(img, center, radius, color, t):
    """Rotating targeting brackets + center dot."""
    x, y = int(center[0]), int(center[1])
    spin = (t * 80) % 360
    for a in range(0, 360, 90):
        ar = math.radians(a + spin)
        x1 = int(x + math.cos(ar) * radius)
        y1 = int(y + math.sin(ar) * radius)
        x2 = int(x + math.cos(ar) * (radius + 7))
        y2 = int(y + math.sin(ar) * (radius + 7))
        cv2.line(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), radius, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)


def draw_hand(img, pts, t):
    for a, b in HAND_CONNECTIONS:
        glow_line(img, pts[a], pts[b], CYAN, 2)
    for p in pts:
        cv2.circle(img, p, 3, CYAN, -1, cv2.LINE_AA)
    for tip in FINGERTIPS:
        reticle(img, pts[tip], 7, AMBER, t)


def target_crosshair(img, p, color, t):
    h, w = img.shape[:2]
    x, y = p
    cv2.line(img, (0, y), (w, y), CYAN_D, 1, cv2.LINE_AA)
    cv2.line(img, (x, 0), (x, h), CYAN_D, 1, cv2.LINE_AA)
    reticle(img, p, 15, color, t)


def frame_brackets(img, color):
    h, w = img.shape[:2]
    L = 34
    for cx, cy, dx, dy in [(12, 12, 1, 1), (w - 12, 12, -1, 1),
                           (12, h - 12, 1, -1), (w - 12, h - 12, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 2, cv2.LINE_AA)


def panel(img, x1, y1, x2, y2, alpha=0.35):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (35, 18, 0), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN_D, 1, cv2.LINE_AA)


def arc_reactor(img, center, t):
    """Pulsing arc-reactor emblem."""
    x, y = center
    pulse = int(6 + 3 * math.sin(t * 4))
    for r, c in [(18, CYAN_D), (12, CYAN)]:
        cv2.circle(img, (x, y), r, c, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), pulse, WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), pulse + 3, CYAN, 1, cv2.LINE_AA)


# ===========================================================================
#  Camera setup
# ===========================================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ===========================================================================
#  Main loop
# ===========================================================================
loop_start = time.time()
last_ts_ms = -1
fps = 0.0
last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    FRAME_H, FRAME_W = frame.shape[0], frame.shape[1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fire-and-forget async detection (result arrives via _on_result)
    ts_ms = int((time.time() - loop_start) * 1000)
    if ts_ms <= last_ts_ms:
        ts_ms = last_ts_ms + 1
    last_ts_ms = ts_ms
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    landmarker.detect_async(mp_image, ts_ms)

    t = time.time() - loop_start

    # Snapshot the most recent landmarks for rendering
    with result_lock:
        hands_px = list(latest_hands_px)

    # --- HUD ---
    frame_brackets(frame, CYAN_D)

    for pts in hands_px:
        draw_hand(frame, pts, t)
    if hands_px:
        target_crosshair(frame, hands_px[0][8], RED if grabbed else AMBER, t)

    # smoothed FPS
    now = time.time()
    inst = 1.0 / max(now - last_time, 1e-6)
    fps = 0.9 * fps + 0.1 * inst
    last_time = now

    # telemetry panel (top-left)
    panel(frame, 12, 12, 250, 96)
    arc_reactor(frame, (34, 34), t)
    hud_text(frame, "S.T.A.R.K. HUD", (58, 30), 0.55, CYAN, 1)
    hud_text(frame, "// SYSTEM ONLINE", (58, 48), 0.4, AMBER, 1)
    hud_text(frame, f"FPS {fps:4.0f}   HANDS {len(hands_px)}", (24, 78), 0.45, WHITE, 1)

    # gesture readout (bottom-left)
    panel(frame, 12, FRAME_H - 46, 250, FRAME_H - 12)
    hud_text(frame, f"GESTURE: {current_gesture}", (24, FRAME_H - 24), 0.5, CYAN, 1)

    cv2.imshow("STARK // Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
