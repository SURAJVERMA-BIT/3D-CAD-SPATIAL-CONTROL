import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from concurrent.futures import ThreadPoolExecutor
import time

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Variables for interaction
zoom_sensitivity = 20
rotation_sensitivity = 0.5
grab_threshold = 40
dynamic_frame_skip = 2
prev_distance = None
grabbed = False
prev_hand_position = None
double_click_done = False
frame_counter = 0

# Thread pool executor
executor = ThreadPoolExecutor(max_workers=3)

# Calculate distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Hand gesture processing
def process_hand_gestures(landmarks_list):
    global prev_distance, grabbed, double_click_done, prev_hand_position

    # Zoom Logic (Two Hands)
    if len(landmarks_list) == 2:
        hand1_index = landmarks_list[0][8]
        hand2_index = landmarks_list[1][8]
        current_distance = calculate_distance(hand1_index, hand2_index)

        if prev_distance is not None:
            zoom_change = int((current_distance - prev_distance) * zoom_sensitivity)
            if zoom_change != 0:
                pyautogui.scroll(zoom_change)

        prev_distance = current_distance
    else:
        prev_distance = None

    # Grabbing and Rotation Logic (Single Hand)
    if len(landmarks_list) == 1:
        hand_landmarks = landmarks_list[0]
        thumb_tip = hand_landmarks[4]
        index_finger_tip = hand_landmarks[8]
        grab_distance = calculate_distance(thumb_tip, index_finger_tip)

        if grab_distance < grab_threshold:  # Grab detected
            if not grabbed:
                pyautogui.mouseDown()
                grabbed = True

        else:  # Release detected
            if grabbed:
                pyautogui.mouseUp()
                grabbed = False

        if grabbed:
            index_finger = hand_landmarks[8]
            if prev_hand_position is not None:
                dx = index_finger[0] - prev_hand_position[0]
                dy = index_finger[1] - prev_hand_position[1]
                pyautogui.moveRel(dx * rotation_sensitivity, dy * rotation_sensitivity, duration=0)

            prev_hand_position = index_finger
        else:
            prev_hand_position = None

# Camera Feed
cap = cv2.VideoCapture(0)

def adjust_frame_skip(fps):
    """Adjust frame skip dynamically based on FPS."""
    global dynamic_frame_skip
    if fps < 20:
        dynamic_frame_skip = min(dynamic_frame_skip + 1, 5)
    elif fps > 25 and dynamic_frame_skip > 1:
        dynamic_frame_skip -= 1

# Start processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_counter += 1

    if frame_counter % dynamic_frame_skip == 0:  # Skip frames dynamically
        result = hands.process(rgb_frame)
        landmarks_list = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Process gestures asynchronously
        executor.submit(process_hand_gestures, landmarks_list)

    # Display the feed
    cv2.putText(frame, "Optimized Gesture Control", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Gesture Control", frame)

    # Dynamic FPS-Based Frame Skip Adjustment
    if frame_counter % 30 == 0:  # Check every 30 frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        adjust_frame_skip(fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
