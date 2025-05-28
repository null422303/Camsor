import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.mouse import Button, Controller as MouseController
import pyautogui

# ------------------- Configuration and Initialization -------------------
# Initialize mouse controller and retrieve screen dimensions
mouse = MouseController()
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Initialize MediaPipe Hands with optimized detection parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Constants for processing
BRIGHTNESS_FACTOR = 50  # Brightness adjustment for better hand detection
SMOOTHING_ALPHA = 0.6  # Smoothing factor for exponential moving average
MAX_Y_NORM = 0.75  # Maximum y-coordinate in normalized space for mapping
CLICK_COOLDOWN = 0.75  # Cooldown period for mouse clicks (seconds)
MOVE_INTERVAL = 0.01  # Minimum interval between mouse movements (seconds)
PROCESS_EVERY_N_FRAMES = 2  # Process hands every N frames for performance
MAX_NO_HAND_FRAMES = 10  # Frames to wait before resetting cursor position

# ------------------- Webcam Setup -------------------
def initialize_webcam():
    """Initialize webcam with fallback resolutions for compatibility."""
    cap = cv2.VideoCapture(0)
    resolutions = [(640, 480), (320, 240)]  # Preferred resolutions
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == width:
            print(f"Webcam initialized at {width}x{height} resolution")
            return cap, width, height
    print("Warning: Failed to set webcam resolution, using default")
    return cap, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize webcam and get frame dimensions
cap, FRAME_WIDTH, FRAME_HEIGHT = initialize_webcam()

# ------------------- Utility Functions -------------------
def adjust_brightness(frame, brightness_factor):
    """Increase frame brightness efficiently for better hand detection."""
    return np.clip(frame.astype(np.int16) + brightness_factor, 0, 255).astype(np.uint8)

def get_topmost_point(hand_landmarks):
    """Identify the topmost point in the hand mesh for cursor control."""
    min_y = float('inf')
    top_x, top_y = 0, 0
    for landmark in hand_landmarks.landmark:
        if landmark.y < min_y:
            min_y = landmark.y
            top_x = landmark.x
            top_y = landmark.y
    return top_x, top_y

def is_fist(hand_landmarks):
    """Detect if the hand is in a fist gesture for click action."""
    finger_pairs = [(8, 6), (12, 10)]  # Index and middle finger tip vs pip
    for tip_idx, pip_idx in finger_pairs:
        if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y:
            return False
    return True

# ------------------- Window Setup -------------------
# Configure display window
cv2.namedWindow('Camsor', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camsor', 320, 240)  # Set display window size
# Position window in bottom-right corner with 10-pixel margin
cv2.moveWindow('Camsor', SCREEN_WIDTH - 320 - 10, SCREEN_HEIGHT - 240 - 10)

# ------------------- Main Loop Variables -------------------
prev_x, prev_y = None, None  # Previous smoothed cursor position
last_click_time = 0  # Timestamp of last mouse click
last_move_time = 0  # Timestamp of last mouse movement
frame_counter = 0  # Counter for frame skipping
no_hand_frames = 0  # Counter for frames without detected hands
last_results = None  # Store last hand detection results

# ------------------- Main Processing Loop -------------------
while True:
    start_time = time.time()

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Preprocess frame: flip horizontally and adjust brightness
    frame = cv2.flip(frame, 1)
    frame = adjust_brightness(frame, BRIGHTNESS_FACTOR)

    # Process hands every N frames or if no hands detected recently
    frame_counter += 1
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0 or no_hand_frames > 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        last_results = results
    else:
        results = last_results

    current_time = time.time()
    # Process detected hands
    if results and results.multi_hand_landmarks and results.multi_handedness:
        no_hand_frames = 0  # Reset no-hand counter
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks[:2], results.multi_handedness[:2]):
            hand_label = handedness.classification[0].label

            # Right hand: Control cursor movement
            if hand_label == 'Right' and current_time - last_move_time >= MOVE_INTERVAL:
                norm_x, norm_y = get_topmost_point(hand_landmarks)
                # Map normalized coordinates to screen dimensions
                x = norm_x * SCREEN_WIDTH
                y = (norm_y / MAX_Y_NORM) * SCREEN_HEIGHT

                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y
                    continue

                # Apply exponential moving average for smooth cursor movement
                x = SMOOTHING_ALPHA * x + (1 - SMOOTHING_ALPHA) * prev_x
                y = SMOOTHING_ALPHA * y + (1 - SMOOTHING_ALPHA) * prev_y

                # Constrain cursor within screen boundaries
                x = max(0, min(x, SCREEN_WIDTH - 1))
                y = max(0, min(y, SCREEN_HEIGHT - 1))

                # Update mouse position
                mouse.position = (x, y)
                last_move_time = current_time
                prev_x, prev_y = x, y

                # Draw cursor indicator on frame
                frame_x = int(norm_x * FRAME_WIDTH)
                frame_y = int(norm_y * FRAME_HEIGHT)
                cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)

            # Left hand: Detect fist for clicking
            elif hand_label == 'Left':
                if is_fist(hand_landmarks) and current_time - last_click_time >= CLICK_COOLDOWN:
                    mouse.click(Button.left, 1)
                    last_click_time = current_time

            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        no_hand_frames += 1
        if no_hand_frames >= MAX_NO_HAND_FRAMES:
            prev_x, prev_y = None, None  # Reset cursor position after prolonged absence

    # Display processed frame
    cv2.imshow('Camsor', frame)

    # Log frame processing time every 30 frames
    frame_time = (time.time() - start_time) * 1000
    if frame_counter % 30 == 0:
        print(f"Frame processing time: {frame_time:.2f}ms")

    # Exit on Escape key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------- Cleanup -------------------
cap.release()
cv2.destroyAllWindows()
hands.close()

# Credits: Grok AI, Null422303(me obviously)
