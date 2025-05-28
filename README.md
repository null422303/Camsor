# Camsor: Hand Gesture Mouse Control

This project uses computer vision to control the mouse cursor with hand gestures captured via a webcam. It leverages MediaPipe Hands for hand detection and tracking, allowing the right hand to move the cursor and the left hand to perform clicks when forming a fist.

## Features
- **Cursor Movement**: Move the cursor using the topmost point of the right hand.
- **Click Detection**: Perform a left-click by making a fist with the left hand.
- **Smooth Cursor Control**: Uses exponential moving average for smooth cursor movement.
- **Optimized Performance**: Processes hand detection every few frames to reduce CPU usage.
- **Brightness Adjustment**: Enhances webcam feed for better hand detection in low-light conditions.

## Requirements
- Python 3.8+
- Webcam
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-gesture-mouse-control.git
   cd hand-gesture-mouse-control
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python hand_gesture_control.py
   ```

## Usage
- **Right Hand**: Move your right hand to control the mouse cursor. The topmost point of the hand maps to the cursor position.
- **Left Hand**: Make a fist with your left hand to perform a left-click (0.75-second cooldown between clicks).
- **Exit**: Press the `Esc` key to close the application.
- A small window (`Camsor`) displays the webcam feed with hand landmarks and cursor position, positioned in the bottom-right corner of the screen.

## Notes
- Ensure good lighting for optimal hand detection.
- The program supports up to two hands and uses fallback resolutions if the preferred webcam settings are unavailable.
- Frame processing time is logged every 30 frames for performance monitoring.

## Dependencies
Listed in `requirements.txt`:
- opencv-python
- mediapipe
- numpy
- pynput
- pyautogui

## License
This project is licensed under the MIT License.
