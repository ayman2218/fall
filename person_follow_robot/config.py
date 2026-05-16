"""Configuration for person-follow robot (Raspberry Pi 5 + L298N).

BCM PIN NUMBERING
---------------
These are placeholders. Update them to match your wiring.

Tips:
- ENA/ENB should be PWM-capable GPIOs.
- IN pins are standard digital outputs.
- If your robot drives backward when it should drive forward, swap IN1/IN2 (or IN3/IN4)
  or set INVERT_LEFT / INVERT_RIGHT.
"""

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FLIP_FRAME = True  # mirror for "selfie" behavior; set False if you want natural camera view

# L298N GPIO pins (BCM) — CHANGE THESE
LEFT_EN_PWM = 18   # ENA (PWM)
LEFT_IN1 = 23
LEFT_IN2 = 24

RIGHT_EN_PWM = 19  # ENB (PWM)
RIGHT_IN3 = 27
RIGHT_IN4 = 22

# Optional direction inversion
INVERT_LEFT = False
INVERT_RIGHT = False

# PWM
PWM_FREQUENCY_HZ = 1000

# Follow control tuning
TARGET_DISTANCE_M = 0.8
STOP_DISTANCE_M = 0.45   # too close: stop/creep
MAX_DISTANCE_M = 3.5     # if farther than this, we treat distance as unreliable/far

# How strongly we steer to center the person
K_TURN = 1.2  # proportional gain on normalized x error

# How strongly we move forward/back to reach target distance
K_FWD = 0.9   # proportional gain on distance error

# Speed limits (0..1)
MAX_FWD = 0.7
MAX_REV = 0.35
MAX_TURN = 0.6

# Smoothing (0=no smoothing, 1=very slow). 0.2..0.4 is typical.
SMOOTHING = 0.25

# Pose + distance estimation parameters
SHOULDER_WIDTH_METERS = 0.40
CAMERA_FOV_DEG = 55

# Person detection + safety
NO_PERSON_FRAMES_TO_STOP = 8

# MediaPipe options
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"  # keep same style as your fall_detector.py
