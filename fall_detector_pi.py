#!/usr/bin/env python3
"""
Fall Detection System optimized for Raspberry Pi 5
- Reduced resolution (320x240)
- Frame skipping (every 2nd frame)
- Lite MediaPipe model
- Minimal UI rendering
"""

import cv2
import numpy as np
import time
from collections import deque
import math
import sys

print("🍓 Fall Detection on Raspberry Pi 5")
print("Press Q or ESC to quit\n")

# Reduce resolution for Pi performance
RESOLUTION = (320, 240)  # Instead of 640x480
SKIP_FRAMES = 2  # Process every 2nd frame
TARGET_FPS = 15  # Lower FPS for Pi

cap = None
camera_config = None

# Essayer caméra USB
configs = [
    (0, cv2.CAP_ANY),
    (1, cv2.CAP_ANY),
    (2, cv2.CAP_ANY),
]

for cam_idx, backend in configs:
    print(f"Trying camera {cam_idx}...", end="")
    test_cap = cv2.VideoCapture(cam_idx, backend)
    
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None:
            print(f" ✓ OK!")
            cap = test_cap
            camera_config = (cam_idx, backend)
            break
        else:
            print(" opened but no frames")
    else:
        print(" not available")
    
    test_cap.release()
    time.sleep(0.2)

if cap is None:
    print("❌ Could not open camera!")
    sys.exit(1)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"✓ Camera: index={camera_config[0]}, res={RESOLUTION}, fps={TARGET_FPS}")

# Load MediaPipe (lite version)
print("Loading MediaPipe...")
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    from mediapipe import Image, ImageFormat
    
    model_path = 'models/pose_landmarker_lite.task'
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
    )
    pose = vision.PoseLandmarker.create_from_options(options)
    pose_backend = "tasks"
    print("✓ MediaPipe loaded (tasks)")
except Exception as e:
    print(f"Tasks API failed: {e}")
    print("Falling back to MediaPipe solutions API...")
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lite version
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose_backend = "solutions"
        print("✓ MediaPipe loaded (solutions)")
    except Exception as e2:
        print(f"\n❌ MediaPipe failed to load: {e2}")
        print("\n--- FIX ---")
        print("This is usually a protobuf version conflict.")
        print("Run the following inside your virtual environment:")
        print('  pip install "protobuf>=3.20.3,<4.0.0"')
        print("  pip install --upgrade mediapipe")
        print("Then restart the script.")
        sys.exit(1)

# Parameters
SHOULDER_WIDTH_METERS = 0.4
CAMERA_FOV = 55
fall_threshold = 0.08  # Slightly higher than desktop (0.05)
falling_speed_threshold = 0.02
fall_buffer = deque(maxlen=5)  # Smaller buffer for Pi
fall_confirmed = False
fall_confirmed_time = 0
prev_nose_y = None
prev_shoulder_y = None
FALL_DISPLAY_DURATION = 2.0

frame_count = 0
start_time = time.time()

print("Starting detection...\n")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame read failed")
        break
    
    h, w = frame.shape[:2]
    frame_count += 1
    
    # Skip frames for performance
    if frame_count % SKIP_FRAMES != 0:
        cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.imshow("Fall Detection - Pi5", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break
        continue
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if pose_backend == "tasks":
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = pose.detect(mp_image)
            pose_landmarks = getattr(results, "pose_landmarks", None)
        else:
            results = pose.process(rgb_frame)
            pose_landmarks = getattr(results, "pose_landmarks", None)
        
        person_detected = False
        is_falling = False
        distance_meters = None
        landmarks = None
        
        if pose_landmarks:
            if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
                landmarks = pose_landmarks[0]
                person_detected = True
            elif hasattr(pose_landmarks, "landmark"):
                landmarks = pose_landmarks.landmark
                person_detected = True
        
        if person_detected and landmarks and len(landmarks) >= 24:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            nose_y = nose.y
            shoulder_y = min(left_shoulder.y, right_shoulder.y)
            
            # Fall detection
            if nose_y > shoulder_y + fall_threshold:
                is_falling = True
            
            # Velocity detection
            if prev_nose_y is not None:
                nose_velocity = nose_y - prev_nose_y
                shoulder_velocity = shoulder_y - prev_shoulder_y
                
                if nose_velocity > falling_speed_threshold and nose_velocity > shoulder_velocity + 0.01:
                    is_falling = True
            
            prev_nose_y = nose_y
            prev_shoulder_y = shoulder_y
            
            # Distance
            shoulder_distance_pixels = abs(left_shoulder.x - right_shoulder.x) * w
            if shoulder_distance_pixels > 0:
                focal_length_pixels = (w / 2) / math.tan(math.radians(CAMERA_FOV / 2))
                distance_meters = (SHOULDER_WIDTH_METERS * focal_length_pixels) / shoulder_distance_pixels
            
            # Confirmation logic
            fall_buffer.append(is_falling)
            is_newly_confirmed = (sum(fall_buffer) >= 2) and not fall_confirmed
            fall_confirmed = sum(fall_buffer) >= 2
            
            if is_newly_confirmed:
                fall_confirmed_time = time.time()
                print(f"🚨 FALL DETECTED! Distance: {distance_meters:.2f}m")
        else:
            fall_buffer.clear()
            if fall_confirmed and time.time() - fall_confirmed_time > FALL_DISPLAY_DURATION:
                fall_confirmed = False
        
        # Draw minimal UI (for performance)
        if fall_confirmed:
            status_text = "⚠️ FALL!"
            color = (0, 0, 255)
        elif is_falling:
            status_text = "? Fall"
            color = (0, 165, 255)
        else:
            status_text = "✓ OK"
            color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        if distance_meters:
            dist_text = f"Distance: {distance_meters:.2f}m"
            dist_color = (0, 255, 0) if distance_meters >= 1.0 else (0, 0, 255)
            cv2.putText(frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
            
            # Robot alert
            if distance_meters < 1.0 and (is_falling or fall_confirmed):
                cv2.putText(frame, "🚨 ALERT 🚨", (w//3, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Fall Detection - Pi5", frame)
        
        # Exit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
            
    except Exception as e:
        print(f"Error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
print("\n✓ Done!")
