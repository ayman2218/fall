import cv2
import numpy as np
import time
from collections import deque
import math
import sys

print("🎥 Fall Detection with Video Support")

# Vérifier si un fichier vidéo est fourni
video_file = sys.argv[1] if len(sys.argv) > 1 else None

cap = None

if video_file:
    print(f"Opening video file: {video_file}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_file}")
        exit(1)
    print("✓ Video file loaded")
else:
    print("No video file provided, trying to open webcam...")
    
    # Essayer différentes caméras et backends
    camera_config = None
    configs = [
        (2, cv2.CAP_ANY),
        (2, cv2.CAP_MSMF),
        (2, cv2.CAP_DSHOW),
        (2, 0),
        (2, 1),
        (2, 1400),
        (1, cv2.CAP_ANY),
        (0, cv2.CAP_ANY),
    ]

    for cam_idx, backend in configs:
        print(f"Trying camera {cam_idx} with backend {backend}...", end="")
        test_cap = cv2.VideoCapture(cam_idx, backend)
        
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None and frame.shape[0] > 0:
                print(f" ✓ SUCCESS!")
                cap = test_cap
                camera_config = (cam_idx, backend)
                break
            else:
                print(" opened but no frames")
        else:
            print(" not available")
        
        test_cap.release()
        time.sleep(0.5)

    if cap is None or not cap.isOpened():
        print("\n❌ Could not open any camera!")
        exit(1)

    print(f"\n✓ Camera configured: index={camera_config[0]}, backend={camera_config[1]}")

# Configuration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Load PoseLandmarker
print("Loading MediaPipe PoseLandmarker...")

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    import mediapipe as mp
    from mediapipe import Image, ImageFormat
    
    model_path = 'models/pose_landmarker_lite.task'
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
    )
    pose = vision.PoseLandmarker.create_from_options(options)
    print("✓ MediaPipe PoseLandmarker loaded")
    pose_backend = "tasks"
    
except Exception as e:
    print(f"Failed: {e}")
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_backend = "solutions"
    print("✓ Using mp.solutions.pose")

# Parameters
SHOULDER_WIDTH_METERS = 0.4
CAMERA_FOV = 55
fall_threshold = 0.05  # Très sensible - moindre chute détectée
falling_speed_threshold = 0.02  # Détection par vélocité (changement rapide)
fall_buffer = deque(maxlen=10)
fall_confirmed = False
fall_time = 0
fall_confirmed_time = 0  # Temps quand fall est confirmé
prev_nose_y = None
prev_shoulder_y = None
FALL_DISPLAY_DURATION = 2.0  # Garder "FALL!" affiché 2 secondes minimum

# Tracking
frame_count = 0
start_time = time.time()
robot_alert_time = 0

print("Starting detection...")
print("Press Q or ESC to quit\n")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or camera disconnected")
        break

    h, w = frame.shape[:2]
    
    # Detect poses
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if pose_backend == "tasks":
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = pose.detect(mp_image)
            pose_landmarks = getattr(results, "pose_landmarks", None)
        else:
            results = pose.process(rgb_frame)
            pose_landmarks = getattr(results, "pose_landmarks", None)
        
        # Check if person detected
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
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Get y-coordinates
            nose_y = nose.y
            shoulder_y = min(left_shoulder.y, right_shoulder.y)
            hip_y = max(left_hip.y, right_hip.y)
            
            # Debug: Print positions (moins souvent)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: nose_y={nose_y:.3f}, shoulder_y={shoulder_y:.3f}, diff={nose_y - shoulder_y:.3f}, threshold={fall_threshold}")
            
            # Check if fallen
            if nose_y > shoulder_y + fall_threshold:
                is_falling = True
            
            # Also detect by falling speed (rapid downward movement)
            if prev_nose_y is not None:
                nose_velocity = nose_y - prev_nose_y
                shoulder_velocity = shoulder_y - prev_shoulder_y
                
                if nose_velocity > falling_speed_threshold and nose_velocity > shoulder_velocity + 0.01:
                    is_falling = True
                    if frame_count % 30 == 0:
                        print(f"  → Falling detected by speed: nose_v={nose_velocity:.4f}, shoulder_v={shoulder_velocity:.4f}")
            
            prev_nose_y = nose_y
            prev_shoulder_y = shoulder_y
            
            # Estimate distance
            shoulder_distance_pixels = abs(left_shoulder.x - right_shoulder.x) * w
            if shoulder_distance_pixels > 0:
                focal_length_pixels = (w / 2) / math.tan(math.radians(CAMERA_FOV / 2))
                distance_meters = (SHOULDER_WIDTH_METERS * focal_length_pixels) / shoulder_distance_pixels
            
            # Add to buffer for fall confirmation
            fall_buffer.append(is_falling)
            # Besoin de 2+ confirmations seulement (au lieu de 3)
            is_newly_confirmed = (sum(fall_buffer) >= 2) and not fall_confirmed
            fall_confirmed = sum(fall_buffer) >= 2
            
            if is_newly_confirmed:
                fall_confirmed_time = time.time()
                print(f"🚨 FALL DETECTED! Distance: {distance_meters:.2f}m")
        else:
            fall_buffer.clear()
            # Fall reste confirmé pendant FALL_DISPLAY_DURATION
            if fall_confirmed and time.time() - fall_confirmed_time > FALL_DISPLAY_DURATION:
                fall_confirmed = False
        
        # Visual feedback
        if person_detected:
            # Draw skeleton
            if pose_backend == "tasks":
                for i, lm in enumerate(landmarks):
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            else:
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Status text
        if fall_confirmed:
            status_color = (0, 0, 255)  # Red
            status_text = "⚠️ FALL!"
            border_color = (0, 0, 255)
        elif is_falling:
            status_color = (0, 165, 255)  # Orange
            status_text = "? Fall"
            border_color = (0, 165, 255)
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "✓ OK"
            border_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Distance
        if distance_meters:
            dist_text = f"Distance: {distance_meters:.2f}m"
            dist_color = (0, 255, 0) if distance_meters >= 1.0 else (0, 0, 255)
            cv2.putText(frame, dist_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_color, 2)
            
            # Robot alert
            if distance_meters < 1.0 and (is_falling or fall_confirmed):
                cv2.putText(frame, "🚨 ROBOT ALERT! 🚨", (w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Border
        thickness = 3
        if robot_alert_time and time.time() - robot_alert_time < 1.0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)
        
        # Display
        cv2.imshow("Fall Detection", frame)
        
        # Input
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        
    except Exception as e:
        print(f"Error: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
print("Done!")
