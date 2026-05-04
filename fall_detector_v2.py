import cv2
import numpy as np
import time
from collections import deque
import math

print("🎥 Fall Detection with Distance & Robot Alert")
print("Press Q or ESC to quit\n")

# Essayer différentes caméras et backends
cap = None
camera_config = None

# Configurations à essayer (camera_index, backend)
# Priorité: Camera 2 d'abord
configs = [
    (2, cv2.CAP_ANY),
    (2, cv2.CAP_MSMF),
    (1, cv2.CAP_ANY),
    (0, cv2.CAP_ANY),
    (1, cv2.CAP_MSMF),
    (0, cv2.CAP_MSMF),
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
cap.set(cv2.CAP_PROP_FPS, 30)
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
fall_threshold = 0.3
fall_buffer = deque(maxlen=10)
fall_confirmed = False
fall_time = 0

# Tracking
frame_count = 0
start_time = time.time()
robot_alert_time = 0

print("\nStarting detection...")
print("Robot Alert: < 1 meter + Fall Detected\n")

consecutive_failures = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            consecutive_failures += 1
            print(f"⚠ Frame read failed ({consecutive_failures}/10)")
            if consecutive_failures >= 10:
                print("❌ Too many frame read failures")
                break
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        person_detected = False
        is_falling = False
        distance_meters = 999
        
        try:
            if pose_backend == "tasks":
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                results = pose.detect(mp_image)
            else:
                results = pose.process(rgb)
            
            pose_landmarks = getattr(results, "pose_landmarks", None)
            if pose_landmarks:
                if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
                    landmarks = pose_landmarks[0]
                    person_detected = True
                elif hasattr(pose_landmarks, "landmark"):
                    landmarks = pose_landmarks.landmark
                    person_detected = True
                
                if person_detected and len(landmarks) >= 24:
                    # Key points
                    nose = landmarks[0]
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    # Shoulder y position
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    
                    # Calculate distance
                    shoulder_distance_pixels = abs(left_shoulder.x - right_shoulder.x) * w
                    if shoulder_distance_pixels > 10:
                        focal_length_pixels = (w / 2) / math.tan(math.radians(CAMERA_FOV / 2))
                        distance_meters = (SHOULDER_WIDTH_METERS * focal_length_pixels) / shoulder_distance_pixels
                    
                    # Fall detection
                    if nose.y > shoulder_y + fall_threshold:
                        is_falling = True
                    
                    # Draw skeleton
                    for lm in landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Draw connections
                    pose_edges = [
                        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                        (11, 23), (23, 25), (12, 24), (24, 26)
                    ]
                    for start, end in pose_edges:
                        if start < len(landmarks) and end < len(landmarks):
                            x1 = int(landmarks[start].x * w)
                            y1 = int(landmarks[start].y * h)
                            x2 = int(landmarks[end].x * w)
                            y2 = int(landmarks[end].y * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        except Exception as e:
            print(f"Pose processing error: {e}")
        
        # Track fall state
        fall_buffer.append(is_falling)
        
        if person_detected:
            fall_count = sum(fall_buffer)
            if fall_count >= 3:
                fall_confirmed = True
                fall_time = time.time()
            elif time.time() - fall_time > 2:
                fall_confirmed = False
        else:
            fall_confirmed = False
        
        # Draw UI
        status = "FALL!" if fall_confirmed else ("? Fall" if is_falling else "OK")
        color = (0, 0, 255) if fall_confirmed else ((0, 165, 255) if is_falling else (0, 255, 0))
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        if person_detected:
            dist_text = f"Dist: {distance_meters:.2f}m"
            dist_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 0)
            cv2.putText(frame, dist_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, dist_color, 2)
            
            # ROBOT ALERT
            if distance_meters < 1.0 and (is_falling or fall_confirmed):
                robot_alert_time = time.time()
                cv2.putText(frame, "🚨 ROBOT ALERT! 🚨", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, f"Close person at {distance_meters:.2f}m - HELP!", (20, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
                
                # Blink border
                if int(time.time() * 3) % 2 == 0:
                    cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 5)
                
                print(f"🚨 ALERT: Person at {distance_meters:.2f}m FALLING!")
        else:
            cv2.putText(frame, "NO PERSON", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Fall Detection + Robot Alert", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    print("\n✓ Closing...")
    cap.release()
    if pose_backend == "tasks":
        pose.close()
    else:
        pose.close()
    cv2.destroyAllWindows()
    print("✓ Done!")
