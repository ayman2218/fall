import cv2
import numpy as np
import time
from collections import deque

print("🎥 Fall Detection with Camera")
print("Using: Camera 2, MediaFoundation backend")
print("Press Q or ESC to quit\n")

# Ouvrir la caméra avec le bon backend (MediaFoundation)
cap = cv2.VideoCapture(2, cv2.CAP_MSMF)

if not cap.isOpened():
    print("❌ Could not open camera 2")
    exit(1)

print("✓ Camera opened!")

# Configuration
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Charger MediaPipe Pose
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    import mediapipe as mp
    from mediapipe import Image, ImageFormat
    
    # Load PoseLandmarker
    model_path = 'models/pose_landmarker_lite.task'
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
    )
    pose = vision.PoseLandmarker.create_from_options(options)
    print("✓ MediaPipe PoseLandmarker loaded")
    pose_backend = "tasks"
    
except Exception as e:
    print(f"⚠ PoseLandmarker failed: {e}")
    print("Trying mp.solutions.pose...")
    
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose_backend = "solutions"
        print("✓ mp.solutions.pose loaded")
    except Exception as e2:
        print(f"❌ Both pose backends failed: {e2}")
        exit(1)

# Fall detection parameters
fall_threshold = 0.3  # How low the head needs to be
fall_buffer = deque(maxlen=10)  # Track last 10 frames
fall_confirmed = False
fall_time = 0

# Distance estimation parameters
# Calibration: shoulder width is approximately 0.4m
# We use the shoulder width to estimate distance
SHOULDER_WIDTH_METERS = 0.4  # meters
CAMERA_FOV = 55  # degrees (typical for laptop cameras)

frame_count = 0
start_time = time.time()

print("\nStarting detection...")
print("Distance Alert: < 1 meter\n")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        person_detected = False
        is_falling = False
        distance_meters = 999  # Default: far away
        
        try:
            if pose_backend == "tasks":
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                results = pose.detect(mp_image)
            else:
                results = pose.process(rgb)
            
            # Check if pose detected
            pose_landmarks = getattr(results, "pose_landmarks", None)
            if pose_landmarks:
                if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
                    landmarks = pose_landmarks[0]
                    person_detected = True
                elif hasattr(pose_landmarks, "landmark"):
                    landmarks = pose_landmarks.landmark
                    person_detected = True
                
                if person_detected and len(landmarks) >= 24:
                    # Get key points
                    nose = landmarks[0]
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    
                    # Calculate positions
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    hip_y = (left_hip.y + right_hip.y) / 2
                    
                    # Estimate distance based on shoulder width
                    shoulder_distance_pixels = abs(left_shoulder.x - right_shoulder.x) * w
                    if shoulder_distance_pixels > 0:
                        # Using pinhole camera model: distance = (real_width * focal_length) / pixel_width
                        # Simplified: distance_meters = (SHOULDER_WIDTH * w) / (2 * shoulder_distance_pixels * tan(FOV/2))
                        import math
                        focal_length_pixels = (w / 2) / math.tan(math.radians(CAMERA_FOV / 2))
                        distance_meters = (SHOULDER_WIDTH_METERS * focal_length_pixels) / shoulder_distance_pixels
                    else:
                        distance_meters = 0
                    
                    # Simple fall detection: head very low
                    if nose.y > shoulder_y + fall_threshold:
                        is_falling = True
                    
                    # Draw skeleton
                    for lm in landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Draw connections (skeleton)
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
            print(f"Error processing pose: {e}")
        
        # Track fall state
        fall_buffer.append(is_falling)
        
        if person_detected:
            # Confirm fall if detected in multiple frames
            fall_count = sum(fall_buffer)
            if fall_count >= 3:
                fall_confirmed = True
                fall_time = time.time()
            elif time.time() - fall_time > 2:
                fall_confirmed = False
        else:
            fall_confirmed = False
        
        # Draw status
        status = "FALL DETECTED!" if fall_confirmed else ("Possible Fall" if is_falling else "Normal")
        color = (0, 0, 255) if fall_confirmed else ((0, 165, 255) if is_falling else (0, 255, 0))
        
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        if not person_detected:
            cv2.putText(frame, "NO PERSON", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            # Display distance
            distance_text = f"Distance: {distance_meters:.2f}m"
            distance_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 0)
            cv2.putText(frame, distance_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, distance_color, 2)
            
            # ALERT if distance < 1m AND fall detected
            if distance_meters < 1.0 and (is_falling or fall_confirmed):
                alert_text = "🚨 ROBOT ALERT! CLOSE FALL DETECTED! 🚨"
                cv2.putText(frame, "*** ROBOT ALERT ***", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, f"Person {distance_meters:.2f}m away - ASSIST NEEDED", (20, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # Blink effect
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 5)
        
        # FPS counter
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Fall Detection", frame)
        
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
