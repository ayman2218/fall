"""Person following robot using MediaPipe Pose.

Goal
----
Keep the robot near the person by:
- steering so the person stays centered in the image
- moving forward/back to keep target distance

This is designed to live alongside your existing fall detection scripts.

Run
---
python3 person_follow_robot/follow_pose.py

Notes
-----
- Update pins in person_follow_robot/config.py
- L298N requires separate motor power (battery). Share GND with the Pi.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2

from person_follow_robot.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FLIP_FRAME,
    LEFT_EN_PWM,
    LEFT_IN1,
    LEFT_IN2,
    RIGHT_EN_PWM,
    RIGHT_IN3,
    RIGHT_IN4,
    PWM_FREQUENCY_HZ,
    INVERT_LEFT,
    INVERT_RIGHT,
    TARGET_DISTANCE_M,
    STOP_DISTANCE_M,
    MAX_DISTANCE_M,
    K_TURN,
    K_FWD,
    MAX_FWD,
    MAX_REV,
    MAX_TURN,
    SMOOTHING,
    SHOULDER_WIDTH_METERS,
    CAMERA_FOV_DEG,
    NO_PERSON_FRAMES_TO_STOP,
    POSE_MODEL_PATH,
)

from person_follow_robot.motor_l298n import L298NMotorDriver


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class PersonState:
    detected: bool
    cx_norm: float | None  # 0..1
    distance_m: float | None


def estimate_distance_m(landmarks, w: int) -> float | None:
    """Estimate distance using shoulder width (same idea as your existing code)."""
    try:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        shoulder_distance_pixels = abs(left_shoulder.x - right_shoulder.x) * w
        if shoulder_distance_pixels <= 10:
            return None

        focal_length_pixels = (w / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG / 2.0))
        distance_m = (SHOULDER_WIDTH_METERS * focal_length_pixels) / shoulder_distance_pixels
        return float(distance_m)
    except Exception:
        return None


def estimate_center_x_norm(landmarks) -> float | None:
    """Compute a stable horizontal center estimate from several landmarks."""
    idxs = [0, 11, 12, 23, 24]  # nose, shoulders, hips
    xs = []
    for i in idxs:
        try:
            xs.append(float(landmarks[i].x))
        except Exception:
            pass
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def detect_person_pose(pose, rgb_frame):
    """Unify MediaPipe outputs into a landmarks list."""
    results = pose.detect(rgb_frame)
    pose_landmarks = getattr(results, "pose_landmarks", None)

    if not pose_landmarks:
        return None

    # tasks PoseLandmarker returns a list of pose(s)
    if isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
        return pose_landmarks[0]

    # fallback: mp.solutions style
    if hasattr(pose_landmarks, "landmark"):
        return pose_landmarks.landmark

    return None


def build_pose():
    """Build MediaPipe pose detector (tasks PoseLandmarker)."""
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    print("\n🤖 Person Follow (Pose) - Raspberry Pi 5 + L298N")
    print("Press Q or ESC to quit\n")

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Motors
    motors = L298NMotorDriver(
        left_en_pwm=LEFT_EN_PWM,
        left_in1=LEFT_IN1,
        left_in2=LEFT_IN2,
        right_en_pwm=RIGHT_EN_PWM,
        right_in3=RIGHT_IN3,
        right_in4=RIGHT_IN4,
        pwm_frequency_hz=PWM_FREQUENCY_HZ,
        invert_left=INVERT_LEFT,
        invert_right=INVERT_RIGHT,
    )

    pose = build_pose()

    # smoothing state
    left_cmd = 0.0
    right_cmd = 0.0

    no_person_frames = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                no_person_frames += 1
                motors.stop()
                time.sleep(0.05)
                continue

            if FLIP_FRAME:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe Tasks expects mp.Image; but pose.detect also accepts mp.Image.
            from mediapipe import Image, ImageFormat

            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            landmarks = None
            person = PersonState(detected=False, cx_norm=None, distance_m=None)

            try:
                landmarks = detect_person_pose(pose, mp_image)
            except Exception:
                landmarks = None

            if landmarks is not None and len(landmarks) >= 25:
                cx_norm = estimate_center_x_norm(landmarks)
                dist_m = estimate_distance_m(landmarks, w)
                person = PersonState(True, cx_norm, dist_m)
                no_person_frames = 0
            else:
                no_person_frames += 1

            # Safety stop if lost
            if no_person_frames >= NO_PERSON_FRAMES_TO_STOP:
                left_cmd = lerp(left_cmd, 0.0, 0.5)
                right_cmd = lerp(right_cmd, 0.0, 0.5)
                motors.set_speeds(left_cmd, right_cmd)
                cv2.putText(
                    frame,
                    "NO PERSON -> STOP",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
            else:
                # Control
                # Turn: center person
                turn = 0.0
                if person.cx_norm is not None:
                    error_x = person.cx_norm - 0.5  # -0.5..+0.5
                    turn = K_TURN * error_x * 2.0   # scale to -1..+1 range
                    turn = clamp(turn, -MAX_TURN, MAX_TURN)

                # Forward: keep distance
                forward = 0.0
                if person.distance_m is not None and 0.1 < person.distance_m < MAX_DISTANCE_M:
                    # Too close: stop (or you can reverse slowly)
                    if person.distance_m < STOP_DISTANCE_M:
                        forward = 0.0
                    else:
                        error_d = person.distance_m - TARGET_DISTANCE_M
                        forward = K_FWD * error_d

                    forward = clamp(forward, -MAX_REV, MAX_FWD)

                # Mix to differential drive
                target_left = clamp(forward - turn, -1.0, 1.0)
                target_right = clamp(forward + turn, -1.0, 1.0)

                # Smooth
                left_cmd = lerp(left_cmd, target_left, 1.0 - SMOOTHING)
                right_cmd = lerp(right_cmd, target_right, 1.0 - SMOOTHING)

                motors.set_speeds(left_cmd, right_cmd)

                # UI overlay
                status = "FOLLOW" if person.detected else "SEARCH"
                cv2.putText(
                    frame,
                    f"{status}  L={left_cmd:+.2f} R={right_cmd:+.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                if person.cx_norm is not None:
                    cx_px = int(person.cx_norm * w)
                    cv2.line(frame, (cx_px, 0), (cx_px, h), (255, 0, 0), 2)
                cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)

                if person.distance_m is not None:
                    cv2.putText(
                        frame,
                        f"Dist: {person.distance_m:.2f}m (target {TARGET_DISTANCE_M:.2f}m)",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            # FPS
            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Person Follow (Pose)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    finally:
        print("\n✓ Closing...")
        try:
            motors.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            pose.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
