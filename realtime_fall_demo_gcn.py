import argparse
import collections
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def extract_pose_frame(result, *, n_joints: int, channels: tuple[str, ...]) -> np.ndarray:
    """Return per-frame tensor (V,C) in MediaPipe landmark order.

    If no pose detected, returns zeros.
    """
    c = len(channels)
    pose_landmarks = getattr(result, "pose_landmarks", None)
    if not pose_landmarks:
        return np.zeros((n_joints, c), dtype=np.float32)

    feats = np.zeros((n_joints, c), dtype=np.float32)
    # MediaPipe Tasks returns pose_landmarks as a list (one per detected person).
    # MediaPipe Solutions returns a landmark list at pose_landmarks.landmark.
    if isinstance(pose_landmarks, list):
        lms = pose_landmarks[0]  # first detected pose
    elif hasattr(pose_landmarks, "landmark"):
        lms = pose_landmarks.landmark
    else:
        lms = pose_landmarks

    for j in range(min(n_joints, len(lms))):
        lm = lms[j]
        for ci, ch in enumerate(channels):
            if ch == "x":
                feats[j, ci] = float(lm.x)
            elif ch == "y":
                feats[j, ci] = float(lm.y)
            elif ch == "z":
                feats[j, ci] = float(lm.z)
            elif ch == "v":
                # Prefer visibility if present; else fall back to presence.
                if hasattr(lm, "visibility") and lm.visibility is not None:
                    feats[j, ci] = float(lm.visibility)
                elif hasattr(lm, "presence") and lm.presence is not None:
                    feats[j, ci] = float(lm.presence)
                else:
                    feats[j, ci] = 0.0
            else:
                raise ValueError(f"Unsupported channel: {ch}")

    return feats


def pose_detected(result) -> bool:
    pose_landmarks = getattr(result, "pose_landmarks", None)
    if not pose_landmarks:
        return False
    try:
        if isinstance(pose_landmarks, list):
            return len(pose_landmarks) > 0 and len(pose_landmarks[0]) > 0
        if hasattr(pose_landmarks, "landmark"):
            return len(pose_landmarks.landmark) > 0
        return len(pose_landmarks) > 0
    except Exception:
        return True


def open_capture(*, video: str | None, camera_index: int) -> cv2.VideoCapture:
    if video:
        cap = cv2.VideoCapture(video)
        return cap

    # On Windows, some cameras work more reliably with explicit backends.
    backends: list[int]
    if os.name == "nt":
        backends = [
            getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            cv2.CAP_ANY,
        ]
    else:
        backends = [cv2.CAP_ANY]

    last_cap: cv2.VideoCapture | None = None
    for backend in backends:
        cap = cv2.VideoCapture(int(camera_index), int(backend))
        last_cap = cap
        if cap.isOpened():
            return cap

    return last_cap if last_cap is not None else cv2.VideoCapture(int(camera_index))


def resolve_existing_path(path: str | os.PathLike[str], *, candidates: list[str | os.PathLike[str]]) -> str:
    """Return first existing path among [path] + candidates, else original path as str."""
    p0 = Path(path)
    if p0.is_file():
        return str(p0)
    for c in candidates:
        pc = Path(c)
        if pc.is_file():
            return str(pc)
    return str(p0)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time fall demo: MediaPipe Pose -> ST-GCN (PyTorch)")
    p.add_argument(
        "--model",
        default=r"C:\Users\p\Pictures\data_organised_images_224\gcn_fall_model.pt",
        help="Trained PyTorch checkpoint (.pt) produced by train_gcn_fall_detector.py",
    )
    p.add_argument(
        "--norm-npz",
        default=r"C:\Users\p\Pictures\data_organised_images_224\gcn_sequences.npz",
        help="NPZ containing mu/sigma/seq_len/channels (produced by prepare_gcn_sequences.py)",
    )
    p.add_argument("--threshold", type=float, default=0.67)
    p.add_argument(
        "--min-consecutive-fall",
        type=int,
        default=1,
        help="Require this many consecutive frames above threshold to declare FALL (default: 1).",
    )
    p.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="After a fall is detected, keep displaying FALL for this many seconds (default: 0 = disabled).",
    )
    p.add_argument("--video", default=None, help="Path to a video file. If omitted, webcam is used.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument(
        "--pose-task-model",
        default=os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task"),
        help="Path to MediaPipe PoseLandmarker .task model file (downloaded automatically if missing).",
    )
    args = p.parse_args()

    import torch

    import mediapipe as mp

    from train_gcn_fall_detector import MEDIAPIPE_POSE_EDGES, build_adjacency, build_model, normalize_adjacency

    landmarker = None
    pose = None
    pose_backend = "unknown"

    script_dir = Path(__file__).resolve().parent

    norm_npz_path = resolve_existing_path(
        args.norm_npz,
        candidates=[
            script_dir / "gcn_sequences.npz",
            script_dir / "models" / "gcn_sequences.npz",
        ],
    )
    model_path = resolve_existing_path(
        args.model,
        candidates=[
            script_dir / "gcn_fall_model.pt",
            script_dir / "models" / "gcn_fall_model.pt",
        ],
    )

    data = np.load(norm_npz_path, allow_pickle=True)
    mu = data["mu"].astype(np.float32).reshape(-1)
    sigma = data["sigma"].astype(np.float32).reshape(-1)
    seq_len = int(np.asarray(data.get("seq_len", [30])).reshape(-1)[0])
    n_joints = int(np.asarray(data.get("n_joints", [33])).reshape(-1)[0])
    channels = tuple([str(x) for x in np.asarray(data.get("channels", ["x", "y", "z", "v"])).tolist()])

    if mu.shape[0] != len(channels) or sigma.shape[0] != len(channels):
        raise ValueError(f"mu/sigma size mismatch: mu={mu.shape} sigma={sigma.shape} channels={len(channels)}")

    edges = MEDIAPIPE_POSE_EDGES if n_joints == 33 else []
    A = build_adjacency(n_joints, edges)
    A_norm = normalize_adjacency(A)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_channels=len(channels), A_norm=A_norm).to(device)

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prefer MediaPipe Tasks PoseLandmarker when available; otherwise fall back to mp.solutions.pose.
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import RunningMode

        task_path = os.path.abspath(str(args.pose_task_model))
        if not os.path.isfile(task_path):
            os.makedirs(os.path.dirname(task_path), exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )
            print(f"Downloading PoseLandmarker model to: {task_path}", flush=True)
            print(f"URL: {url}", flush=True)
            urllib.request.urlretrieve(url, task_path)

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=RunningMode.VIDEO,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        pose_backend = "tasks"
        print("Pose backend: MediaPipe Tasks (PoseLandmarker)", flush=True)
    except Exception as e:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_backend = "solutions"
        print(f"Pose backend: mp.solutions.pose (fallback). Reason: {type(e).__name__}: {e}", flush=True)

    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        face_mesh = None
        print(f"Face Mesh error: {e}", flush=True)

    cap = open_capture(video=args.video, camera_index=int(args.camera))
    if args.video:
        src_desc = str(args.video)
    else:
        src_desc = f"camera {int(args.camera)}"

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {src_desc}. "
            "Try --camera 1/2, and close other apps using the camera (Zoom/Teams/WhatsApp browser tab)."
        )

    # Reduce latency where supported.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    buf: collections.deque[np.ndarray] = collections.deque(maxlen=int(seq_len))

    min_consecutive_fall = max(1, int(args.min_consecutive_fall))
    fall_streak = 0
    fall_hold_until = 0.0

    last_time = time.time()
    fps = 0.0

    print("Press Q or ESC to quit.", flush=True)

    consecutive_read_failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if args.video:
                    break
                if consecutive_read_failures >= 60:
                    raise RuntimeError(
                        "Camera opened but no frames were received. "
                        "Try a different --camera index, or ensure the camera isn't blocked by another app."
                    )
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- EYE BLINK DETECTION ---
            eyes_status = "YEUX: ?"
            if face_mesh is not None:
                mesh_res = face_mesh.process(rgb)
                if mesh_res.multi_face_landmarks:
                    lms = mesh_res.multi_face_landmarks[0].landmark
                    def dist(i1, i2):
                        return np.hypot(lms[i1].x - lms[i2].x, lms[i1].y - lms[i2].y)
                    # Right eye (MediaPipe indices from image's perspective)
                    ear_right = (dist(160, 144) + dist(158, 153)) / (2.0 * max(1e-6, dist(33, 133)))
                    # Left eye
                    ear_left = (dist(385, 380) + dist(387, 373)) / (2.0 * max(1e-6, dist(362, 263)))
                    
                    if ear_left < 0.20 and ear_right < 0.20:
                        eyes_status = "YEUX: FERMES"
                    else:
                        eyes_status = "YEUX: OUVERTS"
            # ------------------------------------

            if pose_backend == "tasks" and landmarker is not None:
                # Build MediaPipe Image and run pose detection
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                result = pose.process(rgb) if pose is not None else None
            person_ok = pose_detected(result)
            feats_vc = extract_pose_frame(result, n_joints=n_joints, channels=channels)
            buf.append(feats_vc)

            prob = None
            if len(buf) == buf.maxlen:
                x = np.stack(list(buf), axis=0).astype(np.float32)  # (T,V,C)
                x = (x - mu.reshape(1, 1, -1)) / sigma.reshape(1, 1, -1)
                x_t = torch.from_numpy(x[None, :, :, :]).to(device)  # (1,T,V,C)
                with torch.no_grad():
                    logit = model(x_t)
                    prob = float(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0])

            is_fall = False
            movement_var = 1.0
            head_drop = 0.0

            if prob is not None:
                # 1. Variance du mouvement sur les 5 dernières frames (pour vérifier si "gelé"/immobile au sol)
                if len(buf) >= 5:
                    recent = np.stack(list(buf)[-5:])  # (5, V, C)
                    # Variance moyenne des (x,y) sur 5 frames
                    movement_var = np.var(recent[:, :, :2], axis=0).mean()
                    
                    # 2. Chute de la tête : Nez Y_fin - Nez Y_début (positif = baisse vers le sol)
                    full_seq = np.stack(list(buf))     # (T, V, C)
                    head_drop = full_seq[-1, 0, 1] - full_seq[0, 0, 1]

                if prob >= float(args.threshold):
                    # CONFIRMATION : 
                    # Si chute "brusque de la tête" (>0.15) OU "immobilité" dans les 5 dernières frames (<0.001)
                    # ET SI les yeux sont fermés : confirmation absolue de la chute (100%)
                    if eyes_status == "YEUX: FERMES":
                        fall_streak += 2.0 # 100% certain, on déclenche l'alerte
                    elif head_drop > 0.15 or movement_var < 0.002:
                        fall_streak += 1.0
                    else:
                        # Moins confiant, on demande 1 frame de plus de streak
                        fall_streak += 0.5 
                else:
                    fall_streak = 0


                if fall_streak >= min_consecutive_fall:
                    is_fall = True
                    if float(args.hold_seconds) > 0:
                        fall_hold_until = max(fall_hold_until, time.time() + float(args.hold_seconds))

            if time.time() < fall_hold_until:
                is_fall = True

            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_time = now

            if not person_ok:
                label = "NO PERSON"
                text = f"{label}  fps={fps:.1f}"
            else:
                label = "WARMUP" if prob is None else ("FALL" if is_fall else "NORMAL")
                prefix = "DETECTED"
                extra_info = f" dp:{float(head_drop):.2f} var:{float(movement_var):.3f}" if prob is not None else ""
                text = (
                    f"{prefix} {label}  fps={fps:.1f}{extra_info}"
                    if prob is None
                    else f"{prefix} {label}  p={prob:.2f}  fps={fps:.1f}{extra_info}"
                )
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if label == "FALL" else ((0, 255, 0) if label != "NO PERSON" else (0, 255, 255)),
                2,
            )

            # Affichage de l'état des yeux sur une 2ème ligne
            if person_ok:
                eyes_color = (0, 0, 255) if eyes_status == "YEUX: FERMES" else (0, 255, 0)
                cv2.putText(
                    frame,
                    eyes_status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    eyes_color,
                    2,
                )

            cv2.imshow("Fall Detection (GCN)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        if landmarker is not None:
            landmarker.close()
        if pose is not None:
            pose.close()
        if face_mesh is not None:
            face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
