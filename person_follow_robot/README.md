# Person Follow Robot (Raspberry Pi 5 + L298N + Pose)

This folder adds **person-following** to your existing fall-detection repository **without changing your old scripts**.

## What it does
- Uses **MediaPipe Pose** to detect a person
- Estimates:
  - person horizontal center (to **turn** left/right)
  - distance using shoulder-width (to **move forward/back**) 
- Drives **DC motors via L298N** using GPIO PWM
- Safety: if no person is detected for a few frames, motors stop

## Wiring notes (important)
- Motors must be powered by a **separate battery / motor supply** (NOT the Pi 5V)
- Connect **GND of motor supply** to **GND of Raspberry Pi** (common ground)
- L298N ENA/ENB should be connected to GPIOs you can PWM

## Setup
1. Edit pins and settings:
   - `person_follow_robot/config.py`

2. Install dependencies:
   ```bash
   pip3 install opencv-python mediapipe
   ```
   (On Raspberry Pi you may prefer system packages / venv depending on your setup.)

3. Run:
   ```bash
   python3 person_follow_robot/follow_pose.py
   ```

## Tuning
Edit in `config.py`:
- `TARGET_DISTANCE_M` (example 0.8)
- `K_TURN`, `K_FWD`
- `MAX_FWD`, `MAX_TURN`
- `NO_PERSON_FRAMES_TO_STOP`

## Troubleshooting
- If robot goes backward when it should go forward:
  - swap IN1/IN2 for left motor (or set `INVERT_LEFT=True`)
  - swap IN3/IN4 for right motor (or set `INVERT_RIGHT=True`)
- If distance is unstable:
  - ensure the camera sees both shoulders
  - increase `STOP_DISTANCE_M` and reduce `MAX_FWD`
