# 🍓 Implémentation Fall Detection sur Raspberry Pi 5

## 📋 Prérequis Matériels

- **Raspberry Pi 5** (ou Pi 4)
- **Caméra USB** (compatible avec OpenCV)
- **Alimentation**: 27W min (Pi 5)
- **Stockage**: 32GB microSD (classe 10)
- **RAM**: 4GB+ recommandé
- **Connexion réseau**: Ethernet ou WiFi

---

## 🚀 Installation Étape par Étape

### 1️⃣ **Setup Raspberry Pi OS**

```bash
# Mettre à jour le système
sudo apt update
sudo apt upgrade -y

# Installer essentiels (Bullseye+)
sudo apt install -y python3-pip python3-dev git libatlas-base-dev
sudo apt install -y libwebp6 libtiff5 libharfbuzz0b libwebpmux3

# ⚠️ NOTE: libjasper-dev removed in Bullseye - not needed for fall detection
```

### 2️⃣ **Installer Python 3.11**

```bash
# Vérifier la version de Python
python3 --version

# Si < 3.11, installer
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Définir par défaut
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### 3️⃣ **Créer Virtual Environment**

```bash
cd /home/pi
python3.11 -m venv fall_env
source fall_env/bin/activate
```

### 4️⃣ **Installer Dépendances**

```bash
pip install --upgrade pip setuptools wheel

# Dépendances principales (ordre important)
pip install numpy
pip install opencv-python
pip install mediapipe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow
```

⚠️ **Note**: TensorFlow/PyTorch sur Pi5 peuvent être lents. **Alternative: Compiler localement ou utiliser EdgeTPU**

### 5️⃣ **Copier les Fichiers**

```bash
# Cloner le repository
git clone https://github.com/ayman2218/fall.git
cd fall

# Créer structure
mkdir -p models
cp fall_detector_v2.py .
```

---

## 📁 Fichiers Nécessaires

```
/home/pi/fall/
├── fall_detector_v2.py              (Script principal)
├── models/
│   ├── pose_landmarker_lite.task    (5.8 MB)
│   ├── gcn_fall_model.pt            (782 KB)
│   └── gcn_sequences.npz            (4.7 MB)
└── requirements.txt
```

### **requirements.txt** (Raspberry Pi optimisé)

```txt
opencv-python==4.8.0.74
mediapipe==0.10.3
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
tensorflow==2.13.0
```

---

## ⚙️ Configuration Raspberry Pi

### 🎥 **Configurer la Caméra USB**

```bash
# Vérifier si caméra détectée
ls -la /dev/video*

# Pour CSI Camera (si utilisée)
sudo raspi-config
# -> Interface Options -> Camera -> Enable
```

### 💾 **Optimisations Mémoire**

```bash
# Augmenter GPU memory (optionnel)
sudo nano /boot/firmware/config.txt
# Ajouter ou modifier:
gpu_mem=256

# Sauvegarder et rebooter
sudo reboot
```

### 🌡️ **Monitoring Température/CPU**

```bash
# Installer monitoring
sudo apt install -y iotop htop

# Vérifier avant exécution
htop

# Vérifier température
vcgencmd measure_temp
```

---

## 🔧 Adapter le Code pour Raspberry Pi

### **Version Optimisée: fall_detector_pi.py**

```python
#!/usr/bin/env python3
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
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS on Pi
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"✓ Camera: index={camera_config[0]}, res={RESOLUTION}")

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
except:
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
        cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
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
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
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
print("✓ Done!")
```

---

## 🚀 Exécution sur Raspberry Pi

### **Option 1: Exécution Directe**

```bash
# Activer l'environnement
cd /home/pi/fall
source fall_env/bin/activate

# Lancer le script
python3 fall_detector_pi.py
```

### **Option 2: Service Systemd (Auto-start)**

```bash
# Créer fichier service
sudo nano /etc/systemd/system/fall-detection.service
```

```ini
[Unit]
Description=Fall Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/fall
ExecStart=/home/pi/fall/fall_env/bin/python3 fall_detector_pi.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Activer le service
sudo systemctl enable fall-detection.service
sudo systemctl start fall-detection.service

# Vérifier le statut
sudo systemctl status fall-detection.service

# Voir les logs
sudo journalctl -u fall-detection.service -f
```

### **Option 3: Exécution Distante (SSH)**

```bash
# Depuis Windows/Mac
ssh pi@<IP_RASPBERRY_PI>
cd /home/pi/fall && source fall_env/bin/activate
python3 fall_detector_pi.py
```

---

## 📊 Optimisations Performance

| Optimisation | Effet | Mise en œuvre |
|---|---|---|
| Réduction résolution | -40% CPU | RESOLUTION = (320, 240) |
| Skip frames | -50% CPU | SKIP_FRAMES = 2 |
| Model lite | -35% Memory | model_complexity=0 |
| Buffer réduit | -20% Memory | maxlen=5 au lieu de 10 |
| FPS réduit | -30% CPU | cap.set(FPS, 15) |
| GPU Memory | +Memory | gpu_mem=256 |

---

## 🐛 Troubleshooting

### ❌ "Cannot open camera"
```bash
# Vérifier caméra
ls -la /dev/video*
v4l2-ctl --list-devices
```

### ❌ "ImportError: No module named 'mediapipe'"
```bash
source fall_env/bin/activate
pip install mediapipe --upgrade
```

### ❌ "Out of memory"
```bash
# Vérifier RAM
free -m

# Réduire buffer
fall_buffer = deque(maxlen=3)

# Augmenter swap (si nécessaire)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=512/' /etc/dphys-swapfile
sudo dphys-swapfile swapon
```

### ❌ "TensorFlow ImportError"
```bash
# Version légère pour Pi
pip uninstall tensorflow
pip install tensorflow-lite

# Ou utiliser PyTorch CPU uniquement
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 📡 Envoyer les Données (Optionnel)

### **Alertes via Email**

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(distance):
    sender = "your_email@gmail.com"
    password = "app_password"
    receiver = "alert@example.com"
    
    msg = MIMEText(f"FALL DETECTED at {distance:.2f}m!")
    msg['Subject'] = "🚨 Fall Alert"
    msg['From'] = sender
    msg['To'] = receiver
    
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender, password)
    server.sendmail(sender, receiver, msg.as_string())
    server.quit()

# Dans le code principal:
if is_newly_confirmed:
    send_alert(distance_meters)
```

### **Alertes via MQTT (IoT)**

```python
import paho.mqtt.client as mqtt

client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

# Publier alerte
def publish_alert(distance):
    client.publish("home/fall/alert", f"FALL at {distance:.2f}m")
```

---

## ✅ Checklist Final

- [ ] Raspberry Pi 5 allumé avec OS à jour
- [ ] Caméra USB branchée et détectée
- [ ] Python 3.11+ installé
- [ ] Virtual environment créé
- [ ] Tous les packages pip installés
- [ ] Fichiers modèles téléchargés
- [ ] fall_detector_pi.py exécuté avec succès
- [ ] Détections de chute affichées à l'écran
- [ ] Service systemd configuré (optionnel)
- [ ] Alertes testées (optionnel)

---

## 📞 Support

**Issues fréquentes?**
```bash
# Logs détaillés
python3 fall_detector_pi.py 2>&1 | tee fall_detection.log

# Diagnostics
df -h  # Espace disque
free -m  # Mémoire
ps aux | grep python  # Processus
```

---

**🎉 Vous êtes prêt à déployer sur Raspberry Pi 5!**
