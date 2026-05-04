# Fall Detection System with Robot Integration

Une système de détection de chutes en temps réel utilisant MediaPipe avec intégration robotique pour les personnes à risque.

## Fonctionnalités

- **Detection en temps reel**: Capture vidéo webcam avec analyse de pose
- **MediaPipe Pose**: Détection de 33 points de pose corporelle
- **Detection de chutes**: Algorithme basé sur la position du nez vs. épaules
- **Calcul de distance**: Calcule la distance caméra-personne en mètres
- **Alerte robotique**: Déclenche l'alerte si distance < 1m ET chute détectée
- **Feedback visuel**: Code couleur (vert/orange/rouge) et bordure clignotante

## Installation

```bash
pip install opencv-python mediapipe numpy torch torchvision tensorflow
```

## Utilisation

```bash
python fall_detector_v2.py
```

### Contrôles
- **Q**: Quitter
- **ESC**: Quitter
- **SPACE**: Pause

## Architecture

```
fall_detection/
├── fall_detector_v2.py          # Production script principal
├── fall_detector.py             # Version alternative
├── realtime_fall_demo_gcn.py   # Demo avec GCN
├── train_gcn_fall_detector.py  # Entraînement du modèle
├── models/
│   ├── gcn_fall_model.pt       # Modèle PyTorch
│   ├── gcn_sequences.npz       # Données normalisées
│   └── pose_landmarker_lite.task  # Modèle MediaPipe
└── datasets/
    └── [données d'entraînement]
```

## Caractéristiques Techniques

### Detection de chutes
```
- Seuil: nose.y > shoulder_y + 0.30
- Confirmation: 3+ images dans 10-frame buffer
- Latence: ~50ms/frame à 30fps
```

### Distance (Focal length-based)
```
distance_meters = (0.4m * focal_length_pixels) / shoulder_distance_pixels
où focal_length = (width/2) / tan(FOV/2)
```

### Alerte robotique
```
trigger = distance < 1.0m AND (is_falling OR fall_confirmed)
```

## Caméras supportées

- Webcam C170 OK
- WebCam 2x OK
- USB2.0 HD UVC WebCam OK
- Support multi-index (auto-détection 2->1->0)

## Resolution des problèmes

### Caméra non détectée
- Essai automatique indices 0, 1, 2
- Backends: DirectShow (1280x720) ou MediaFoundation (640x480 @ 30fps)

### Performances basses
- Réduire la résolution
- Activer GPU CUDA pour MediaPipe
- Utiliser DirectShow au lieu de MediaFoundation

## Développeurs

Projet de détection de chutes avec intégration robotique pour assistance aux personnes âgées.

## Licence

MIT

## References

- MediaPipe: https://mediapipe.dev
- PyTorch: https://pytorch.org
- OpenCV: https://opencv.org
