# Rapport — Détection totale de chute (IA + Physique + Yeux) — état au 14/04/2026

## Objectif
Ouvrir la webcam, détecter si une personne est présente, et afficher un état en temps réel :
- **NO PERSON** : aucune personne/pose détectée
- **DETECTED NORMAL** : personne détectée, pas de chute
- **DETECTED FALL** : chute détectée

La détection de chute se fait via un modèle entraîné (classification binaire) sur une **séquence de poses**.

---

## Où on est arrivé (fonctionnel)
- La démo GCN ouvre la webcam (`--camera 0` par défaut) et affiche l’état en overlay.
- Le script calcule aussi `fps` (images traitées par seconde).
- Détection hyper-stable combinée IA + Physique via :
  - L'analyse du plongeon direct (`dp > 0.15`).
  - L'analyse d'absence de mouvement après la chute (`var < 0.002`).
  - L'analyse de l'état des yeux (fermeture bilatérale → perte de connaissance) grâce au Face Mesh, affichant en temps réel le statut Oculaire Rouge/Vert.
- Nouveaux dossiers d'entraînement extraits et intégrés (1378, 1260, 1843).
- Compatibilité MediaPipe :
  - Utilise **MediaPipe Tasks (PoseLandmarker)** si disponible.
  - Mixé avec les réseaux **mp.solutions.face_mesh** (EAR) et **mp.solutions.pose**.

---

## Comment ça détecte “FALL” (résumé)
1. Pour chaque frame, MediaPipe extrait 33 landmarks (pose) et on récupère des features par landmark : `x, y, z, v`.
2. Parallèlement, **MediaPipe Face Mesh** analyse les paupières pour mesurer le ratio EAR (Eye Aspect Ratio). On regarde si les deux yeux passent sous 0.20 (yeux fermés).
3. On remplit un buffer des `seq_len` dernières frames (ex: 30) → tenseur `(T, V, C)`.
4. On calcule le score de "Chute de la tête" (`dp`, différence Y du nez) et le score d'immobilité au sol (`var`, variance sur 5 dernières frames).
5. On passe la séquence dans un modèle **ST-GCN** (PyTorch) qui retourne un score de base (logit).
6. `sigmoid(logit)` donne une probabilité `p` de posture anormale.
7. **Validation Heuristique & Finale** :
   - Si `p >= threshold`, on valide la chute SEULEMENT SI : la tête a chuté brusquement (`dp > 0.15`), OU la personne ne bouge plus (`var < 0.002`).
   - **CONFIDENCE 100%** : Si une posture délicate est identifiée ET que les DEUX YEUL sont FERMÉS (évanouissement), l'alerte maximale (score +2.0) est déclenchée sans délai.

---

## Commandes utiles
### Lancer la démo webcam (GCN)
Terminal :
- `python realtime_fall_demo_gcn.py --camera 0`

Jupyter (cellule) :
- `!python realtime_fall_demo_gcn.py --camera 0`

### Réduire les faux positifs
- `python realtime_fall_demo_gcn.py --camera 0 --threshold 0.75 --min-consecutive-fall 3 --hold-seconds 2`

### Si la caméra ne s’ouvre pas
- Essayer `--camera 1` ou `--camera 2`
- Fermer les applis qui bloquent la webcam (Teams/Zoom/WhatsApp Web, etc.)

---

## Fichiers utilisés et leur rôle
### Démo temps réel
- `realtime_fall_demo_gcn.py`
  - Démo principale : webcam → pose → buffer séquence → modèle GCN → overlay (NO PERSON / DETECTED NORMAL / DETECTED FALL)
  - Charge :
    - un checkpoint PyTorch `.pt` (modèle entraîné)
    - un `.npz` de normalisation (mu/sigma + seq_len, channels)
  - Télécharge automatiquement `models/pose_landmarker_lite.task` si manquant (quand backend Tasks est utilisé)

- `realtime_fall_demo.py`
  - Démo alternative (LSTM Keras/TensorFlow) : webcam → pose → séquence → modèle LSTM
  - Utile si tu veux comparer GCN vs LSTM

### Entraînement / préparation
- `prepare_gcn_sequences.py`
  - Construit des fenêtres temporelles `(T,V,C)` à partir d’un CSV de keypoints
  - Fait le split train/test
  - Calcule `mu` et `sigma` par channel
  - Sauvegarde dans `gcn_sequences.npz`

- `train_gcn_fall_detector.py`
  - Définit le graphe du squelette (connexions entre landmarks) via `MEDIAPIPE_POSE_EDGES`
  - Construit le modèle ST-GCN (PyTorch)
  - Entraîne et sauvegarde un checkpoint `.pt`

- `extract_keypoints_to_csv.py`
  - Extrait les keypoints/landmarks depuis tes images/vidéos et génère un CSV (source des scripts de préparation)

- `train_lstm_fall_detector.py`
  - Entraîne le modèle LSTM (TensorFlow/Keras) utilisé par `realtime_fall_demo.py`

### Modèles
- `models/pose_landmarker_lite.task`
  - Modèle MediaPipe PoseLandmarker utilisé par `realtime_fall_demo_gcn.py` quand le backend Tasks fonctionne

---

## Pourquoi ça peut détecter “droite” mais pas “dos/ventre/gauche”
- Le modèle généralise seulement sur ce qu’il a vu pendant l’entraînement.
- Si les chutes “dos/ventre/gauche” sont peu présentes (ou caméra/angle différent), la précision baisse.

Pistes fortes d’amélioration :
- Ajouter des exemples de chutes dans **toutes** les directions (droite/gauche/dos/ventre)
- Ajouter un **flip horizontal** (miroir) pour rendre gauche/droite équivalents
- Stabiliser la scène (corps entier visible, lumière, caméra fixe)

---

## Dépendances (indicatif)
- `opencv-python` (cv2)
- `numpy`
- `mediapipe`
- `torch` (pour GCN)
- `tensorflow` (pour la démo LSTM uniquement)

---

## Prochaine étape recommandée
1. Tester la caméra avec le tout dernier fichier `realtime_fall_demo_gcn.py` et la vérifier sur de *vraies chutes de profil/devant/derrière* via ton corps.
2. Évaluer si le seuil EAR (Fermeture des yeux = `0.20`), le seuil de descente de tête (`dp > 0.15`) et d’immobilité (`var < 0.002`) sont bien adaptés à l’éclairage de la caméra en situation réelle.
3. Continuer d’ajouter et d’étiqueter de nouveaux dossiers physiques pour ré-enrichir le modèle si les postures ventre/dos nécessitent d’être revues.
