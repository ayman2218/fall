#!/usr/bin/env python3
"""
Push vers GitHub avec dulwich (Git pur Python, pas besoin de CLI)
"""
import os
import sys

print("📦 Fall Detection - GitHub Push avec Dulwich\n")

# Installer dulwich
try:
    from dulwich.repo import Repo
    from dulwich.objects import Blob, Tree, Commit
    print("✓ Dulwich déjà installé\n")
except ImportError:
    print("Installation de Dulwich...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dulwich', '-q'])
    from dulwich.repo import Repo
    from dulwich.objects import Blob, Tree, Commit
    print("✓ Dulwich installé\n")

import time
import os
from datetime import datetime

repo_dir = r"C:\Users\asus\fall_github_simple"
repo_url = "https://github.com/ayman2218/fall.git"

print(f"Dossier: {repo_dir}")
print(f"Repo: {repo_url}\n")

os.chdir(repo_dir)

try:
    print("1️⃣ Initialisation du repository...")
    repo = Repo.init(repo_dir)
    print("   ✓ Initialisation OK\n")
    
    print("2️⃣ Ajout de tous les fichiers...")
    
    # Parcourir et ajouter les fichiers
    added_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Ignorer .git
        dirs[:] = [d for d in dirs if d != '.git']
        
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.npz', '.pt', '.task', '.gitignore')):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, repo_dir)
                try:
                    repo.stage([rel_path])
                    added_files.append(rel_path)
                    print(f"   + {rel_path}")
                except Exception as e:
                    print(f"   ⚠ {rel_path}: {e}")
    
    print(f"   ✓ {len(added_files)} fichiers ajoutés\n")
    
    print("3️⃣ Création du commit...")
    commit_message = b"Add Fall Detection System with Robot Integration\n\n- MediaPipe Pose Detection\n- Real-time fall detection\n- Distance calculation\n- Robot alert system (< 1m)"
    
    timestamp = int(time.time())
    author_time = b"Ayman <ayman@example.com> %d +0000" % timestamp
    
    commit = Commit()
    commit.tree = repo.index.commit(repo.object_store, committer=author_time, author=author_time, message=commit_message)
    
    print("   ✓ Commit créé\n")
    
    print("4️⃣ Configuration du remote...")
    config = repo.get_config()
    config.set((b'remote', b'origin'), b'url', repo_url.encode())
    config.release()
    print("   ✓ Remote 'origin' configuré\n")
    
    print("5️⃣ Préparation du push...")
    print("   ⚠ INFO: Dulwich ne supporte pas HTTPS directement")
    print("   → Utilisez plutôt une clé SSH ou Personal Access Token")
    print("   → Ou utilisez GitHub CLI/Web UI\n")
    
    print("✅ Repository prêt pour le push!\n")
    print("   Pour pousser avec SSH:")
    print("   1. Générez une clé SSH: ssh-keygen")
    print("   2. Ajoutez-la sur GitHub")
    print("   3. Utilisez: git@github.com:ayman2218/fall.git\n")
    
    print("   Ou utilisez la Web UI:")
    print(f"   {repo_url}\n")
    
except Exception as e:
    print(f"\n❌ Erreur: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
