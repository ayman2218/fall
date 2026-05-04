#!/usr/bin/env python3
"""
Script pour pousser vers GitHub sans Git CLI - utilise GitPython
"""
import os
import sys

print("📦 Fall Detection - GitHub Push avec GitPython\n")

# Installer/importer GitPython
try:
    from git import Repo
    print("✓ GitPython déjà installé\n")
except ImportError:
    print("Installation de GitPython...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'GitPython', '-q'])
    from git import Repo
    print("✓ GitPython installé\n")

# Configuration
repo_dir = r"C:\Users\asus\fall_github_simple"
repo_url = "https://github.com/ayman2218/fall.git"

print(f"Dossier: {repo_dir}")
print(f"Repo: {repo_url}\n")

os.chdir(repo_dir)

try:
    print("1️⃣ Initialisation du repository...")
    repo = Repo.init(repo_dir)
    print("   ✓ Initialisation OK\n")
    
    print("2️⃣ Configuration de l'utilisateur...")
    with repo.config_writer() as git_config:
        git_config.set_value("user", "name", "Ayman").release()
        git_config.set_value("user", "email", "ayman@example.com").release()
    print("   ✓ Config utilisateur OK\n")
    
    print("3️⃣ Ajout de tous les fichiers...")
    repo.index.add('*')
    print("   ✓ Fichiers ajoutés\n")
    
    print("4️⃣ Création du commit...")
    repo.index.commit("Add Fall Detection System with Robot Integration\n\n- MediaPipe Pose Detection\n- Real-time fall detection\n- Distance calculation\n- Robot alert system (< 1m)")
    print("   ✓ Commit créé\n")
    
    print("5️⃣ Ajout du remote origin...")
    # Supprimer si existe déjà
    try:
        repo.remote('origin').delete(repo)
    except:
        pass
    origin = repo.create_remote('origin', repo_url)
    print("   ✓ Remote ajouté\n")
    
    print("6️⃣ Renommage de la branche en 'main'...")
    repo.heads.master.checkout()
    repo.heads.master.rename('main')
    print("   ✓ Branche renommée en 'main'\n")
    
    print("7️⃣ Push vers GitHub...")
    print("   ⚠ Vous serez invité à saisir vos identifiants\n")
    origin.push(refspec='main:main', set_upstream=True)
    print("   ✓ Push réussi!\n")
    
    print("="*60)
    print("✅ SUCCÈS! Code poussé vers GitHub!")
    print("="*60)
    print(f"\nVérifiez sur: {repo_url}")
    print("\n")
    
except Exception as e:
    print(f"\n❌ Erreur: {e}\n")
    
    # Essayer une approche alternative si l'erreur persiste
    if "credentials" in str(e).lower() or "authentication" in str(e).lower():
        print("💡 ASTUCE: Utiliser un Personal Access Token")
        print("   1. Allez à: https://github.com/settings/tokens/new")
        print("   2. Cochez 'repo' et 'write:packages'")
        print("   3. Générez le token")
        print("   4. Utilisez-le comme mot de passe\n")
    
    sys.exit(1)
