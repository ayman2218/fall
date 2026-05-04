#!/usr/bin/env python3
"""
Télécharger et installer Git portable
"""
import urllib.request
import os
import subprocess

url = "https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/PortableGit-2.47.0-64-bit.7z.exe"
dest = r"C:\PortableGit.exe"
extract_to = r"C:\PortableGit"

print("📥 Téléchargement de Git Portable...")
try:
    urllib.request.urlretrieve(url, dest)
    print(f"✓ Téléchargé: {dest}")
    
    print("\n📦 Extraction...")
    subprocess.run([dest, f"-o{extract_to}", "-y"], check=True)
    print(f"✓ Extrait dans: {extract_to}")
    
    print("\n✅ Git Portable prêt!")
    print(f"   {extract_to}\\bin\\git.exe")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
