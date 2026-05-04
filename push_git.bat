@echo off
REM Installation et Push Git

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo  Fall Detection - Installation Git + Push
echo ================================================================================
echo.

REM Chercher git
where git >nul 2>&1
if !errorlevel! equ 0 (
    echo Git trouvé! Proceeding...
    goto push
)

echo Git non trouvé. Installation...
echo.

REM Essayer winget
winget install --id Git.Git --silent
if !errorlevel! equ 0 (
    echo Git installé via winget!
    goto push
)

REM Essayer chocolatey
choco install git -y
if !errorlevel! equ 0 (
    echo Git installé via chocolatey!
    goto push
)

echo.
echo ERREUR: Impossible d'installer Git automatiquement
echo.
echo Solutions manuelles:
echo 1. Installez Git: https://git-scm.com/download/win
echo 2. Utilisez GitHub Desktop: https://desktop.github.com/
echo 3. Utilisez la Web UI: https://github.com/ayman2218/fall.git
echo.
exit /b 1

:push
echo.
echo ================================================================================
echo  Commandes Git
echo ================================================================================
echo.

cd /d C:\Users\asus\fall_github_simple

echo 1. Git init...
git init

echo.
echo 2. Configuration utilisateur...
git config user.name "Ayman"
git config user.email "ayman@example.com"

echo.
echo 3. Ajout des fichiers...
git add .

echo.
echo 4. Commit...
git commit -m "Add Fall Detection System with Robot Integration"

echo.
echo 5. Ajout du remote...
git remote add origin https://github.com/ayman2218/fall.git

echo.
echo 6. Changement de branche en main...
git branch -M main

echo.
echo 7. Push vers GitHub...
echo Vous serez invité à saisir vos identifiants...
echo.
pause

git push -u origin main

if !errorlevel! equ 0 (
    echo.
    echo ================================================================================
    echo  SUCCESS!
    echo ================================================================================
    echo.
    echo Code poussé sur: https://github.com/ayman2218/fall.git
    echo.
) else (
    echo.
    echo ERREUR lors du push
    echo.
    echo Astuce: Utilisez un Personal Access Token
    echo https://github.com/settings/tokens/new
    echo.
)

pause
