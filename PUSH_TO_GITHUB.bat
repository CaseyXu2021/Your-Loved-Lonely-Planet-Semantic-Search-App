@echo off
REM Edit the two lines below with your GitHub username and repo name, then run this file.
set GITHUB_USER=Caseyxu2021
set GITHUB_REPO=Your-Loved-Lonely-Planet-Semantic-Search-App

cd /d "%~dp0"
git remote remove origin 2>nul
git remote add origin https://github.com/%GITHUB_USER%/%GITHUB_REPO%.git
git push -u origin main
pause
