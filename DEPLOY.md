# Publish to GitHub and Deploy (Share with Users)

## Part 1: Publish to GitHub

### 1.1 Install Git (if not installed)
- Download: https://git-scm.com/download/win
- Install with default options.

### 1.2 Create a GitHub account
- Go to https://github.com and sign up.

### 1.3 Create a new repository on GitHub
1. Log in to GitHub → click **"+"** (top right) → **New repository**.
2. **Repository name**: e.g. `lonely-planet-travel-app`
3. **Public**.
4. Do **not** check "Add a README" (you already have files).
5. Click **Create repository**.

### 1.4 Push your project from your computer
Open **PowerShell** or **Command Prompt**, then:

```powershell
cd "C:\Users\Casey\Desktop\Lonely Planet Semantic Search App_Group7"

# Initialize Git (only first time)
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Lonely Planet Travel App"

# Add GitHub as remote (replace YOUR_USERNAME and YOUR_REPO with your GitHub username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub (main branch)
git branch -M main
git push -u origin main
```

When prompted for password, use a **Personal Access Token** (not your GitHub password):
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token.
- Enable scope: **repo**.
- Copy the token and paste it when Git asks for password.

---

## Part 2: Deploy on Streamlit Community Cloud (free, public URL)

### 2.1 Sign up
- Go to https://share.streamlit.io
- Click **Sign up with GitHub** and authorize.

### 2.2 Deploy the app
1. Click **New app**.
2. **Repository**: select `YOUR_USERNAME/lonely-planet-travel-app` (or your repo name).
3. **Branch**: `main`.
4. **Main file path**: `app_simplified_original.py`
5. **App URL**: optional subdomain, e.g. `lonely-planet-travel-app`.
6. Click **Deploy**.

Wait a few minutes. When done, you get a link like:
`https://lonely-planet-travel-app-xxxx.streamlit.app`

### 2.3 Share with users
- Send this link to anyone. They open it in a browser, no install needed.
- Note: The app needs **knowledge_bases** (`.pkl` files) and **book_picture**, **cover_picture** folders in the repo for full functionality. If you don’t push large data, add a small sample or document in README where to place files.

---

## Part 3: Optional – Keep repo clean (.gitignore)

Create a file named `.gitignore` in your project folder with:

```
# Python
__pycache__/
*.py[cod]
*.pkl
.env
.venv/
venv/

# Optional: ignore large data if you don't want to push to GitHub
# knowledge_bases/
# book_picture/
# cover_picture/
```

If you **do** want the app to work on Streamlit Cloud without extra setup, **do not** ignore `knowledge_bases/`, `book_picture/`, `cover_picture/` (or remove those lines from `.gitignore`). If the files are too large, use Git LFS or document how to add data after clone.

---

## Quick checklist

| Step | Action |
|------|--------|
| 1 | Create GitHub repo (public) |
| 2 | `git init` → `git add .` → `git commit` → `git remote add origin` → `git push` |
| 3 | Go to share.streamlit.io → New app → connect repo → set main file to `app_simplified_original.py` → Deploy |
| 4 | Copy the `.streamlit.app` URL and share with users |
