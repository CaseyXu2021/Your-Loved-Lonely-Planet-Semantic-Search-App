# Lonely Planet Intelligent Travel Guide

Semantic search over Lonely Planet guides with AI-powered recommendations (Azure OpenAI).

**Group 7:** Casey · Jasper · Maxine · Paew

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_simplified_original.py
```

Open the URL shown in the terminal (e.g. http://localhost:8501).

---

## Project layout

| Path | Description |
|------|-------------|
| `app_simplified_original.py` | Main Streamlit app |
| `image_carousel.py` | Destination image display |
| `knowledge_bases/` | Pickle files (embeddings + chunks) per destination |
| `book_picture/<destination>/` | Images per destination |
| `cover_picture/` | Cover images for sidebar |

---

## Deploy

See [DEPLOY.md](DEPLOY.md) for GitHub and Streamlit Cloud deployment steps.
