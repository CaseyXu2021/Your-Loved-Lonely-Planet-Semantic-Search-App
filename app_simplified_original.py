"""
Lonely Planet Intelligent Travel Guide_Group7
Semantic Search + LLM-powered travel recommendations.
Version 2.0
Group 7:
**Casey: Code Demo
**Jasper: Code Revision
**Maxine: Demo Vedio, Proposal Picture
**Paew: Streamlit UI, Proposal

"""

import streamlit as st
import numpy as np
import pickle
import os
import base64
from pathlib import Path
import re
import time
from typing import List, Optional, Dict, Tuple
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from image_carousel import image_carousel


# =============================================================================
# Page configuration and path setup
# =============================================================================

st.set_page_config(
    page_title="🌍 You Loved Lonely Planet: Intelligent Travel Guide",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = _BASE_DIR / "knowledge_bases"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Azure OpenAI configuration
# =============================================================================

AZURE_API_KEY = "83b7af53ecf54420b4f4c9efa1218016"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini"


# =============================================================================
# Embedding model loading (cached to avoid repeated loading)
# =============================================================================

@st.cache_resource
def get_embedding_model():
    """Load multilingual embedding model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =============================================================================
# Knowledge base and semantic search
# =============================================================================

def load_knowledge_base(country_name: str) -> Optional[Dict]:
    """Load knowledge base pickle for the specified country."""
    file_path = DATA_DIR / f"{country_name}.pkl"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None


def semantic_search(
    query: str,
    country_name: str,
    top_k: int = 5,
) -> Tuple[List[Dict], np.ndarray]:
    """Retrieve top-k most relevant text chunks via cosine similarity on embeddings."""
    kb = load_knowledge_base(country_name)
    if not kb:
        return [], np.array([])

    model = get_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True)

    nn = NearestNeighbors(n_neighbors=min(top_k, len(kb["chunks"])), metric="cosine")
    nn.fit(kb["embeddings"])

    distances, indices = nn.kneighbors(query_vec)
    similarities = 1 - distances[0]

    results = []
    for idx, score in zip(indices[0], similarities):
        chunk = kb["chunks"][idx].copy()
        chunk["similarity"] = float(score)
        chunk["relevance_pct"] = f"{score * 100:.1f}%"
        results.append(chunk)

    return results, similarities


# =============================================================================
# Keyword extraction from retrieved texts
# =============================================================================

def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """Extract top-n frequent keywords after filtering stop words."""
    combined_text = " ".join(texts).lower()

    stop_words = {
        "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "a", "an", "this", "that", "these", "those", "it", "its", "page",
        "lonely", "planet", "guide", "book", "edition", "chapter", "visit",
        "located", "km", "hours", "day", "days", "week", "weeks", "month",
    }

    words = re.findall(r"\b[a-zA-Z\u4e00-\u9fa5]{3,}\b", combined_text)
    word_freq = {}
    for word in words:
        if word not in stop_words and not word.isdigit():
            word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


# =============================================================================
# RAG prompt construction for LLM
# =============================================================================

def build_rag_prompt(query: str, contexts: List[Dict]) -> str:
    """Build the RAG prompt with reference chunks and answer format instructions."""
    context_text = "\n\n".join(
        [
            f"[Source: Page {c['page']}, Relevance: {c['relevance_pct']}]\n{c['text'][:1000]}"
            for c in contexts
        ]
    )

    prompt = f"""Based on the following excerpts from Lonely Planet travel guides, answer the user's question.

[Reference Document Content]
{context_text}

[User Question]
{query}

[Answer Requirements]
1. First list 3-5 core keywords (Key Highlights)
2. Then provide detailed travel advice with specific place names and practical suggestions, sue bulletpoint with emoji for better readability
3. Mark information sources with page numbers (Page X), use yellow to high the page number
4. If document lacks relevant information, state "Unable to determine based on current information"
5. Maintain a friendly and professional travel advisor tone

Please output in the following format:

**Keywords**: Keyword1, Keyword2, Keyword3...

**Detailed Suggestions**:
(Detailed content with page references like [Page 23])

**Extra Tips**:
(Optional practical suggestions)

**Recommended Questions**
(Optional recommended questions based on the user's question)"""


    return prompt


# =============================================================================
# Azure OpenAI API call for response generation
# =============================================================================

def generate_response(prompt: str, api_key: str = AZURE_API_KEY) -> Optional[str]:
    """Call Azure OpenAI chat completion API and return the assistant message."""
    try:
        if not api_key or api_key.strip() == "":
            return None

        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-10-21",
            azure_endpoint=AZURE_ENDPOINT,
        )

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional travel advisor familiar with Lonely Planet guides. "
                    "Provide concise, practical travel advice based on the given context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Authentication" in error_msg or "invalid" in error_msg.lower():
            return None
        return f"Error: {error_msg}"


# =============================================================================
# Sidebar: destinations list, options, and API config
# =============================================================================

def render_sidebar(available_countries: List[str], cover_picture_path: Path):
    """Render sidebar with destination covers, search options, and API settings."""
    st.markdown(
        """<style> section[data-testid="stSidebar"] .stImage img {
            width: 100% !important; max-width: 100% !important; object-fit: contain;
        } </style>""",
        unsafe_allow_html=True,
    )
    st.subheader("Available Destinations")
    for country in available_countries:
        with st.container():
            st.caption(f"**{country}**")
            cover_image_path = cover_picture_path / f"{country}.png"
            if cover_image_path.exists():
                st.image(str(cover_image_path), width="stretch")
            else:
                st.info(f"No cover for {country}")

    st.divider()
    st.subheader("Advanced Options")
    search_top_k = st.slider(
        "Retrieve top K segments",
        3,
        15,
        5,
        help="Number of context chunks to retrieve per query.",
    )
    use_llm = st.toggle(
        "Enable AI Recommendations",
        value=True,
        help="Use Azure OpenAI for response generation.",
    )

    st.subheader("Azure OpenAI Configuration")
    api_key_input = st.text_input(
        "API Key",
        value=AZURE_API_KEY,
        type="password",
        placeholder="Paste your Azure OpenAI API key",
        help="Primary or Secondary key from Azure.",
    )
    show_debug = st.toggle(
        "Show Debug Info",
        value=False,
        help="Display retrieval statistics.",
    )

    st.divider()
    st.info(
        "**Knowledge Base Source:** Lonely Planet Publications\n\n"
        "**AI Engine:** HKUST OpenAI Azure API"
    )

    return search_top_k, use_llm, api_key_input, show_debug


# =============================================================================
# Main content: destination selector and book picture viewer
# =============================================================================

def render_destination_panel(available_countries: List[str], cover_picture_path: Path):
    """Render left column with country selector and book picture gallery."""
    st.subheader("Step1. Choose Destination")
    selected_country = st.selectbox(
        "Select a country:",
        available_countries,
        index=0,
        label_visibility="collapsed",
    )

    if selected_country:
        kb = load_knowledge_base(selected_country)
        if kb:
            with st.container(border=True):
                col1, col2 = st.columns(2)
                col1.metric("Documents", kb["total_pages"])
                col2.metric("Segments", len(kb["chunks"]))

        image_carousel(selected_country, key="main_carousel")

    return selected_country


# =============================================================================
# Search results and AI response display
# =============================================================================

def render_search_results(
    query: str,
    selected_country: str,
    results: List[Dict],
    scores: np.ndarray,
    keywords: List[str],
    use_llm: bool,
    api_key_input: str,
    search_top_k: int,
    show_debug: bool,
):
    """Render search results, AI recommendations, and source documents."""
    st.divider()

    if use_llm:
        with st.spinner("Generating response..."):
            prompt = build_rag_prompt(query, results)
            answer = generate_response(prompt, api_key_input)

            if answer is None:
                st.info(
                    "**Azure OpenAI not configured.** Showing source documents.\n\n"
                    "Configure API Key in the sidebar to enable AI responses."
                )
            elif answer and not answer.startswith("Error"):
                st.subheader("AI-Powered Recommendations")
                st.markdown(answer)
            else:
                st.warning(answer)

    st.subheader("Source Documents")

    if len(results) > 1:
        tabs = st.tabs([f"Result {i + 1}" for i in range(len(results))])
        for tab, res in zip(tabs, results):
            with tab:
                col_rel1, col_rel2 = st.columns([3, 1])
                col_rel1.progress(res["similarity"], text=f"Relevance: {res['relevance_pct']}")
                col_rel2.caption(f"Page {res['page']}")
                st.markdown(f"**Excerpt:**\n{res['text']}")
                with st.expander("View full segment"):
                    st.write(res["text"])
                    st.caption(f"Chunk ID: {res['chunk_id']}")
    else:
        for res in results:
            with st.container(border=True):
                col_rel1, col_rel2 = st.columns([3, 1])
                col_rel1.progress(res["similarity"], text=f"Relevance: {res['relevance_pct']}")
                col_rel2.caption(f"Page {res['page']}")
                st.markdown(f"**Excerpt:**\n{res['text']}")
                with st.expander("View full segment"):
                    st.write(res["text"])
                    st.caption(f"Chunk ID: {res['chunk_id']}")

    if show_debug:
        st.divider()
        with st.expander("Debug Information"):
            debug_info = {
                "query": query,
                "country": selected_country,
                "retrieved_segments": len(results),
                "top_k_requested": search_top_k,
                "average_similarity": float(np.mean(scores)),
                "min_similarity": float(np.min(scores)),
                "max_similarity": float(np.max(scores)),
                "keywords_extracted": keywords,
                "llm_enabled": use_llm,
                "azure_api_configured": bool(api_key_input and len(api_key_input) > 10),
            }
            st.json(debug_info)


# =============================================================================
# Splash screen with video and tagline
# =============================================================================

def render_splash():
    """Show splash screen with video and tagline for 5 seconds."""
    animation_path = _BASE_DIR / "Animation rotate.webm"
    splash_html = (
        '<div style="display:flex; flex-direction:column; align-items:center; '
        'justify-content:center; text-align:center; padding:40px 20px;">'
    )
    if animation_path.exists():
        with open(animation_path, "rb") as f:
            vid_b64 = base64.b64encode(f.read()).decode()
        splash_html += (
            f'<video autoplay loop muted playsinline style="width:auto; height:auto; '
            f'max-width:100%; object-fit:contain; margin-bottom:24px;">'
            f'<source src="data:video/webm;base64,{vid_b64}" type="video/webm"></video>'
        )
    splash_html += """
        <div style="font-size: 22px; color: #ffffff; margin-bottom: 16px; font-weight: 500;">
            Jobs fill your pocket, but adventures fill your soul.
        </div>
        <div style="font-size: 14px; color: #ffffff;">
            Lonely Planet Travel Semantic Search App · Casey, Jasper, Maxine & Paew
        </div>
    </div>"""
    st.markdown(splash_html, unsafe_allow_html=True)
    time.sleep(5)
    st.session_state.splash_done = True
    st.rerun()


# =============================================================================
# Main app entry
# =============================================================================

def main():
    """Main application flow: sidebar, layout, search, and results."""
    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False

    if not st.session_state.splash_done:
        render_splash()
        return

    if not DATA_DIR.exists():
        st.error("No knowledge bases found.")
        st.stop()

    available_countries = sorted(
        [f.stem for f in DATA_DIR.iterdir() if f.suffix == ".pkl"]
    )
    if not available_countries:
        st.error("No knowledge bases found.")
        st.stop()

    cover_picture_path = _BASE_DIR / "cover_picture"

    with st.sidebar:
        search_top_k, use_llm, api_key_input, show_debug = render_sidebar(
            available_countries, cover_picture_path
        )

    st.title("Lonely Planet Intelligent Travel Guide")
    st.caption("Fast Semantic Search + AI-Powered Recommendations by Azure OpenAI")

    col1, col2 = st.columns([1, 3], gap="large")

    with col1:
        selected_country = render_destination_panel(
            available_countries, cover_picture_path
        )

    with col2:
        st.subheader("Step2. Type Your Travel Questions")
        query = st.text_area(
            "Ask about your destination:",
            placeholder=(
                "Examples:\n"
                "What are the best attractions?\n"
                "Local food recommendations?\n"
                "Best time to visit?\n"
                "Budget travel tips?"
            ),
            height=100,
            label_visibility="collapsed",
        )
        search_clicked = st.button("Step3. Search & Analyze", type="primary", width="stretch")

        if search_clicked and query and selected_country:
            with st.spinner("Searching knowledge base..."):
                results, scores = semantic_search(
                    query, selected_country, top_k=search_top_k
                )

                if not results:
                    st.error("No relevant content found.")
                else:
                    keywords = extract_keywords([r["text"] for r in results])
                    render_search_results(
                        query,
                        selected_country,
                        results,
                        scores,
                        keywords,
                        use_llm,
                        api_key_input,
                        search_top_k,
                        show_debug,
                    )

        elif search_clicked and not query:
            st.warning("Hey please enter a question:)")
        elif search_clicked and not selected_country:
            st.warning("Hey please select a destination:)")


if __name__ == "__main__":
    main()
