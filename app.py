#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
from utils.faiss_retriever import FaissRetriever

# Load environment variables from .env file
load_dotenv()



# Set page configuration
st.set_page_config(
    page_title="Dialysis Care Search",
    page_icon="🔍", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern, beautiful DCC Search engine CSS
st.markdown("""
<style>
body, .stApp, .main, .block-container {
    background: #f5f7fa !important;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}
.main .block-container {
    max-width: 700px !important;
    margin: 0 auto !important;
    padding-top: 48px !important;
}
.logo {
    text-align: center;
    margin-bottom: 32px;
    padding: 0 20px;
}
.search-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 580px;
    margin: 0 auto 32px auto;
    padding: 32px;
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 4px 16px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(226, 232, 240, 0.8);
    visibility: visible !important;
    opacity: 1 !important;
    position: relative;
    backdrop-filter: blur(10px);
}

.search-input-wrapper {
    position: relative;
    width: 100%;
    margin-bottom: 20px;
}

.search-icon {
    position: absolute;
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 18px;
    color: #6b7280;
    z-index: 2;
}

.suggestions-container {
    width: 100%;
}

.suggestions-title {
    color: #6b7280;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 12px;
    text-align: center;
}

.suggestions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    width: 100%;
}

.suggestion-btn {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 14px;
    color: #374151;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.suggestion-btn:hover {
    background: #e5e8ee;
    border-color: #2a4d8f;
    color: #2a4d8f;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(42, 77, 143, 0.1);
}

.suggestion-btn:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(42, 77, 143, 0.1);
}
.stTextInput {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.stTextInput > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.stTextInput > div > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.stTextInput > div > div > input {
    width: 100% !important;
    height: 60px !important;
    border: 2px solid #e1e5e9 !important;
    border-radius: 20px !important;
    outline: none !important;
    padding: 0 24px 0 56px !important;
    font-size: 17px !important;
    color: #1a1a1a !important;
    background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    font-weight: 400 !important;
}

.stTextInput > div > div > input:focus {
    background: linear-gradient(145deg, #ffffff, #f0f4ff) !important;
    border: 2px solid #2a4d8f !important;
    box-shadow: 0 8px 32px rgba(42, 77, 143, 0.2), 0 4px 16px rgba(42, 77, 143, 0.1) !important;
    transform: translateY(-2px) !important;
}

.stTextInput > div > div > input:hover {
    border: 2px solid #cbd5e1 !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.12), 0 3px 12px rgba(0,0,0,0.06) !important;
    transform: translateY(-1px) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    opacity: 0.8 !important;
}

.stTextInput > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
}


.stTextInput > div > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
}

.results-container {
    max-width: 650px;
    margin: 0 auto;
    padding: 0 0px;
}
.results-stats {
    color: #7a7f87;
    font-size: 15px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e8ee;
}
.result {
    margin-bottom: 22px;
    max-width: 650px;
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 1px 8px 0 rgba(40, 60, 90, 0.06);
    border: 1px solid #e5e8ee;
    padding: 18px 22px 14px 22px;
    position: relative;
    overflow: hidden;
}
.result-url {
    margin-bottom: 2px;
}
.result-url-text {
    color: #4b6e4b;
    font-size: 13px;
    line-height: 1.3;
    word-break: break-all;
}
.result-title {
    color: #1a2a4d;
    font-size: 20px;
    line-height: 1.3;
    font-weight: 600;
    margin: 0 0 3px 0;
    text-decoration: none;
    display: block;
    transition: color 0.13s;
}
.result-title:hover, .result-title:focus {
    text-decoration: underline;
    color: #2a4d8f;
}
.result-title:visited {
    color: #6a3da8;
}
.result-snippet {
    color: #3d4156;
    font-size: 15px;
    line-height: 1.58;
    margin: 0 0 8px 0;
    word-wrap: break-word;
}
.result-meta {
    color: #8a8f97;
    font-size: 12px;
    padding-top: 4px;
}
.stSpinner > div {
    margin: 2rem auto !important;
}
.stApp > header {display: none;}
/* .stDeployButton {display: none;} */
#MainMenu {display: none;}
footer {display: none;}
.stException {display: none;}
/* div[data-testid="stToolbar"] {display: none;} */

/* Ensure deploy button is visible */
.stDeployButton {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 9999 !important;
}
/* Ensure search input is visible and prevent duplication */
.stTextInput, .stTextInput > div, .stTextInput > div > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 100% !important;
}

/* Hide any duplicate inputs */
.stTextInput:not(:first-of-type) {
    display: none !important;
}


@media (max-width: 600px) {
    .main .block-container {
        padding: 24px 2px !important;
    }
    .logo img {
        max-width: 180px !important;
    }

    .search-container {
        padding: 16px;
        margin: 0 auto 24px auto;
    }

    .stTextInput > div > div > input {
        height: 52px !important;
        font-size: 16px !important;
        padding: 0 20px 0 48px !important;
    }

    .search-icon {
        left: 14px;
        font-size: 16px;
    }

    .suggestions-grid {
        grid-template-columns: 1fr;
        gap: 6px;
    }

    .suggestion-btn {
        padding: 8px 12px;
        font-size: 13px;
    }

    .result-title {
        font-size: 16px;
    }
    .result {
        padding: 10px 6px 8px 6px;
    }
    .stButton > button {
    font-size: 14px !important;
    padding: 10px 16px !important;
    margin: 4px 2px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #374151 !important;
    transition: all 0.2s ease !important;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.stButton > button:hover {
    background: #1a2a4d !important;
    border-color: #1a2a4d !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(26, 42, 77, 0.2) !important;
}

}
</style>
""", unsafe_allow_html=True)

# DCC Logo (smaller, more subtle)
st.markdown('''
<div class="logo">
    <img src="https://eadn-wc01-6859330.nxedge.io/wp-content/uploads/2021/07/DCC-Logo-Clause_Rebranded_888404-600x227.png" 
         alt="Dialysis Care Center" 
         style="max-width: 180px; height: auto; margin-bottom: 10px;">
    <div style="font-size:18px; color:#1a2a4d; font-weight:600; margin-top:8px; letter-spacing:0.5px;">Dialysis Care Center Vector Search
</div>
''', unsafe_allow_html=True)

# Initialize embedding model and retriever
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
if 'faiss_retriever' not in st.session_state:
    st.session_state.faiss_retriever = FaissRetriever()

# Suggested search terms
suggested_searches = [
    "dialysis treatments",
    "kidney diet recipes", 
    "dialysis center locations",
    "peritoneal dialysis",
    "hemodialysis treatment",
    "kidney transplant information",
    "renal diet guidelines",
    "dialysis complications",
    "home dialysis options",
    "dialysis nutrition tips"
]

# Initialize search query in session state
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Handle suggestion clicks
if 'suggestion_clicked' not in st.session_state:
    st.session_state.suggestion_clicked = None



# Create a better search interface using Streamlit components
st.markdown('''
<div class="custom-search-container">

</div>
''', unsafe_allow_html=True)



search_query = st.text_input(
    label="Search for dialysis information",
    value=st.session_state.search_query,
    placeholder="Search dialysis treatments, locations, and care information...",
    key="main_search",
    label_visibility="collapsed",
    help="Enter your search query"
)

# Update session state when text input changes
if search_query != st.session_state.search_query:
    st.session_state.search_query = search_query

# Suggestions section
st.markdown('''
<div class="suggestions-container">
    <div class="suggestions-title">Popular searches:</div>
    <div class="suggestions-grid">
''', unsafe_allow_html=True)

# Create suggestion buttons in a grid layout
cols = st.columns(2)
for i, suggestion in enumerate(suggested_searches):
    col_idx = i % 2
    with cols[col_idx]:
        if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
            st.session_state.search_query = suggestion
            st.rerun()

st.markdown('''
    </div>
</div>
</div>
''', unsafe_allow_html=True)



# Suppress noisy logs from chromadb and other libraries
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)

# Helper to render results
def render_results(results, search_query):
    if results:
        st.markdown(f'''
        <div class="results-container">
            <div class="results-stats">
                About {len(results)} results for "{search_query}"
            </div>
        </div>
        ''', unsafe_allow_html=True)
        for result in results:
            title = result.get('title') or f"{result.get('category', '').replace('-', ' ').title()} Information"
            url = result.get('url')
            display_url = url.replace('https://', '').replace('http://', '') if url else ''
            if display_url.endswith('/'):
                display_url = display_url[:-1]
            snippet = result.get('content', '')[:180]
            if len(result.get('content', '')) > 180:
                last_space = snippet.rfind(' ')
                if last_space > 120:
                    snippet = snippet[:last_space] + '...'
                else:
                    snippet = snippet + '...'
            st.markdown(f'''
            <div class="result">
                <div class="result-url">
                    <span class="result-url-text">{display_url}</span>
                </div>
                <a class="result-title" href="{url if url else '#'}" target="_blank">{title}</a>
                <div class="result-snippet">{snippet}</div>
                <div class="result-meta">Distance: {result.get('distance', 0):.3f}</div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="results-container">
            <div class="results-stats">
                No results found for "{search_query}". Please try a different search.
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Search logic and results
if search_query:
    with st.spinner("Searching..."):
        try:
            query_embedding = st.session_state.embedding_model.encode([search_query])
            results = st.session_state.faiss_retriever.search(query_embedding, top_k=10)
            st.session_state.search_results = results
        except Exception as e:
            st.error(f"Search error: {e}")
            st.session_state.search_results = []
            import traceback
            st.session_state['last_traceback'] = traceback.format_exc()
else:
    st.session_state.search_results = []

# Display results
if st.session_state.get('search_results'):
    render_results(st.session_state['search_results'], search_query)

# Optionally, for debugging, you can show the traceback in the UI (commented out by default):
# if 'last_traceback' in st.session_state:
#     st.expander("Show error details").write(st.session_state['last_traceback'])

# Hide chat input and history for pure search engine mode
# (If you want to keep chat, move it to a separate tab or page)

# Footer
st.markdown('''
<div style="
    text-align: center;
    padding: 40px 20px 20px 20px;
    margin-top: 60px;
    border-top: 1px solid #e5e8ee;
    background: #f8fafc;
    font-size: 14px;
    color: #6b7280;
">
    Made by <a href="https://olisemeka.dev" target="_blank" style="
        color: #2a4d8f;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    " onmouseover="this.style.color='#1a365d'" onmouseout="this.style.color='#2a4d8f'">olisemeka</a>
</div>
''', unsafe_allow_html=True)
