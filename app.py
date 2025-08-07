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
    page_title="Semantic Search Engine",
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, modern CSS with improved spacing and design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        max-width: 800px;
        padding: 2rem 1rem;
        margin: 0 auto;
    }
    
    /* Header Section */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .logo-section {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a202c;
        margin: 1rem 0 0.5rem 0;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #4a5568;
        font-weight: 400;
        margin: 0;
    }
    
    /* Search Container */
    .search-section {
        background: white;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Search Input Styling */
    .stTextInput > div > div > input {
        height: 48px !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 0 16px !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        color: #1a202c !important;
        background: white !important;
        transition: border-color 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3182ce !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(49, 130, 206, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0 !important;
        font-weight: 400 !important;
    }
    
    /* Suggestions Section */
    .suggestions-container {
        margin-top: 1rem;
    }
    
    .suggestions-title {
        font-size: 0.875rem;
        color: #4a5568;
        font-weight: 500;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton > button {
        background: #3182ce !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
        height: auto !important;
        min-height: 36px !important;
    }
    
    .stButton > button:hover {
        background: #2c5aa0 !important;
    }
    
    /* Results Section */
    .results-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .results-stats {
        color: #4a5568;
        font-size: 0.875rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .result-item {
        background: #f7fafc;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        transition: border-color 0.2s ease;
    }
    
    .result-item:hover {
        border-color: #cbd5e0;
    }
    
    .result-url {
        color: #38a169;
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
        font-weight: 500;
    }
    
    .result-title {
        color: #1a202c;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        line-height: 1.4;
        text-decoration: none;
        display: block;
    }
    
    .result-title:hover {
        color: #3182ce;
        text-decoration: underline;
    }
    
    .result-snippet {
        color: #4a5568;
        font-size: 0.875rem;
        line-height: 1.5;
        margin-bottom: 0.25rem;
    }
    
    .result-meta {
        color: #a0aec0;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #3182ce !important;
    }
    
    /* Hide Streamlit Elements */
    .stApp > header,
    #MainMenu,
    footer,
    .stException {
        display: none;
    }
    
    /* Footer */
    .footer-section {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #4a5568;
        font-size: 0.875rem;
    }
    
    .footer-section a {
        color: #3182ce;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    
    .footer-section a:hover {
        color: #2c5aa0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        
        .logo-section,
        .search-section,
        .results-container {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .main-title {
            font-size: 1.5rem;
        }
        
        .stTextInput > div > div > input {
            height: 44px !important;
            font-size: 16px !important;
        }
        
        .result-item {
            padding: 0.75rem;
        }
        
        .stButton > button {
            font-size: 0.8rem !important;
            padding: 0.4rem 0.8rem !important;
            min-height: 32px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('''
<div class="header-container">
    <div class="logo-section">
        <div class="main-title">Semantic Search Engine</div>
        <div class="subtitle">AI-powered search for comprehensive information discovery</div>
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #718096;">
            <strong>This searches through the pages of </strong> <a href="https://dccdialysis.com/" target="_blank" style="color: #3182ce; text-decoration: none;">Dialysis Care Center.</a>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Initialize embedding model and retriever
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
if 'faiss_retriever' not in st.session_state:
    st.session_state.faiss_retriever = FaissRetriever()

# Suggested search terms
suggested_searches = [
    "treatments",
    "diet recipes", 
    "center locations",
    "medical procedures",
    "treatment options",
    "health information",
    "care guidelines",
    "complications",
    "home care options",
    "nutrition tips"
]

# Initialize search query in session state
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Handle suggestion clicks
if 'suggestion_clicked' not in st.session_state:
    st.session_state.suggestion_clicked = None



# Search Section
st.markdown('<div class="search-section">', unsafe_allow_html=True)

search_query = st.text_input(
    label="Search for information",
    value=st.session_state.search_query,
    placeholder="Search for treatments, locations, guidelines, and care information...",
    key="main_search",
    label_visibility="collapsed",
    help="Enter your search query to find relevant information"
)

# Update session state when text input changes
if search_query != st.session_state.search_query:
    st.session_state.search_query = search_query

# Suggestions section
st.markdown('''
<div class="suggestions-container">
    <div class="suggestions-title">Popular searches:</div>
</div>
''', unsafe_allow_html=True)

# Create suggestion buttons in a more organized grid
cols = st.columns(5)
for i, suggestion in enumerate(suggested_searches):
    col_idx = i % 5
    with cols[col_idx]:
        if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
            st.session_state.search_query = suggestion
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)



# Suppress noisy logs from chromadb and other libraries
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)

# Helper to render results with improved styling
def render_results(results, search_query, search_time):
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    if results:
        st.markdown(f'''
        <div class="results-stats">
            Found {len(results)} results for "{search_query}" in {search_time:.2f} seconds
        </div>
        ''', unsafe_allow_html=True)
        
        for result in results:
            title = result.get('title') or f"{result.get('category', '').replace('-', ' ').title()} Information"
            url = result.get('url')
            display_url = url.replace('https://', '').replace('http://', '') if url else ''
            if display_url.endswith('/'):
                display_url = display_url[:-1]
            
            snippet = result.get('content', '')[:200]
            if len(result.get('content', '')) > 200:
                last_space = snippet.rfind(' ')
                if last_space > 150:
                    snippet = snippet[:last_space] + '...'
                else:
                    snippet = snippet + '...'
            
            distance = result.get('distance', 0)
            relevance_score = max(0, 100 - (distance * 100))
            
            st.markdown(f'''
            <div class="result-item">
                <div class="result-url">{display_url}</div>
                <a class="result-title" href="{url if url else '#'}" target="_blank">{title}</a>
                <div class="result-snippet">{snippet}</div>
                <div class="result-meta">Relevance: {relevance_score:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="results-stats">
            No results found for "{search_query}" in {search_time:.2f} seconds. Try different keywords or check the spelling.
        </div>
        <div style="text-align: center; padding: 2rem; color: #4a5568;">
            <p>💡 <strong>Search tips:</strong></p>
            <p style="margin: 0.5rem 0;">• Try broader terms or synonyms</p>
            <p style="margin: 0.5rem 0;">• Use specific keywords</p>
            <p style="margin: 0.5rem 0;">• Check out the popular searches above</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Search logic and results
import time

if search_query:
    with st.spinner("Searching..."):
        start_time = time.time()
        try:
            query_embedding = st.session_state.embedding_model.encode([search_query])
            results = st.session_state.faiss_retriever.search(query_embedding, top_k=10)
            search_time = time.time() - start_time
            st.session_state.search_results = results
            st.session_state.search_time = search_time
        except Exception as e:
            search_time = time.time() - start_time
            st.error(f"Search error: {e}")
            st.session_state.search_results = []
            st.session_state.search_time = search_time
            import traceback
            st.session_state['last_traceback'] = traceback.format_exc()
else:
    st.session_state.search_results = []
    st.session_state.search_time = 0

# Display results
if st.session_state.get('search_results') is not None:
    search_time = st.session_state.get('search_time', 0)
    render_results(st.session_state['search_results'], search_query, search_time)

# Optionally, for debugging, you can show the traceback in the UI (commented out by default):
# if 'last_traceback' in st.session_state:
#     st.expander("Show error details").write(st.session_state['last_traceback'])

# Hide chat input and history for pure search engine mode
# (If you want to keep chat, move it to a separate tab or page)

# Footer
st.markdown('''
<div class="footer-section">
    <p>Built with ❤️ by <a href="https://olisemeka.dev" target="_blank">Olisemeka</a></p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7;">
        Powered by AI-driven vector search • Fork this project to create your own semantic search engine at <a href="https://github.com/OliseNS/vector_search" target="_blank">https://github.com/OliseNS/vector_search</a>
    </p>
</div>
''', unsafe_allow_html=True)
