# pages/portfolio_docs.py
import streamlit as st
from pathlib import Path

# st.title("Portfolio Documentation")

st.set_page_config(
    page_title = "Documentation",
    page_icon = ":material/article:",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0c0f;
    --card: #141820;
    --border: #1e2530;
    --text: #e8edf5;
    --muted: #7f8b99;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

div[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #090b0e 0%, #0d1117 100%);
}

div[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stAppToolbar"],
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapsedControl"],
header {
    background: rgba(0,0,0,0) !important;
}

button[kind="header"] {
    background: transparent !important;
    border: none !important;
}

div[data-testid="stMarkdownContainer"] {
    color: var(--text);
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: var(--text);
}

div[data-testid="stMarkdownContainer"] code {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #4a9eff !important;
    background-color: rgba(74, 158, 255, 0.12) !important;
    border: 1px solid rgba(74, 158, 255, 0.22) !important;
    border-radius: 4px !important;
    padding: 0.12rem 0.38rem !important;
}
</style>
""", unsafe_allow_html=True)

# read md file
markdown_text = Path("docs/portfolio_docs.md").read_text(encoding="utf-8")

# show md content
st.markdown(markdown_text, unsafe_allow_html=True)