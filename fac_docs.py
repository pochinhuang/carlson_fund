# pages/portfolio_docs.py
import streamlit as st
from pathlib import Path

# st.title("Factor Documentation")

st.set_page_config(
    page_title = "Documentation",
    page_icon = ":material/article:",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
)


# read md file
markdown_text = Path("docs/fac_docs.md").read_text(encoding="utf-8")

# show md content
st.markdown(markdown_text, unsafe_allow_html=True)