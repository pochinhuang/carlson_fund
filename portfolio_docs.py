# pages/portfolio_docs.py
import streamlit as st
from pathlib import Path

st.title("📚 Portfolio Documentation")

# read md file
markdown_text = Path("portfolio_docs.md").read_text(encoding="utf-8")

# show md content
st.markdown(markdown_text, unsafe_allow_html=True)