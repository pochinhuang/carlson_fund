import streamlit as st

st.set_page_config(
    page_title = "Home",
    page_icon = ":material/finance:",
    layout = "wide",         
    initial_sidebar_state = "expanded" 
)

#0d1117

st.markdown("""
<style>
div[data-testid="stHeader"] {
    background: #000000 !important;
}
div[data-testid="stToolbar"] {
    background: #000000 !important;
}
div[data-testid="stAppToolbar"] {
    background: #000000 !important;
}
div[data-testid="collapsedControl"] {
    background: #000000 !important;
}
div[data-testid="stSidebarCollapsedControl"] {
    background: #000000 !important;
}
button[kind="header"] {
    background: transparent !important;
    border: none !important;
}
header {
    background: #000000 !important;
}

.sidebar-created-by {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 10px;
}

.sidebar-profile-link {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
    color: #f4f7fb !important;
    margin-bottom: 12px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 15px;
    font-weight: 600;
}

.sidebar-profile-link:hover {
    color: #00e5a0 !important;
}

.sidebar-profile-link img {
    width: 24px;
    height: 24px;
    border-radius: 4px;
}

.sidebar-school {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: #dfe7f1;
    margin-top: 6px;
}

.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.75;
    letter-spacing: 0.3px;
    color: #9fb0c2;
    margin-top: 12px;
}

.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 12px 0 10px 0;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

###########
# sidebar #
###########

with st.sidebar:
    st.image("imgs/carlson_funds_enterprise_logo.jpeg", width=150)
    st.markdown("_Real funds, real students, real results._")



with st.sidebar.container(border=True, height=180):
    linkedin_url_brian = "https://www.linkedin.com/in/brian-huang-239884191"
    st.markdown(
        f'''
        <a href="{linkedin_url_brian}" target="_blank" class="sidebar-profile-link">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
            <span>Brian Huang</span>
        </a>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-school">M.S. Financial Mathematics<br>University of Minnesota</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-school">Quantitative Analyst<br>David S. Kidwell Funds Enterprise</div>',
        unsafe_allow_html=True
    )

########################################################################################################


pages = {  
    "Portfolio Dashboard": [
        # st.Page("dashboard.py", title = "Dashboard", icon=":material/dashboard:")
        st.Page("dashboard.py", title = "Dashboard", icon=":material/dashboard:")
    ],
    "Risk Management": [
        st.Page("simulation.py", title = "Risk Analytics", icon=":material/assessment:"),
        st.Page("portfolio_docs.py", title = "Documentation", icon=":material/article:")
    ],
    "Factor Analysis": [
        st.Page("fac.py", title="Factor Model", icon=":material/insights:"),
        st.Page("fac_docs.py", title = "Documentation", icon=":material/article:")
    ],
}

pg = st.navigation(pages)
pg.run()
