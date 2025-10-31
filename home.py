import streamlit as st

st.set_page_config(
    page_title = "Home",
    page_icon = "📊",
    layout = "wide",         
    initial_sidebar_state = "expanded" 
)

###########
# sidebar #
###########
with st.sidebar:
    st.image("carlson_funds_enterprise_logo.jpeg", width=150)
    st.markdown("_Real funds, real students, real results._")


    st.write("`Created by:`")
    # my linkedin
    linkedin_url_brian = "https://www.linkedin.com/in/brian-huang-239884191"
    st.markdown(f'<a href="{linkedin_url_brian}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Brian Huang`</a>', unsafe_allow_html=True)
    # ioannis
    # linkedin_url_ioannis = "https://www.linkedin.com/in/ioannis-petropoulos"
    # st.markdown(f'<a href="{linkedin_url_ioannis}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Ioannis Petropoulos`</a>', unsafe_allow_html=True)

pages = {  
    "📊 Portfolio Dashboard": [
        st.Page("test.py", title = "Dashboard")
    ],
    "⚠️ Risk Management": [
        st.Page("rm.py", title = "Risk Analytics"),
        st.Page("portfolio_docs.py", title = "Documentation"),
        # st.Page("pages/rm_docs.py", title="Documentation"),
    ],
    "📈 Factor Analysis": [
        st.Page("fac.py", title="Factor Model"),
        # st.Page("pages/fac_docs.py", title="Documentation"),
    ],
}

pg = st.navigation(pages)
pg.run()