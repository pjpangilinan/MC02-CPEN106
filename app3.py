import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import pages as pg

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide", page_icon="lawatch.svg", initial_sidebar_state="collapsed")

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

st.markdown(f"""
<style>
.navbar {{
    background-color: #406606;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 9999;
}}
.navbar-left {{
    display: flex;
    align-items: center;
}}
.navbar-left img {{
    height: 45px;
    margin-right: 15px;
}}
.navbar a {{
    color: white;
    text-decoration: none;
    font-size: 18px;
    font-weight: bold;
    margin-left: 30px;
}}
.navbar a.active {{
    color: #f8cc63;
    border-bottom: 3px solid #f8cc63;
}}
.navbar a:hover {{
    color: #f8cc63;
}}
</style>

<div class="navbar">
    <div class="navbar-left">
        <img src="https://i.ibb.co/Mg7hysP/lawatch-logo.png" alt="LaWatch">
        <span style="font-size: 22px; color: white; font-weight: bold;">LaWatch</span>
    </div>
    <div>
        <a href="/?page=Dashboard" class="{ 'active' if st.session_state.page == 'Dashboard' else '' }">Dashboard</a>
        <a href="/?page=Recommendations" class="{ 'active' if st.session_state.page == 'Recommendations' else '' }">Recommendations</a>
    </div>
</div>
<br><br><br><br><br>
""", unsafe_allow_html=True)


# Query param handler
page = st.query_params.get("page", st.session_state.page)
st.session_state.page = page

# Page routing
if st.session_state.page == "Dashboard":
    pg.show_dashboard()
elif st.session_state.page == "Recommendations":
    pg.show_rec()
else:
    pg.show_home()
