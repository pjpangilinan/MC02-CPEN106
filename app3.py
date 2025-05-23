import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import pages as pg

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide", page_icon="lawatch.svg", initial_sidebar_state="collapsed")

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide", page_icon="lawatch.svg", initial_sidebar_state="collapsed")

page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Recommendations"])

functions = {
    "Home": pg.show_home,
    "Dashboard": pg.show_dashboard,
    "Recommendations": pg.show_rec,
}

go_to = functions.get(page)
if go_to:
    go_to()


go_to = functions.get(page)
if go_to:
    go_to()
