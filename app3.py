import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar

import pages as pg

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide", page_icon="lawatch.svg", initial_sidebar_state="collapsed")

pages = ["Dashboard", "Recommendations", "Lol"]

styles = {
    "nav": {
        "background-color": "#3b6203",
        "height": "6rem",
        "display": "flex",
        "align-items": "center",
        "justify-content": "space-between",
        "margin-bottom": "0",
    },
    "img": {
        "height": "4.5rem",
    },
    "span": {
        "color": "white",
        "font-size": "1.2rem",
        "padding": "12px 20px",
        "font-weight": "bold",
    },
    "active": {
        "color": "#f8cc63",
        "text-decoration": "underline",
        "font-weight": "bold",
    }
}
page = st_navbar(
    pages,
    logo_path="lawatch.svg",
    styles=styles
)

functions = {
    "Home": pg.show_home,
    "Dashboard": pg.show_dashboard,
    "Recommendations": pg.show_rec,
}



go_to = functions.get(page)
if go_to:
    go_to()
