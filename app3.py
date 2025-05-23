import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar

import pages as pg

st.set_page_config(page_title="Taal Lake Water Quality Dashboard", layout="wide", page_icon="lawatch.svg", initial_sidebar_state="collapsed")

pages = ["Dashboard", "Recommendations"]

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
        "color": "#f8cc63",  # gold-ish active color
        "text-decoration": "underline",
        "font-weight": "bold",
    }
}
page = st_navbar(
    pages,
    logo_path="lawatch.svg",
    styles=styles,
)

functions = {
    "Home": pg.show_home,
    "Dashboard": pg.show_dashboard,
    "Recommendations": pg.show_rec,
}
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
    }
    
    .nav-list li:nth-child(1) span {
        color: orange !important;
        font-style: italic !important;
    }

    .nav-bottom-line {
        position: fixed;
        top: 6rem; /* Adjust to the height of your navbar */
        left: 0;
        right: 0;
        height: 0.5rem;
        background-color: #1a4723;
        z-index: 9999;
    }

    /* Optional: remove top border if unwanted */
    body {
        margin: 0;
        padding: 0;
    }

    header[data-testid="stHeader"] {
        margin: 0;
        padding: 0;
        border-top: none;
    }
    </style>
    <div class="nav-bottom-line"></div>
    """,
    unsafe_allow_html=True
)


go_to = functions.get(page)
if go_to:
    go_to()
