"""Crowdcast app entry point: shared page setup and navigation.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
REPO_URL = "https://github.com/Charan1845/crowdcast"

st.set_page_config(page_title="Crowdcast", page_icon=str(ASSETS / "icon.svg"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large", link=REPO_URL)

page = st.navigation(
    [
        st.Page(
            "app_pages/count_crowd.py",
            title="Count a crowd",
            icon=":material/groups:",
            default=True,
            help="Upload a crowd photo to get a people count and a heatmap of where the model found people.",
        ),
    ],
    position="hidden",   # a single page for now, so no navigation bar
)
page.run()
