"""Crowdcast app entry point: shared page setup and top navigation.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair  # noqa: F401  (imported here once, so two sessions starting together don't race to import it)
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
        st.Page(
            "app_pages/metro_best_time.py",
            title="Metro crowd check",
            icon=":material/train:",
            help="Upload a photo taken at a Namma Metro station: the app counts the people, shows how busy that "
                 "hour usually is, and suggests the best time to travel.",
        ),
    ],
    position="top",
)
page.run()
