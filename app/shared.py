"""Cached model loading and counting, shared by the app's pages."""
import io

import streamlit as st
from PIL import Image

import crowd_model as cm


@st.cache_resource(show_spinner=False)
def get_model(kind: str):
    return cm.load_model(kind)


@st.cache_data(show_spinner=False, max_entries=20)
def count_people(image_bytes: bytes, kind: str):
    """Returns (image, heatmap overlay, people count, was_resized)."""
    image, resized = cm.prepare_image(Image.open(io.BytesIO(image_bytes)))
    dmap = cm.predict_density(get_model(kind), image)
    return image, cm.heatmap_overlay(image, dmap), float(max(dmap.sum(), 0)), resized


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
