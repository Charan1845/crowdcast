"""Cached model loading and counting, shared by the app's pages."""
import io

import streamlit as st
from PIL import Image

import crowd_model as cm


@st.cache_resource(show_spinner=False)
def get_model(kind: str):
    return cm.load_model(kind)


@st.cache_resource(show_spinner=False)
def get_detector():
    return cm.load_detector()


@st.cache_data(show_spinner=False, max_entries=20)
def count_photo(image_bytes: bytes):
    """Returns (image, CountResult, was_resized). Picks detector or CSRNet automatically."""
    image, resized = cm.prepare_image(Image.open(io.BytesIO(image_bytes)))
    result = cm.count_auto(image, {"street": get_model("street"), "dense": get_model("dense")}, get_detector())
    return image, result, resized


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
