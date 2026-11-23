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
def count_photo(image_bytes: bytes, counting_version: str = ""):
    """Returns (image, CountResult, was_resized). Picks detector or CSRNet automatically.
    counting_version is part of the cache key, so a change to the counting code can't serve old answers."""
    image, resized = cm.prepare_image(Image.open(io.BytesIO(image_bytes)))
    result = cm.count_auto(image, {"street": get_model("street"), "dense": get_model("dense")}, get_detector())
    return image, result, resized


def feedback_configured() -> bool:
    """True when the Hugging Face secrets for saving corrections are set."""
    try:
        return bool(st.secrets.get("HF_TOKEN")) and bool(st.secrets.get("HF_FEEDBACK_REPO"))
    except FileNotFoundError:   # no secrets file at all (e.g. a fresh local checkout)
        return False
    except st.errors.StreamlitSecretNotFoundError:
        return False


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
