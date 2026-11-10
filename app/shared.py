"""Cached model loading and counting, shared by the app's pages."""
import io
from datetime import datetime

import streamlit as st
from PIL import Image

import crowd_model as cm

EXIF_DATETIME_ORIGINAL = 36867   # when the photo was taken
EXIF_DATETIME = 306              # fallback: when the file was last written by the camera
EXIF_IFD = 0x8769


@st.cache_resource(show_spinner=False)
def get_model(kind: str):
    return cm.load_model(kind)


@st.cache_data(show_spinner=False, max_entries=20)
def count_people(image_bytes: bytes, kind: str):
    """Returns (image, heatmap overlay, people count, was_resized)."""
    image, resized = cm.prepare_image(Image.open(io.BytesIO(image_bytes)))
    dmap = cm.predict_density(get_model(kind), image)
    return image, cm.heatmap_overlay(image, dmap), float(max(dmap.sum(), 0)), resized


def photo_taken_at(image_bytes: bytes) -> datetime | None:
    """The time the photo was taken, from its EXIF data, or None if the file doesn't record it."""
    try:
        exif = Image.open(io.BytesIO(image_bytes)).getexif()
        raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        return datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S") if raw else None
    except (ValueError, OSError, AttributeError):
        return None


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
