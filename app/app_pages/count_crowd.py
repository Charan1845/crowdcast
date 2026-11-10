"""Page: upload a crowd photo, get a people count and a density heatmap."""
import streamlit as st
from PIL import UnidentifiedImageError

import crowd_model as cm
from shared import count_people, png_bytes

REPO_URL = "https://github.com/Charan1845/crowdcast"


# ---------- Header ----------
st.title("Count the crowd in any photo")
st.markdown("Upload a crowd photo and a deep-learning model estimates **how many people** are in it, "
            "with a heatmap of **where** it found them.")
with st.container(horizontal=True, gap="small"):
    st.badge("VGG-16 CSRNet", icon=":material/neurology:", color="primary")
    st.badge("Transfer learning", icon=":material/model_training:", color="gray")
    st.badge("TensorFlow", icon=":material/memory:", color="gray")
    st.badge("Trained on ShanghaiTech", icon=":material/dataset:", color="gray")

st.space("small")
controls, results = st.columns([1, 2], gap="large")

# ---------- Controls ----------
with controls:
    with st.container(border=True):
        st.subheader(":material/tune: Settings")
        kind = st.segmented_control(
            "Crowd type",
            options=list(cm.MODELS),
            format_func=lambda k: cm.MODELS[k].label,
            default="street",
            required=True,
            key="kind",
            width="stretch",
            help="Street scene: you can pick out individual people (tens to a few hundred). "
                 "Dense crowd: mostly a sea of heads (hundreds to thousands).",
        )
        uploaded = st.file_uploader("Crowd photo", type=["jpg", "jpeg", "png"], key="photo")
        st.caption(":material/lock: Photos are only used to count people and are not stored.")

    with st.expander(":material/help: Which crowd type should I pick?"):
        st.markdown(
            "- **Street scene**: pavements, squares, shopping streets. You can see individual people.\n"
            "- **Dense crowd**: rallies, stadiums, festivals. Mostly tiny heads packed together.\n\n"
            "Not sure? Try both. The app suggests switching if the count looks off for the model."
        )

# ---------- Results ----------
with results:
    if uploaded is None:
        with st.container(border=True, height="stretch"):
            st.subheader(":material/photo_camera: How it works")
            step1, step2, step3 = st.columns(3)
            step1.markdown("**1. Pick a crowd type**\n\nStreet scene or dense crowd. Each uses a model trained on that kind of photo.")
            step2.markdown("**2. Upload a photo**\n\nAny JPG or PNG. Large photos are scaled to 1024 px, the size the models were trained on.")
            step3.markdown("**3. Get the count**\n\nThe model draws a density map of where people are, and its total is the count.")
            st.caption("Counting takes about 2 seconds on a laptop CPU.")
        st.stop()

    info = cm.MODELS[kind]
    try:
        with st.spinner("Counting people..."):
            image, overlay, count, resized = count_people(uploaded.getvalue(), kind)
    except UnidentifiedImageError:
        st.error("That file couldn't be read as an image. Try a JPG or PNG.", icon=":material/broken_image:")
        st.stop()

    low, high = max(0, count - info.test_mae), count + info.test_mae
    m1, m2, m3 = st.columns(3)
    m1.metric("People counted", f"{count:,.0f}", icon=":material/groups:", border=True,
              help="Sum of the predicted density map.")
    m2.metric("Likely range", f"{low:,.0f} – {high:,.0f}", icon=":material/straighten:", border=True,
              help=f"± the model's average error on its {info.dataset} test photos ({info.test_mae:g} people).")
    m3.metric("Model", info.label.split()[0], icon=":material/neurology:", border=True,
              help=f"{info.label} model, trained on {info.dataset}. Test photos averaged {info.test_mean_count} people.")

    if kind == "street" and count > 500:
        st.warning("This looks like a dense crowd. The **Dense crowd** model may count it better.",
                   icon=":material/lightbulb:")
    elif kind == "dense" and count < 100:
        st.warning("This looks like a fairly sparse scene. The **Street scene** model may count it better.",
                   icon=":material/lightbulb:")

    col_photo, col_heat = st.columns(2)
    col_photo.image(image, caption="Your photo", alt="The uploaded crowd photo", width="stretch")
    col_heat.image(overlay, caption="Where the model found people",
                   alt=f"The photo with a heatmap over the areas where the model counted about {count:,.0f} people",
                   width="stretch")

    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button("Download heatmap", png_bytes(overlay), file_name="crowdcast_heatmap.png",
                           mime="image/png", icon=":material/download:", on_click="ignore")
        if resized:
            st.caption("Scaled down to 1024 px on its longest side, the size the models were trained on.")

# ---------- Footer ----------
st.divider()
with st.expander(":material/info: About the model"):
    st.markdown(
        f"""
**CSRNet** (density estimation) with a **VGG-16** front end pretrained on ImageNet, fine-tuned on
ShanghaiTech. Instead of drawing a box around each person, which fails when heads overlap, it predicts
a density map, and the count is its sum.

| Model | Trained on | Test MAE | Published CSRNet |
|---|---|---|---|
| Street scene | ShanghaiTech Part B (sparse streets) | 13.4 | 10.6 |
| Dense crowd | ShanghaiTech Part A (dense crowds) | 79.6 | 68.2 |

MAE is the average number of people the count is off by on test photos the model never saw.
Photos unlike ShanghaiTech (unusual angles, lighting or places) may count less accurately.
Code, notebooks and full results: [{REPO_URL.removeprefix('https://')}]({REPO_URL}).
"""
    )
