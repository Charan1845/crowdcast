"""Page: upload a crowd photo, get a people count with boxes (small groups) or a heatmap (crowds)."""
import hashlib

import streamlit as st
from PIL import UnidentifiedImageError

import crowd_model as cm
from feedback import save_feedback
from shared import count_photo, feedback_configured, png_bytes

REPO_URL = "https://github.com/Charan1845/crowdcast"
feedback_ready = feedback_configured()
METHOD_TEXT = {
    "detector": ("Each person found", ":material/person_search:",
                 "A group: a person detector (EfficientDet, trained on everyday photos) found and boxed each person."),
    "street": ("Crowd density", ":material/blur_on:",
               "A crowd: CSRNet estimated the count from a density map of where people are (Street scene model)."),
    "dense": ("Dense crowd density", ":material/blur_on:",
              "A dense crowd: CSRNet estimated the count from a density map (Dense crowd model)."),
}

# ---------- Header ----------
st.title("Count the crowd in any photo")
st.markdown("Upload a photo, from a small group to a packed stadium. Crowdcast picks the right method by itself: "
            "it **boxes each person** in groups, and estimates **crowd density** for big crowds.")
with st.container(horizontal=True, gap="small"):
    st.badge("VGG-16 CSRNet", icon=":material/neurology:", color="primary")
    st.badge("EfficientDet person detector", icon=":material/person_search:", color="gray")
    st.badge("TensorFlow", icon=":material/memory:", color="gray")

st.space("small")
controls, results = st.columns([1, 2], gap="large")

# ---------- Controls ----------
with controls:
    with st.container(border=True):
        st.subheader(":material/photo_camera: Your photo")
        uploaded = st.file_uploader("Crowd photo", type=["jpg", "jpeg", "png"], key="photo")
        st.caption(":material/lock: Photos are only used to count people and are not stored"
                   + (", unless you choose to share one with a correction." if feedback_ready else "."))

    with st.expander(":material/help: How does it choose?"):
        st.markdown(
            f"1. A crowd model (CSRNet) makes a first estimate, and a person detector looks for each person. "
            f"For bigger groups it also checks four overlapping parts of the photo.\n"
            f"2. **{cm.SMALL_GROUP_MAX} people or fewer:** the detector's count is used, with a box around each person. "
            f"The crowd model overcounts groups photographed at eye level.\n"
            f"3. **More than that:** if the dense-crowd model sees more than {cm.DENSE_SWITCH} people, its estimate is used. "
            f"Otherwise, if the detector found a group that fills only part of the photo, its count is used. "
            f"If not, the street-crowd model's density estimate is used."
        )

# ---------- Results ----------
with results:
    if uploaded is None:
        with st.container(border=True, height="stretch"):
            st.subheader(":material/route: How it works")
            step1, step2, step3 = st.columns(3)
            step1.markdown("**1. Upload a photo**\n\nAny JPG or PNG: a group photo, a street, a stadium.")
            step2.markdown("**2. It picks a method**\n\nSmall groups: find each person. Crowds: estimate density.")
            step3.markdown("**3. Get the count**\n\nWith a box around each person, or a heatmap of where the crowd is.")
            st.caption("Counting takes a few seconds on a laptop CPU.")
        st.stop()

    try:
        with st.spinner("Counting people..."):
            image, result, resized = count_photo(uploaded.getvalue())
    except UnidentifiedImageError:
        st.error("That file couldn't be read as an image. Try a JPG or PNG.", icon=":material/broken_image:")
        st.stop()

    label, icon, explanation = METHOD_TEXT[result.method]
    exact = result.method == "detector"
    count_text = f"{result.count:,.0f}" if exact else f"≈ {result.count:,.0f}"

    m1, m2 = st.columns(2)
    m1.metric("People in this photo", count_text, icon=":material/groups:", border=True,
              help="Number of people the detector found." if exact else "Sum of the predicted density map.")
    m2.metric("Method", label, icon=icon, border=True, help=explanation)
    st.caption(explanation)

    col_photo, col_result = st.columns(2)
    col_photo.image(image, caption="Your photo", alt="The uploaded photo", width="stretch")
    col_result.image(result.overlay,
                     caption=f"{result.count:,.0f} people found" if exact else "Where the model found people",
                     alt=(f"The photo with a numbered box around each of the {result.count:,.0f} people found" if exact else
                          f"The photo with a heatmap over the areas where the model estimated about {result.count:,.0f} people"),
                     width="stretch")

    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button("Download result image", png_bytes(result.overlay), file_name="crowdcast_result.png",
                           mime="image/png", icon=":material/download:", on_click="ignore")
        if resized:
            st.caption("Scaled down to 1024 px on its longest side, the size the models were trained on.")

    # ---------- Feedback (only when the private dataset is configured) ----------
    if feedback_ready:
        photo_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get("feedback_sent") == photo_id:
            st.success("Thanks! Your correction was saved.", icon=":material/check_circle:")
        else:
            with st.expander(":material/edit: Wrong count? Tell us the correct one"):
                with st.form("feedback_form", border=False):
                    correct = st.number_input("Correct number of people", min_value=0, max_value=100_000,
                                              value=round(result.count), step=1)
                    share_photo = st.checkbox("Also save this photo to help improve the model",
                                              help="Off by default. Saved privately, without its EXIF data "
                                                   "(location, device). Only share photos you have the right to share.")
                    submitted = st.form_submit_button("Send correction", icon=":material/send:")
                if submitted:
                    try:
                        with st.spinner("Saving..."):
                            save_feedback(st.secrets["HF_TOKEN"], st.secrets["HF_FEEDBACK_REPO"], result,
                                          int(correct), image, share_photo)
                    except Exception:
                        st.error("Couldn't save that right now. Please try again later.", icon=":material/error:")
                    else:
                        st.session_state["feedback_sent"] = photo_id
                        st.rerun()

# ---------- Footer ----------
st.divider()
with st.expander(":material/info: About the models"):
    st.markdown(
        f"""
**Groups: EfficientDet-Lite4 person detector** (TensorFlow Lite, pretrained on the COCO dataset of everyday photos).
It finds each person and draws a box, which works best when people are clearly visible.

**Crowds: CSRNet** (density estimation) with a **VGG-16** front end pretrained on ImageNet, fine-tuned on ShanghaiTech.
Instead of drawing a box around each person, which fails when heads overlap, it predicts a density map, and the count is its sum.

| Crowd model | Trained on | Test MAE | Published CSRNet |
|---|---|---|---|
| Street scene | ShanghaiTech Part B (street crowds) | 13.4 | 10.6 |
| Dense crowd | ShanghaiTech Part A (dense crowds) | 79.6 | 68.2 |

MAE is the average number of people the count is off by on test photos the model never saw.
Code, notebooks and full results: [{REPO_URL.removeprefix('https://')}]({REPO_URL}).
"""
    )
