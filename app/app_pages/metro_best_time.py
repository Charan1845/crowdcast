"""Page: upload a photo taken at a Namma Metro station, count the people, and suggest the best time to travel.

Counting: the CSRNet models (crowd_model.py). Hourly pattern: app/data/metro_hourly_pattern.csv, produced by
notebooks/04_metro_best_time.ipynb (XGBoost on BMRCL hourly station entries, Aug-Sep 2025).
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from PIL import UnidentifiedImageError

import crowd_model as cm
from shared import count_people, photo_taken_at

PATTERN_FILE = Path(__file__).resolve().parent.parent / "data" / "metro_hourly_pattern.csv"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FIRST_HOUR, LAST_HOUR = 5, 23            # metro carries passengers roughly 5 am to 11 pm
PATTERN_MAE = 54.0                        # notebook 04, test week 22-30 Sep 2025
DENSE_SWITCH = 500                        # above this, the street model is out of its range; use the dense model
LEVELS = [("Quiet", 0.25, "#22C55E"), ("Moderate", 0.5, "#F59E0B"), ("Busy", 0.75, "#F97316"), ("Very busy", 9, "#EF4444")]


@st.cache_data(show_spinner=False)
def load_pattern() -> pd.DataFrame:
    return pd.read_csv(PATTERN_FILE)


def hour_label(h: int) -> str:
    return datetime(2000, 1, 1, h % 24).strftime("%I %p").lstrip("0").lower()


def slot(h: int) -> str:
    return f"{hour_label(h)}–{hour_label(h + 1)}"


def crowd_level(share: float) -> str:
    return next(name for name, cutoff, _ in LEVELS if share < cutoff)


st.title("Metro crowd check")
st.markdown("Took a photo at a **Namma Metro** station? Upload it: the app **counts the people** in it, "
            "tells you **how busy this hour usually is**, and suggests the **best time to travel**.")
with st.container(horizontal=True, gap="small"):
    st.badge("CSRNet people counting", icon=":material/groups:", color="primary")
    st.badge("XGBoost hourly crowd model", icon=":material/insights:", color="gray")
    st.badge("BMRCL ridership, Aug–Sep 2025", icon=":material/dataset:", color="gray")

pattern = load_pattern()
stations = sorted(pattern["station"].unique())
now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

st.space("small")
controls, results = st.columns([1, 2], gap="large")

with controls:
    with st.container(border=True):
        st.subheader(":material/photo_camera: Your photo")
        station = st.selectbox("Station", stations, index=None, placeholder="Type to search 83 stations",
                               key="station", help="The station where you took the photo.")
        uploaded = st.file_uploader("Photo of the platform or concourse", type=["jpg", "jpeg", "png"], key="photo")

    if uploaded is not None:
        taken = photo_taken_at(uploaded.getvalue())
        default = taken or now_ist
        fid = uploaded.file_id   # new photo -> widgets reset to that photo's time
        with st.container(border=True):
            st.subheader(":material/schedule: When")
            st.caption(f"Read from the photo: {taken:%a %d %b, %I:%M %p}." if taken else
                       "The photo doesn't record when it was taken, so the current time is used. Change it if needed.")
            day = st.segmented_control("Day", options=list(range(7)), format_func=lambda d: DAYS[d],
                                       default=default.weekday(), required=True, key=f"day_{fid}", width="stretch")
            hour = st.selectbox("Hour the photo was taken", list(range(FIRST_HOUR, LAST_HOUR)),
                                index=min(max(default.hour, FIRST_HOUR), LAST_HOUR - 1) - FIRST_HOUR,
                                format_func=slot, key=f"hour_{fid}")
            start = min(hour + 1, 20)
            window = st.slider("I can travel between", min_value=FIRST_HOUR, max_value=LAST_HOUR,
                               value=(start, 22), format="%d:00", key=f"window_{fid}",
                               help="The suggestion is the least crowded hour in this window. "
                                    "By default: from the next hour until 10 pm. Widen it if you can travel earlier or later.")

    with st.expander(":material/help: How does this work?"):
        st.markdown(
            f"""
1. **Count:** a CSRNet model (VGG-16, trained on ShanghaiTech street scenes) counts the people in your photo
   and draws a heatmap. Typical error on its test photos: ±13 people.
2. **Hourly pattern:** an XGBoost model, trained on real BMRCL hourly **entries** for 83 stations
   (Aug–Sep 2025), predicts how many people enter this station each hour of a normal day.
   Typical error: ±{PATTERN_MAE:g} entries per hour.
3. **Suggestion:** the least crowded hour in your window. The *people you'd see then* figure scales your photo's
   count by how that hour compares with the hour you took the photo. That assumes the crowd in view rises and
   falls with station entries, so treat it as an **estimate**.

Festivals, events, strikes and heavy rain are not in the pattern.
"""
        )

with results:
    if station is None or uploaded is None:
        with st.container(border=True):
            st.subheader(":material/route: How it works")
            s1, s2, s3 = st.columns(3)
            s1.markdown("**1. Pick the station**\n\nWhere you are, or where you took the photo.")
            s2.markdown("**2. Upload a photo**\n\nOf the platform or concourse. The time is read from the photo if it has one.")
            s3.markdown("**3. Get advice**\n\nPeople counted now, how busy this hour usually is, and the best time to go.")
            missing = " and ".join(x for x, ok in [("a station", station is not None), ("a photo", uploaded is not None)] if not ok)
            st.caption(f"Choose {missing} to start.")
        st.stop()

    # ---- Count the people in the photo ----
    try:
        with st.spinner("Counting people..."):
            image, overlay, count, _ = count_people(uploaded.getvalue(), "street")
            model_used = cm.MODELS["street"]
            if count > DENSE_SWITCH:   # very dense: the dense-crowd model is the better fit
                image, overlay, count, _ = count_people(uploaded.getvalue(), "dense")
                model_used = cm.MODELS["dense"]
    except UnidentifiedImageError:
        st.error("That file couldn't be read as an image. Try a JPG or PNG.", icon=":material/broken_image:")
        st.stop()

    # ---- Compare with the station's usual pattern ----
    lo, hi = window
    day_df = pattern[(pattern["station"] == station) & (pattern["weekday"] == day)
                     & pattern["hour"].between(FIRST_HOUR, LAST_HOUR - 1)].copy()
    peak = pattern.loc[pattern["station"] == station, "expected_entries"].max()
    day_df["level"] = (day_df["expected_entries"] / peak).map(crowd_level) if peak > 0 else "Quiet"
    by_hour = day_df.set_index("hour")
    now_entries = by_hour.loc[hour, "expected_entries"]
    now_level = by_hour.loc[hour, "level"]

    options = day_df[day_df["hour"].between(lo, hi - 1)]
    if hi - lo < 1 or options.empty:
        st.info("Pick a travel window that includes at least one hour.", icon=":material/schedule:")
        st.stop()
    best = options.loc[options["expected_entries"].idxmin()]
    best_hour = int(best["hour"])
    ratio = best["expected_entries"] / now_entries if now_entries > 0 else None
    expected_people = count * ratio if ratio is not None else None

    m1, m2, m3 = st.columns(3)
    m1.metric("People in your photo", f"{count:,.0f}", icon=":material/groups:", border=True,
              help=f"Counted by the {model_used.label} model (typical error ±{model_used.test_mae:g}).")
    m2.metric(f"{slot(hour)} at {station}", now_level, icon=":material/schedule:", border=True,
              help=f"About {now_entries:,.0f} people usually enter this station in this hour on a {DAY_NAMES[day]}, "
                   f"compared with this station's busiest hour of the week.")
    m3.metric("Best time to travel", slot(best_hour), icon=":material/thumb_up:", border=True,
              delta=(f"-{1 - ratio:.0%} people" if ratio is not None and ratio < 0.95 else None),
              delta_color="inverse",   # fewer people is good, so show the drop in green
              help=f"The least crowded hour between {hour_label(lo)} and {hour_label(hi)} on a typical {DAY_NAMES[day]}.")

    if best_hour == hour or (ratio is not None and ratio >= 0.95):
        st.success(f"**Now is already about the quietest time** in your window at {station}. Good time to travel.",
                   icon=":material/check_circle:")
    elif expected_people is not None:
        st.success(
            f"**Go at {slot(best_hour)}** instead. {station} is usually about **{1 - ratio:.0%} less crowded** then. "
            f"A photo like yours would show roughly **{expected_people:,.0f} people instead of {count:,.0f}** (estimate).",
            icon=":material/tips_and_updates:")

    col_photo, col_heat = st.columns(2)
    col_photo.image(image, caption="Your photo", alt=f"Photo taken at {station} metro station", width="stretch")
    col_heat.image(overlay, caption=f"{count:,.0f} people found",
                   alt=f"The photo with a heatmap over the {count:,.0f} people the model found", width="stretch")

    # ---- The day at this station ----
    chart_df = day_df.assign(
        label=day_df["hour"].map(hour_label),
        time=day_df["hour"].map(slot),
        in_window=day_df["hour"].between(lo, hi - 1) | (day_df["hour"] == hour),
        marker=day_df["hour"].map(lambda h: "Your photo" if h == hour else ("Suggested" if h == best_hour else "")),
    )
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("expected_entries:Q", title="people entering per hour"),
        color=alt.Color("level:N", title="Usual crowd",
                        scale=alt.Scale(domain=[n for n, _, _ in LEVELS], range=[c for _, _, c in LEVELS]),
                        legend=alt.Legend(orient="top")),
        opacity=alt.condition("datum.in_window", alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("time:N", title="Hour"),
                 alt.Tooltip("expected_entries:Q", title="Usual entries", format=",.0f"),
                 alt.Tooltip("level:N", title="Crowd")],
    )
    labels = alt.Chart(chart_df[chart_df["marker"] != ""]).mark_text(dy=-10, fontWeight="bold", color="#E6EDF3").encode(
        x=alt.X("label:N", sort=None), y="expected_entries:Q", text="marker:N")
    st.subheader(f"A typical {DAY_NAMES[day]} at {station}")
    st.altair_chart(bars + labels, height=320,
                    alt=f"Bar chart of usual entries per hour at {station} on a {DAY_NAMES[day]}, marking the hour of "
                        f"your photo ({slot(hour)}) and the suggested time ({slot(best_hour)}).")
    st.caption("Faded bars are outside your travel window. Crowd levels are relative to this station's busiest hour of the week.")

st.divider()
st.caption("Ridership data: BMRCL, obtained through RTI by Vivek Mathew and published on "
           "[OpenCity](https://data.opencity.in/dataset/bmrcl-station-wise-ridership-data) "
           "(Creative Commons Non-Commercial). Code and results: "
           "[github.com/Charan1845/crowdcast](https://github.com/Charan1845/crowdcast).")
