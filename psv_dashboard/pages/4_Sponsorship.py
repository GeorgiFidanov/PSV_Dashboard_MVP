# ==========================================
# pages/4_Sponsorship.py
# ==========================================
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_all_cleaned

st.title("Sponsorship Performance")
st.caption("Analyze sponsor ROI, campaign reach, and engagement rates.")

frames = load_all_cleaned()
sponsorship = frames.get("sponsorship", pd.DataFrame())

if sponsorship.empty:
    st.info("No sponsorship data available.")
    st.stop()

st.subheader("📈 ROI by Sponsor")
fig1 = px.bar(
    sponsorship, x="sponsor", y="roi_(%)",
    color="roi_(%)", color_continuous_scale="Greens",
    title="Return on Investment (%) by Sponsor"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📣 Engagement Rate per Platform")
fig2 = px.scatter(
    sponsorship, x="reach", y="engagement_rate",
    color="platform", size="mentions",
    hover_name="sponsor", title="Reach vs Engagement Rate"
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(sponsorship)
