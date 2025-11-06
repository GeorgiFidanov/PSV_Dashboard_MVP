# ==========================================
# pages/5_Reputation.py
# ==========================================
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_all_cleaned

st.title("Club Reputation Insights")
st.caption("Track PSV’s brand perception across regions and media sources.")

frames = load_all_cleaned()
rep = frames.get("reputation", pd.DataFrame())

if rep.empty:
    st.info("No reputation data available.")
    st.stop()

st.subheader("🏙️ Reputation Score by Region")
fig1 = px.bar(
    rep, x="reputation_score", y="region",
    orientation="h", color="reputation_score",
    color_continuous_scale="Blues",
    title="Regional Reputation Scores"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📰 Sentiment per Source")
fig2 = px.scatter(
    rep, x="pos_sentiment_(%)", y="neg_sentiment_(%)",
    size="mentions", color="source",
    hover_name="region", title="Media Sentiment Distribution"
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(rep)
