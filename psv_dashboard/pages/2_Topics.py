# ==========================================
# pages/2_Topics.py — PSV Dashboard Topics
# ==========================================
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_all_cleaned

st.title("Topics Dashboard")
st.caption("Monitor trending discussion themes, engagement, and sentiment.")

frames = load_all_cleaned()
topics = frames.get("topics", pd.DataFrame())

if topics.empty:
    st.info("No topic data available.")
    st.stop()

st.subheader("🔥 Top Topics by Mentions")
fig1 = px.bar(
    topics.sort_values("mentions", ascending=False).head(10),
    x="mentions", y="topic", orientation="h",
    color="engagement", color_continuous_scale="Oranges",
    title="Top Mentioned Topics"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💬 Topic Sentiment Overview")
fig2 = px.scatter(
    topics,
    x="pos_sentiment_(%)", y="neg_sentiment_(%)",
    size="engagement", color="topic",
    hover_name="topic", title="Sentiment Distribution by Topic"
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(topics)
