# ==========================================
# app.py — PSV Dashboard Overview (v4.0 MVP)
# ==========================================
import streamlit as st
import plotly.express as px
from utils import load_all_cleaned, compute_player_index, data_freshness_label, export_excel, export_pdf
import pandas as pd

st.set_page_config(page_title="PSV Dashboard", layout="wide")
st.title("⚽ PSV Unified Marketing Insights — MVP")
st.caption("Marketing Intelligence • Player Value • Fan Sentiment • Reputation Tracking")

# Load data
frames = load_all_cleaned()
players = frames.get("transfermarket", pd.DataFrame())
social = frames.get("socials_topics", pd.DataFrame())
news = frames.get("news_topics", pd.DataFrame())
posts = frames.get("socials_overview", pd.DataFrame())

st.sidebar.header("Filters")
period = st.sidebar.selectbox("Period", ["Last 7 days", "Last 30 days", "Season"])
platforms = st.sidebar.multiselect("Platforms", ["Facebook","Instagram","TikTok","YouTube","Twitter/X"])
st.sidebar.markdown(f"🕓 {data_freshness_label()}")

st.markdown("### 🧭 Overall Club Sentiment")
if not social.empty or not news.empty:
    soc = social[["pos_sentiment_(%)","neg_sentiment_(%)","neu_sentiment_(%)"]].mean()
    nws = news[["pos_sentiment_(%)","neg_sentiment_(%)","neu_sentiment_(%)"]].mean()
    club = (soc.add(nws, fill_value=0)/2).rename("avg")
    fig = px.pie(values=club.values, names=club.index, title="Club-wide Sentiment Split", color=club.index,
                 color_discrete_map={"pos_sentiment_(%)":"#2ca02c","neg_sentiment_(%)":"#d62728","neu_sentiment_(%)":"#1f77b4"})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sentiment data found.")

st.markdown("### 🏆 Player Index Summary")
pi = compute_player_index(players, social, news, posts)
if pi.empty:
    st.warning("Insufficient data for Player Index.")
else:
    fig = px.bar(pi.head(10), x="player_index", y="player", orientation="h", color="player_index",
                 color_continuous_scale="Viridis", title="Top Players by Popularity Index")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pi.head(15))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Export to Excel", export_excel(pi), "player_index.xlsx")
    with c2:
        st.download_button("⬇️ Export to PDF", export_pdf(pi, "Player Index Summary"), "player_index.pdf")

st.markdown("### 🔥 Trending Topics")
if not social.empty:
    top = social.groupby("topic").size().reset_index(name="mentions").sort_values("mentions", ascending=False).head(10)
    fig2 = px.bar(top, x="mentions", y="topic", orientation="h", color="mentions", color_continuous_scale="Oranges")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No topic data available.")

st.markdown("---")
st.caption("© PSV Data Intelligence | MVP v4.0 — Full export & sentiment-ready architecture")
