# ==========================================
# pages/3_Player_Comparison.py
# ==========================================
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_all_cleaned

st.title("Player Comparison")
st.caption("Compare performance, sentiment, and market indicators.")

frames = load_all_cleaned()
players = frames.get("player_comparison", pd.DataFrame())

if players.empty:
    st.info("No player comparison data available.")
    st.stop()

# Dropdown selectors
p1 = st.selectbox("Select Player 1", players["name"].unique())
p2 = st.selectbox("Select Player 2", players["name"].unique())

subset = players[players["name"].isin([p1, p2])]

st.subheader("📊 Market Value & Sentiment")
fig1 = px.scatter(
    subset, x="market_value_eur", y="performance_score",
    color="name", size="mentions", hover_name="name",
    title="Market Value vs Performance",
    text="pos_sentiment_(%)"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🏆 Player Stats Overview")
fig2 = px.bar(
    subset.melt(id_vars=["name"], value_vars=["goals", "assists"]),
    x="variable", y="value", color="name",
    barmode="group", title="Goals & Assists Comparison"
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(subset)
