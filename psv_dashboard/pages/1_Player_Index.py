# ==========================================
# pages/1_Player_Index.py
# ==========================================
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import load_all_cleaned, compute_player_index, export_excel, export_pdf

st.title("📈 Player Index Insights")
frames = load_all_cleaned()
players = frames.get("transfermarket", pd.DataFrame())
social = frames.get("socials_topics", pd.DataFrame())
news = frames.get("news_topics", pd.DataFrame())
posts = frames.get("socials_overview", pd.DataFrame())

pi = compute_player_index(players, social, news, posts)
if pi.empty:
    st.warning("Not enough data for Player Index.")
    st.stop()

query = st.text_input("Search for a player:")
if query:
    pi = pi[pi["player"].str.contains(query, case=False, na=False)]

st.subheader("Top Players by Index")
fig = px.bar(pi.head(20), x="player_index", y="player", orientation="h", color="player_index",
             color_continuous_scale="Viridis", title="Popularity Index")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(pi)

st.subheader("Market Value Correlation (Sponsors View)")
if not players.empty:
    merged = pd.merge(players.rename(columns=str.lower), pi, left_on="name", right_on="player", how="left")
    merged = merged.dropna(subset=["market_value_eur","player_index"])
      
    fig2 = px.scatter(
    merged, 
    x="market_value_eur", 
    y="player_index", 
    color="position",
    hover_name="name", 
    title="Market Value vs Player Popularity Index (€)",
    size="market_value_eur",  # size the dots by market value
    size_max=60               # make the dots clearly visible
    )
    st.plotly_chart(fig2, use_container_width=True)


c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Export Player Index (Excel)", export_excel(pi), "player_index.xlsx")
with c2:
    st.download_button("⬇️ Export (PDF)", export_pdf(pi, "Player Index Insights"), "player_index.pdf")
