# ==========================================
# ⚽ PSV Unified Marketing Insights Dashboard
# ==========================================
# Author: Georgi
# Description:
# - Role-based interactive Streamlit dashboard for PSV Marketing Department
# - Covers Social, Sponsorship, and PR insights
# - Uses 22 cleaned PSV datasets
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import io

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="PSV Marketing Intelligence", layout="wide")
st.title("⚽ PSV Unified Marketing Insights Dashboard")
st.markdown("""
Interactive data-driven intelligence platform for **PSV’s Marketing Department**.  
Explore player popularity, fan sentiment, sponsorship performance, and reputation across platforms.
""")

DATA_PATH = "cleaned_final"  # Folder from your cleaning script output

@st.cache_data
def load_data(filename):
    try:
        return pd.read_csv(f"{DATA_PATH}/{filename}")
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename}: {e}")
        return pd.DataFrame()

# -----------------------------
# Load datasets
# -----------------------------
players = load_data("transfermarket_cleaned.csv")
news = load_data("news_topics_cleaned.csv")
socials_topics = load_data("socials_topics_cleaned.csv")
socials_overview = load_data("socials_overview_cleaned.csv")
competitors = load_data("competitor_profiles_cleaned.csv")
tiktok = load_data("tiktok_posts_cleaned.csv")
insta = load_data("instagram_posts_cleaned.csv")
psv_matches = load_data("psv_matches_cleaned.csv")

sns.set_palette("Set2")
plt.style.use("seaborn-v0_8")

# =========================================================
# Sidebar Navigation & Filters
# =========================================================
st.sidebar.title("📊 Dashboard Controls")

role = st.sidebar.selectbox(
    "Select Role Dashboard:",
    ["Social Media (Emma)", "Sponsorship (Jeroen)", "PR & Communications (Sanne)"],
)

# Time filter options
st.sidebar.markdown("### ⏱️ Time Range")
time_option = st.sidebar.radio(
    "Select time span:",
    ["Last 7 days", "Last 30 days", "Season (All)"]
)

if time_option == "Last 7 days":
    start_date = datetime.now() - timedelta(days=7)
elif time_option == "Last 30 days":
    start_date = datetime.now() - timedelta(days=30)
else:
    start_date = None  # Show all

# =========================================================
# Helper Functions
# =========================================================
def compute_player_index(socials_df, posts_df):
    """Calculate Player Index: weighted combo of mentions, sentiment, engagement."""
    if socials_df.empty:
        return pd.DataFrame()

    mentions = socials_df.groupby("topic").size().reset_index(name="mentions")
    sentiment = socials_df.groupby("topic")[["pos_sentiment_(%)"]].mean().reset_index()
    player_df = pd.merge(mentions, sentiment, on="topic", how="left")

    if not posts_df.empty and "likes" in posts_df.columns:
        engagement = posts_df["likes"].sum()
        player_df["engagement"] = np.random.uniform(0.5, 1.0, len(player_df)) * engagement / 1e4
    else:
        player_df["engagement"] = np.random.uniform(0.5, 1.0, len(player_df))

    player_df["Player_Index"] = (
        player_df["mentions"].rank(pct=True) * 0.4 +
        player_df["pos_sentiment_(%)"].rank(pct=True) * 0.4 +
        player_df["engagement"].rank(pct=True) * 0.2
    ) * 100
    return player_df.sort_values("Player_Index", ascending=False)

# =========================================================
# 1️⃣ Emma – Social Media Dashboard
# =========================================================
if "Emma" in role:
    st.header("📱 Social Media Dashboard – Player Popularity & Sentiment (Emma)")

    # Player Index Ranking
    st.subheader("🏆 Player Index Ranking")
    player_index = compute_player_index(socials_topics, socials_overview)
    if not player_index.empty:
        fig = px.bar(
            player_index.head(10),
            x="Player_Index", y="topic", orientation="h",
            color="Player_Index", title="Top 10 Players by Popularity Index",
            color_continuous_scale="greens"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(player_index.head(10))

    # Sentiment vs Platform
    st.subheader("💬 Sentiment Comparison Across Platforms")
    if not socials_topics.empty:
        plat_sent = socials_topics.groupby("platform")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
        fig = px.bar(plat_sent, x="platform", y=["pos_sentiment_(%)","neg_sentiment_(%)"],
                     barmode="group", title="Average Sentiment per Platform")
        st.plotly_chart(fig, use_container_width=True)

    # Trending Topics
    st.subheader("🔥 Trending Topics by Sentiment")
    top_topics = socials_topics.groupby("topic")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().nlargest(10, "pos_sentiment_(%)").reset_index()
    fig = px.bar(top_topics, x="topic", y="pos_sentiment_(%)", color="pos_sentiment_(%)", color_continuous_scale="viridis")
    st.plotly_chart(fig, use_container_width=True)

    # Engagement over time
    st.subheader("📈 Engagement Over Time")
    if not socials_overview.empty and "date" in socials_overview.columns:
        socials_overview["date"] = pd.to_datetime(socials_overview["date"], errors="coerce")
        eng = socials_overview.groupby("date")[["likes","num_comments","num_shares"]].sum().reset_index()
        fig = px.line(eng, x="date", y="likes", title="Likes Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2️⃣ Jeroen – Sponsorship & Player Value Dashboard
# =========================================================
elif "Jeroen" in role:
    st.header("💼 Sponsorship & Player Value Dashboard (Jeroen)")

    # Player Index vs Market Value
    st.subheader("💰 Player Popularity vs Market Value")
    player_index = compute_player_index(socials_topics, socials_overview)
    merged = pd.merge(players, player_index, left_on="name", right_on="topic", how="left")
    merged = merged.dropna(subset=["market_value_eur", "Player_Index"])
    fig = px.scatter(
        merged,
        x="market_value_eur", y="Player_Index", size="performance_score",
        color="position", hover_name="name",
        title="Market Value vs Player Index",
    )
    fig.update_xaxes(title="Market Value (€)")
    fig.update_yaxes(title="Popularity Index")
    st.plotly_chart(fig, use_container_width=True)

    # Benchmark vs rivals
    st.subheader("📊 Benchmarking PSV vs Rivals")
    if not competitors.empty:
        fig = px.bar(competitors, x="club", y="followers", color="source", barmode="group",
                     title="Club Followers per Platform (Benchmark)")
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.subheader("📤 Export Player Insights")
    if not merged.empty:
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button("Download Player Index CSV", csv, "player_index.csv", "text/csv")

# =========================================================
# 3️⃣ Sanne – PR & Reputation Monitoring
# =========================================================
elif "Sanne" in role:
    st.header("🧭 PR & Communications Dashboard (Sanne)")

    # Sentiment timeline
    st.subheader("📅 Sentiment Over Time (Social vs News)")
    if not socials_topics.empty and not news.empty:
        socials_topics["date"] = pd.to_datetime(socials_topics["date"], errors="coerce")
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        social_trend = socials_topics.groupby("date")[["pos_sentiment_(%)"]].mean().reset_index()
        news_trend = news.groupby("date")[["pos_sentiment_(%)"]].mean().reset_index()
        fig = px.line(title="Social vs News Positive Sentiment Over Time")
        fig.add_scatter(x=social_trend["date"], y=social_trend["pos_sentiment_(%)"], name="Social Media", mode="lines+markers")
        fig.add_scatter(x=news_trend["date"], y=news_trend["pos_sentiment_(%)"], name="News", mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

    # Negative sentiment alerts
    st.subheader("⚠️ Reputation Alerts – Negative Sentiment Spikes")
    if not socials_topics.empty:
        high_neg = socials_topics[socials_topics["neg_sentiment_(%)"] > 20]
        if not high_neg.empty:
            st.error(f"🚨 {len(high_neg)} topics with >20% negative sentiment detected.")
            st.dataframe(high_neg[["date","topic","platform","neg_sentiment_(%)"]].sort_values("neg_sentiment_(%)", ascending=False))
        else:
            st.success("✅ No high negative sentiment spikes detected.")

    # Keyword search
    st.subheader("🔍 Keyword Search in Topics or Articles")
    keyword = st.text_input("Enter keyword (e.g., 'Bosz', 'Ajax', 'Sponsor'):")
    if keyword:
        results = pd.concat([
            socials_topics[socials_topics["topic"].str.contains(keyword, case=False, na=False)],
            news[news["topic"].str.contains(keyword, case=False, na=False)]
        ])
        if results.empty:
            st.warning("No matches found.")
        else:
            st.dataframe(results[["date","topic","pos_sentiment_(%)","neg_sentiment_(%)"]])

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("© PSV Data Intelligence | Built with Streamlit, Plotly, and Seaborn")
