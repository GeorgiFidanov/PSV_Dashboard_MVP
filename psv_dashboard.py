# ==========================================
# ⚽ PSV Unified Marketing Insights Dashboard
# ==========================================
# Author: Georgi
# Description:
# - Interactive Streamlit dashboard for PSV Marketing data
# - Builds on cleaned datasets from previous notebook
# - Adds decision-focused visuals for Emma, Jeroen & Sanne
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="PSV Marketing Intelligence", layout="wide")
st.title("PSV Unified Marketing Insights Dashboard")
st.markdown("Interactive insights for **PSV’s Marketing Department** – bridging social, sentiment, and player data for data-driven brand strategy.")

DATA_PATH = "cleaned"  # Folder from your cleaning script output

@st.cache_data
def load_data(filename):
    path = f"{DATA_PATH}/{filename}"
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename}: {e}")
        return pd.DataFrame()

# Load datasets
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

# -----------------------------
# Navigation
# -----------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    [
        "Overview Pulse",
        "Player Insights",
        "Content & Engagement",
        "Reputation & Sentiment",
        "Sponsorship & Benchmarking",
    ],
)

# =========================================================
# 1. Overview Pulse
# =========================================================
if page == "Overview Pulse":
    st.header("🏟️ PSV Brand & Fan Pulse Overview")

    # Sentiment Pie
    st.subheader("📊 Overall PSV Sentiment Breakdown (Social + News)")
    if not socials_topics.empty:
        total_sentiment = socials_topics[["neg_sentiment_(%)","pos_sentiment_(%)","neu_sentiment_(%)"]].mean()
        fig = px.pie(
            names=["Negative", "Positive", "Neutral"],
            values=total_sentiment,
            color=["Negative", "Positive", "Neutral"],
            color_discrete_map={"Negative":"#d62728","Positive":"#2ca02c","Neutral":"#1f77b4"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Daily Mention Volume
    st.subheader("📈 PSV Mention Volume Over Time")
    if "date" in socials_topics.columns:
        socials_topics["date"] = pd.to_datetime(socials_topics["date"], errors="coerce")
        mentions_over_time = socials_topics.groupby("date").size().reset_index(name="mentions")
        fig = px.line(mentions_over_time, x="date", y="mentions", title="Mentions per Day (All Platforms)", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Match sentiment overlay
    if not psv_matches.empty:
        psv_matches["Date"] = pd.to_datetime(psv_matches["Date"], errors="coerce")
        psv_matches["Result_Type"] = psv_matches["Result"].apply(lambda x: "Win" if "W" in str(x) else "Loss/Draw")
        fig = px.scatter(psv_matches, x="Date", y=["Result_Type"], color="Result_Type",
                         title="Recent Match Outcomes", symbol="Result_Type")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2. Player Insights
# =========================================================
elif page == "Player Insights":
    st.header("👟 Player Insights & Value Dynamics")

    # Quadrant: Sentiment vs Market Value
    if not players.empty and not socials_topics.empty:
        st.subheader("💰 Fan Mood vs Player Market Value Matrix")

        # Merge simplified sentiment data by player name
        player_sent = socials_topics.groupby("topic")[["pos_sentiment_(%)"]].mean().reset_index()
        merged = pd.merge(players, player_sent, left_on="name", right_on="topic", how="left")
        merged = merged.dropna(subset=["market_value_eur","pos_sentiment_(%)"])

        fig = px.scatter(
            merged,
            x="pos_sentiment_(%)",
            y="market_value_eur",
            color="position",
            size="performance_score",
            hover_name="name",
            title="Fan Sentiment vs Market Value",
        )
        fig.update_yaxes(title="Market Value (€)")
        fig.update_xaxes(title="Positive Sentiment (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Quadrant meanings:**
        - 🟩 *Core Assets*: High value, high sentiment  
        - 🟨 *Reputation Risks*: High value, low sentiment  
        - 🟦 *Emerging Assets*: Low value, high sentiment  
        - 🟥 *Low Priority*: Low value, low sentiment
        """)

    # Player Mentions Over Time
    if not socials_topics.empty:
        st.subheader("📆 Player Mentions Over Time")
        top_players = socials_topics["topic"].value_counts().head(10).index.tolist()
        filtered = socials_topics[socials_topics["topic"].isin(top_players)].copy()
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")

        mentions = filtered.groupby(["date", "topic"]).size().reset_index(name="mentions")
        fig = px.line(
            mentions,
            x="date", y="mentions", color="topic",
            title="Top 10 Player Mentions Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 3. Content & Engagement
# =========================================================
elif page == "Content & Engagement":
    st.header("📱 Content & Engagement Insights")

    # Platform ROI radar
    st.subheader("⚙️ Engagement Efficiency per Platform")
    if not socials_overview.empty:
        eff = socials_overview.groupby("source")[["likes","num_comments","num_shares"]].sum().reset_index()
        eff["total_eng"] = eff["likes"] + eff["num_comments"] + eff["num_shares"]
        followers_ref = competitors.groupby("source")["followers"].sum().reset_index()
        eff = pd.merge(eff, followers_ref, on="source", how="left")
        eff["efficiency_%"] = (eff["total_eng"] / eff["followers"]) * 100

        fig = px.bar(eff, x="source", y="efficiency_%", color="source", title="Engagement Efficiency (%) by Platform")
        st.plotly_chart(fig, use_container_width=True)

    # Content type performance
    st.subheader("🎬 Content-Type Performance")
    if not insta.empty:
        fig = px.scatter(
            insta, x="likes", y="num_comments", color="content_type",
            size="followers", hover_name="description",
            title="Instagram: Likes vs Comments by Content Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Posting rhythm heatmap
    st.subheader("🕒 Optimal Posting Times (Engagement Heatmap)")
    if "date_posted" in insta.columns:
        insta["date_posted"] = pd.to_datetime(insta["date_posted"], errors="coerce")
        insta["day"] = insta["date_posted"].dt.day_name()
        insta["hour"] = insta["date_posted"].dt.hour
        insta["engagement"] = insta["likes"] + insta["num_comments"]

        pivot = insta.pivot_table(index="day", columns="hour", values="engagement", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap="YlGnBu")
        plt.title("Average Engagement by Posting Day & Hour")
        st.pyplot(fig)

# =========================================================
# 4. Reputation & Sentiment
# =========================================================
elif page == "Reputation & Sentiment":
    st.header("🧭 Reputation & Sentiment Monitoring")

    # Sentiment volatility over time
    st.subheader("📉 Sentiment Volatility Tracker")
    if not socials_topics.empty:
        socials_topics["date"] = pd.to_datetime(socials_topics["date"], errors="coerce")
        daily = socials_topics.groupby("date")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().dropna()
        daily["volatility"] = daily["pos_sentiment_(%)"].rolling(7).std()
        fig = px.line(daily, x=daily.index, y="volatility", title="Sentiment Volatility (7-day Rolling Std)")
        st.plotly_chart(fig, use_container_width=True)

    # Topic cloud placeholder (for next NLP layer)
    st.subheader("💬 Fan Voice: Top Positive vs Negative Topics")
    if not socials_topics.empty:
        top_pos = socials_topics.nlargest(15, "pos_sentiment_(%)")[["topic","pos_sentiment_(%)"]]
        top_neg = socials_topics.nlargest(15, "neg_sentiment_(%)")[["topic","neg_sentiment_(%)"]]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Positive Topics 🌟**")
            st.dataframe(top_pos)
        with col2:
            st.write("**Negative Topics ⚠️**")
            st.dataframe(top_neg)

# =========================================================
# 5. Sponsorship & Benchmarking
# =========================================================
elif page == "Sponsorship & Benchmarking":
    st.header("🤝 Sponsorship & Competitive Benchmarking")

    # Club followers comparison
    if not competitors.empty:
        st.subheader("📈 Followers by Club & Platform")
        fig = px.bar(competitors, x="club", y="followers", color="source", barmode="group",
                     title="Club Followers per Platform")
        st.plotly_chart(fig, use_container_width=True)

    # Sponsor mentions sentiment (from news)
    st.subheader("🏢 Sponsor Visibility & Sentiment")
    if not news.empty:
        sponsor_mentions = news[news["topic"].str.contains("Sponsor", case=False, na=False)]
        if not sponsor_mentions.empty:
            avg_sent = sponsor_mentions.groupby("topic")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
            fig = px.scatter(
                avg_sent,
                x="pos_sentiment_(%)", y="neg_sentiment_(%)",
                color="topic", size_max=15,
                title="Sponsor Mentions: Positive vs Negative Sentiment"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("© PSV Data Intelligence | Built with Streamlit & Seaborn/Plotly")

