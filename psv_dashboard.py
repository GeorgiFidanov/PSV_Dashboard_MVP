# ==========================================
# ⚽ PSV Unified Marketing Insights Dashboard — v2
# ==========================================
# Author: Georgi
# Description:
# - Autoloads all 22 cleaned datasets from /cleaned_final
# - Role-based interactive dashboards for Emma, Jeroen, Sanne
# - Global time & platform filters, Player Index, benchmarking,
#   sentiment alerting, keyword drill-down, and exports
# ==========================================

import os
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit & Theme
# -----------------------------
st.set_page_config(page_title="PSV Marketing Intelligence", layout="wide")
st.title("⚽ PSV Unified Marketing Insights Dashboard — v2")
st.caption("Interactive decision hub for PSV’s Marketing Department (Social • Sponsorship • PR)")

sns.set_palette("Set2")
plt.style.use("seaborn-v0_8")

DATA_PATH = "cleaned_final"  # confirmed by you

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all_cleaned(data_path: str) -> dict:
    """Autoload all *_cleaned.csv files into a dict keyed by filename stem."""
    frames = {}
    if not os.path.isdir(data_path):
        return frames

    for fname in os.listdir(data_path):
        if not fname.endswith("_cleaned.csv"):
            continue
        key = fname.replace("_cleaned.csv", "")
        fpath = os.path.join(data_path, fname)
        try:
            df = pd.read_csv(fpath)
            # Normalize common date columns
            for col in ["date", "date_posted", "create_time", "Date", "post_date"]:
                if col in df.columns:
                # Parse flexibly, coerce errors, strip any timezone safely
                    df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                    if hasattr(df[col].dt, "tz"):
                        df[col] = df[col].dt.tz_localize(None)
            # Normalize platform labels if present
            if "platform" in df.columns:
                df["platform"] = df["platform"].astype(str).str.title()
            frames[key] = df
        except Exception as e:
            frames[key] = pd.DataFrame()
            st.warning(f"Could not load {fname}: {e}")

    return frames


def first_existing(frames: dict, keys: list[str]) -> pd.DataFrame:
    """Return first non-empty dataframe found for a list of candidate keys."""
    for k in keys:
        df = frames.get(k, pd.DataFrame())
        if not df.empty:
            return df
    return pd.DataFrame()


def platform_name_from_key(key: str) -> str:
    key = key.lower()
    if "facebook" in key: return "Facebook"
    if "instagram" in key: return "Instagram"
    if "tiktok" in key: return "TikTok"
    if "youtube" in key: return "YouTube"
    if "twitter" in key: return "Twitter/X"
    return "Unknown"


def safe_num(df, col, default=0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df))


def within_range(series, start, end):
    series = pd.to_datetime(series, errors="coerce", utc=False)
    if hasattr(series.dt, "tz"):
        series = series.dt.tz_localize(None)
    if isinstance(start, datetime):
        start = start.replace(tzinfo=None)
    if isinstance(end, datetime):
        end = end.replace(tzinfo=None)
    mask = pd.Series(True, index=series.index)
    if start is not None:
        mask &= series >= start
    if end is not None:
        mask &= series <= end
    return mask




# -----------------------------
# Load everything
# -----------------------------
frames = load_all_cleaned(DATA_PATH)

# Key domain tables (best-effort; robust to missing files)
players      = first_existing(frames, ["transfermarket"])
news_topics  = first_existing(frames, ["news_topics"])
social_topics= first_existing(frames, ["socials_topics"])
social_over  = first_existing(frames, ["socials_overview"])
matches      = first_existing(frames, ["psv_matches"])
competitors_all = pd.concat(
    [df.assign(source_key=k) for k, df in frames.items() if "competitor" in k and not frames[k].empty],
    ignore_index=True
) if any("competitor" in k for k in frames.keys()) else pd.DataFrame()

# Platform-specific posts/comments
posts_dfs = []
comments_dfs = []
for k, df in frames.items():
    if df.empty: 
        continue
    if "posts" in k and ("overview" in k or k.endswith("posts")):
        p = df.copy()
        p["platform"] = platform_name_from_key(k)
        # Try unify schema
        # Likes
        if "likes" not in p.columns:
            p["likes"] = safe_num(p, "digg_count", 0) + safe_num(p, "like_count", 0)
        # Comments
        if "num_comments" not in p.columns:
            p["num_comments"] = safe_num(p, "comment_count", 0)
        # Shares
        if "num_shares" not in p.columns:
            p["num_shares"] = safe_num(p, "share_count", 0)
        # Date
        if "date_posted" not in p.columns:
            p["date_posted"] = p.get("date", pd.NaT)
        # Views / plays normalization
        if "views" not in p.columns:
            p["views"] = safe_num(p, "play_count", 0) + safe_num(p, "video_view_count", 0)
        posts_dfs.append(p)
    if "comments" in k:
        c = df.copy()
        c["platform"] = platform_name_from_key(k)
        # unify text/date
        text_col = "comment_text" if "comment_text" in c.columns else ("content" if "content" in c.columns else None)
        if text_col:
            c["comment_text_unified"] = c[text_col]
        if "date_posted" not in c.columns:
            c["date_posted"] = c.get("date", pd.NaT)
        comments_dfs.append(c)

posts_all = pd.concat(posts_dfs, ignore_index=True) if posts_dfs else pd.DataFrame()
comments_all = pd.concat(comments_dfs, ignore_index=True) if comments_dfs else pd.DataFrame()

posts_all["platform"] = posts_all["platform"].str.title().replace({
    "Tiktok": "TikTok", "Youtube": "YouTube", "Twitter": "Twitter/X"
})
comments_all["platform"] = comments_all["platform"].str.title().replace({
    "Tiktok": "TikTok", "Youtube": "YouTube", "Twitter": "Twitter/X"
})

# -----------------------------
# Global filters (Sidebar)
# -----------------------------
st.sidebar.title("📊 Controls")

role = st.sidebar.selectbox(
    "Dashboard Mode",
    ["Social (Emma)", "Sponsorship (Jeroen)", "PR (Sanne)"]
)

time_choice = st.sidebar.radio(
    "Time Range",
    ["Last 7 days", "Last 30 days", "Season (All)", "Custom…"],
    index=1
)

custom_start, custom_end = None, None
if time_choice == "Last 7 days":
    custom_start = datetime.now() - timedelta(days=7)
elif time_choice == "Last 30 days":
    custom_start = datetime.now() - timedelta(days=30)
elif time_choice == "Custom…":
    colA, colB = st.sidebar.columns(2)
    custom_start = colA.date_input("Start date", value=datetime.now() - timedelta(days=30))
    custom_end   = colB.date_input("End date", value=datetime.now())
    # cast to datetime
    custom_start = datetime.combine(custom_start, datetime.min.time())
    custom_end   = datetime.combine(custom_end, datetime.max.time())

platforms_available = sorted(
    set(
        ([p for p in posts_all["platform"].dropna().unique()] if not posts_all.empty else []) +
        ([p for p in social_topics["platform"].dropna().unique()] if "platform" in social_topics.columns else [])
    )
)
selected_platforms = st.sidebar.multiselect(
    "Platforms",
    options=platforms_available if platforms_available else ["Facebook", "Instagram", "TikTok", "YouTube", "Twitter/X"],
    default=platforms_available if platforms_available else []
)

# -----------------------------
# Filtered data views
# -----------------------------
def apply_time_platform_filters(df, date_cols: list[str]):
    if df.empty:
        return df
    sub = df.copy()
    # Time window
    date_col = None
    for c in date_cols:
        if c in sub.columns:
            date_col = c
            break
    if date_col:
        if custom_start and custom_end:
            sub = sub[within_range(sub[date_col], custom_start, custom_end)]
        elif custom_start and not custom_end:
            sub = sub[within_range(sub[date_col], custom_start, None)]
    # Platform filter if available
    if selected_platforms and "platform" in sub.columns and len(selected_platforms) > 0:
        sub = sub[sub["platform"].isin(selected_platforms)]
    return sub

social_topics_f = apply_time_platform_filters(social_topics, ["date"])
news_topics_f   = apply_time_platform_filters(news_topics, ["date"])
posts_all_f     = apply_time_platform_filters(posts_all, ["date_posted", "date"])
comments_all_f  = apply_time_platform_filters(comments_all, ["date_posted", "date"])

# -----------------------------
# Core KPIs & Computations
# -----------------------------
def compute_player_index(social_df: pd.DataFrame,
                         news_df: pd.DataFrame,
                         posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent Player Index:
      0.40 * Mentions (ranked)
      0.40 * Positive Sentiment % (ranked)
      0.20 * Engagement (likes+comments+shares; ranked)
    Assumes 'topic' contains player names for most rows.
    """
    if social_df.empty and news_df.empty:
        return pd.DataFrame(columns=["player", "mentions", "pos_sentiment", "engagement", "player_index"])

    # Mentions & sentiment from both sources
    soc_m = social_df.groupby("topic").size().rename("mentions_social")
    soc_s = social_df.groupby("topic")[["pos_sentiment_(%)"]].mean().rename(columns={"pos_sentiment_(%)":"pos_social"})
    new_m = news_df.groupby("topic").size().rename("mentions_news")
    new_s = news_df.groupby("topic")[["pos_sentiment_(%)"]].mean().rename(columns={"pos_sentiment_(%)":"pos_news"})

    merged = pd.concat([soc_m, new_m, soc_s, new_s], axis=1).fillna(0.0)
    merged["mentions"] = merged["mentions_social"] + merged["mentions_news"]
    # sentiment as weighted blend (social:news = 2:1 to emphasize fan voice)
    merged["pos_sentiment"] = np.where(
        (merged["pos_social"] > 0) | (merged["pos_news"] > 0),
        (merged["pos_social"] * 2 + merged["pos_news"]) / ( (merged["pos_social"]>0).astype(int)*2 + (merged["pos_news"]>0).astype(int) ).replace(0,3),
        0
    )

    # Engagement from posts_df (not by player; used as global scaler)
    if not posts_df.empty:
        posts_df = posts_df.copy()
        posts_df["eng_total"] = safe_num(posts_df, "likes", 0) + safe_num(posts_df, "num_comments", 0) + safe_num(posts_df, "num_shares", 0)
        total_eng = posts_df["eng_total"].sum()
        # Distribute engagement proportionally to mentions to avoid random noise
        total_mentions = merged["mentions"].sum()
        if total_mentions > 0:
            merged["engagement"] = merged["mentions"] / total_mentions * total_eng
        else:
            merged["engagement"] = 0
    else:
        merged["engagement"] = 0

    # Ranking-based index (robust to scale)
    r_mentions  = merged["mentions"].rank(pct=True)
    r_pos       = merged["pos_sentiment"].rank(pct=True)
    r_eng       = merged["engagement"].rank(pct=True)

    merged["player_index"] = (0.40 * r_mentions + 0.40 * r_pos + 0.20 * r_eng) * 100
    out = merged.reset_index().rename(columns={"index":"player"})
    return out.sort_values("player_index", ascending=False)


def engagement_efficiency(posts_df: pd.DataFrame, competitors_df: pd.DataFrame) -> pd.DataFrame:
    """Engagement per follower by platform."""
    if posts_df.empty or competitors_df.empty:
        return pd.DataFrame()
    posts = posts_df.copy()
    posts["eng_total"] = safe_num(posts, "likes", 0) + safe_num(posts, "num_comments", 0) + safe_num(posts, "num_shares", 0)
    eng = posts.groupby("platform")["eng_total"].sum().reset_index()

    # Followers per platform from competitors (sum PSV only if club column present)
    comp = competitors_df.copy()
    if "club" in comp.columns:
        comp_psv = comp[comp["club"].astype(str).str.upper().str.contains("PSV", na=False)]
        if comp_psv.empty:
            comp_psv = comp
    else:
        comp_psv = comp
    followers = comp_psv.groupby("source")["followers"].sum().reset_index().rename(columns={"source":"platform"})
    followers["followers"] = pd.to_numeric(followers["followers"], errors="coerce").fillna(0)

    merged = pd.merge(eng, followers, on="platform", how="left")
    merged["efficiency_%"] = np.where(merged["followers"] > 0, (merged["eng_total"]/merged["followers"]) * 100, np.nan)
    return merged.sort_values("efficiency_%", ascending=False)


def sentiment_timeline_compare(social_df, news_df, press_df=None):
    parts = []
    if not social_df.empty:
        s = social_df.groupby("date")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
        s["source_type"] = "Social"
        parts.append(s)
    if not news_df.empty:
        n = news_df.groupby("date")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
        n["source_type"] = "News"
        parts.append(n)
    if press_df is not None and not press_df.empty:
        p = press_df.groupby("date")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
        p["source_type"] = "Press"
        parts.append(p)

    if not parts:
        return

    df = pd.concat(parts, ignore_index=True)
    fig = px.line(df, x="date", y="pos_sentiment_(%)", color="source_type",
                  title="Positive Sentiment Over Time by Source", markers=True)
    st.plotly_chart(fig, use_container_width=True)


def sponsorship_from_news(news_df):
    """Sponsor sentiment bubble chart based on topic containing 'sponsor'."""
    if news_df.empty or "topic" not in news_df.columns:
        st.info("No sponsor-related topics available.")
        return
    sponsors = news_df[news_df["topic"].str.contains("sponsor", case=False, na=False)].copy()
    if sponsors.empty:
        st.info("No sponsor-related topics detected in the current time range.")
        return
    agg = sponsors.groupby("topic")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean().reset_index()
    agg["volume"] = sponsors.groupby("topic").size().values
    fig = px.scatter(
        agg, x="pos_sentiment_(%)", y="neg_sentiment_(%)", size="volume", color="topic",
        title="Sponsor Mentions: Sentiment vs Volume", size_max=40
        )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Header KPI chips
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_posts = len(posts_all_f) if not posts_all_f.empty else 0
    st.metric("Total Posts (filtered)", f"{total_posts:,}")
with col2:
    total_comments = len(comments_all_f) if not comments_all_f.empty else 0
    st.metric("Total Comments (filtered)", f"{total_comments:,}")
with col3:
    total_topics = len(social_topics_f) + len(news_topics_f)
    st.metric("Total Topics (social+news)", f"{total_topics:,}")
with col4:
    st.metric("Platforms selected", f"{len(selected_platforms) if selected_platforms else 'All'}")

st.markdown("---")

# =========================================================
# Role Dashboards
# =========================================================

# =============== Emma (Social) ===============
if role.startswith("Social"):
    st.header("📱 Emma — Social Media Manager")

    # 1) Player Index
    st.subheader("🏆 Player Index Ranking (transparent)")
    pi = compute_player_index(social_topics_f, news_topics_f, posts_all_f)
    if pi.empty:
        st.info("Not enough data to compute Player Index in current filters.")
    else:
        top_n = st.slider("Show top N players:", 5, 20, 10)
        fig = px.bar(pi.head(top_n), x="player_index", y="player", orientation="h",
                     title="Top Players by Player Index",
                     color="player_index", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("How Player Index is calculated"):
            st.write("""
**Player Index = 0.40 × Mentions (rank) + 0.40 × Positive Sentiment% (rank) + 0.20 × Engagement (rank)**  
- Mentions: combined social + news.  
- Sentiment: blended (social weighted more than news).  
- Engagement: likes + comments + shares distributed by mentions share.  
All components are **ranked** to make the score scale-robust (0–100).
            """)
        st.dataframe(pi.head(top_n))

        csv = pi.to_csv(index=False).encode("utf-8")
        st.download_button("Download Player Index (CSV)", csv, "player_index.csv", "text/csv")

    # 2) Platform sentiment comparison
    st.subheader("💬 Sentiment per Platform")
    if not social_topics_f.empty and "platform" in social_topics_f.columns:
        plat = social_topics_f.groupby("platform")[["pos_sentiment_(%)","neg_sentiment_(%)","neu_sentiment_(%)"]].mean().reset_index()
        fig = px.bar(plat, x="platform", y=["pos_sentiment_(%)","neg_sentiment_(%)","neu_sentiment_(%)"],
                     barmode="group", title="Average Sentiment by Platform")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No platform-level sentiment available in current filters.")

    # 3) Trending topics by platform
    st.subheader("🔥 Trending Topics by Platform")
    if not social_topics_f.empty:
        top_by_topic = (social_topics_f
                        .groupby(["platform","topic"])
                        .size().reset_index(name="mentions")
                        .sort_values(["platform","mentions"], ascending=[True, False]))
        k = st.slider("Top topics per platform:", 3, 20, 10)
        tops = top_by_topic.groupby("platform").head(k)
        fig = px.bar(tops, x="topic", y="mentions", color="platform", facet_col="platform",
                     title=f"Top {k} Topics per Platform", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No social topics in the current selection.")

    # 4) Content effectiveness & posting rhythm
    st.subheader("🎬 Content-Type Performance & Posting Rhythm")
    if not posts_all_f.empty:
        # Content type (if available on Instagram/YouTube)
        content_like = []
        for plat in ["Instagram", "YouTube", "TikTok", "Facebook"]:
            dfp = posts_all_f[posts_all_f["platform"] == plat].copy()
            if "content_type" in dfp.columns:
                g = dfp.groupby("content_type")[["likes","num_comments","num_shares"]].mean().reset_index()
                g["platform"] = plat
                content_like.append(g)
        if content_like:
            ct = pd.concat(content_like, ignore_index=True)
            ct["avg_eng"] = ct["likes"] + ct["num_comments"] + ct["num_shares"]
            fig = px.bar(ct, x="content_type", y="avg_eng", color="platform",
                         title="Average Engagement by Content Type & Platform", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No content_type available in posts for the current filters.")

        # Posting heatmap (use Instagram if present; else any)
        heat_df = posts_all_f.copy()
        date_col = "date_posted" if "date_posted" in heat_df.columns else ("date" if "date" in heat_df.columns else None)
        if date_col:
            heat_df = heat_df.dropna(subset=[date_col])
            heat_df["day"]  = heat_df[date_col].dt.day_name()
            heat_df["hour"] = heat_df[date_col].dt.hour
            heat_df["eng"]  = safe_num(heat_df, "likes", 0) + safe_num(heat_df, "num_comments", 0) + safe_num(heat_df, "num_shares", 0)
            pivot = heat_df.pivot_table(index="day", columns="hour", values="eng", aggfunc="mean").fillna(0)
            fig2, ax = plt.subplots(figsize=(12,5))
            sns.heatmap(pivot, cmap="YlGnBu")
            ax.set_title("Average Engagement Heatmap (Day x Hour)")
            st.pyplot(fig2)
        else:
            st.info("No date information to compute posting heatmap.")

    # 5) Engagement efficiency per platform
    st.subheader("⚙️ Engagement Efficiency per Platform (Engagement / Followers)")
    eff = engagement_efficiency(posts_all_f, competitors_all)
    if not eff.empty:
        fig = px.bar(eff, x="platform", y="efficiency_%", color="platform",
                     title="Engagement Efficiency (%) by Platform")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(eff)
    else:
        st.info("Efficiency requires posts + competitor follower data.")


# =============== Jeroen (Sponsorship) ===============
elif role.startswith("Sponsorship"):
    st.header("💼 Jeroen — Sponsorship & Player Value")

    # Player Index vs Market Value
    st.subheader("💰 Market Value vs Player Popularity Index")
    pi = compute_player_index(social_topics_f, news_topics_f, posts_all_f)
    if pi.empty or players.empty:
        st.info("Need player & topic data to compute the scatter.")
    else:
        right_key = "player" if "player" in pi.columns else ("topic" if "topic" in pi.columns else "name")
        merged = pd.merge(players.rename(columns=str.lower), pi, left_on="name", right_on=right_key, how="left")

        merged = merged.dropna(subset=["market_value_eur", "player_index"])
        fig = px.scatter(
            merged, x="market_value_eur", y="player_index", color="position",
            size="performance_score", hover_name="name",
            title="Market Value (€) vs Player Popularity Index"
        )
        fig.update_xaxes(title="Market Value (€)")
        fig.update_yaxes(title="Popularity Index (0–100)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(merged[["name","position","market_value_eur","goals","assists","player_index"]]
                     .sort_values("player_index", ascending=False).head(20))

    # Sponsorship: sentiment & volume from news
    st.subheader("🏢 Sponsor Mentions & Sentiment")
    sponsorship_from_news(news_topics_f)

    # Benchmarking vs rivals (all competitor files)
    st.subheader("📊 Club Benchmarking Across Platforms")
    if not competitors_all.empty:
        # normalize platform column
        comp = competitors_all.copy()
        plat_col = "source" if "source" in comp.columns else ("platform" if "platform" in comp.columns else None)
        if plat_col:
            comp["platform"] = comp[plat_col]
        else:
            comp["platform"] = comp["source_key"].str.replace("_competitors", "").str.title()
        comp["club"] = comp.get("club", "Unknown").astype(str)
        comp["followers"] = pd.to_numeric(comp.get("followers", np.nan), errors="coerce")

        # Only keep reasonable data
        comp = comp.dropna(subset=["followers"])
        fig = px.bar(comp, x="club", y="followers", color="platform",
                     title="Followers by Club & Platform", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp.head(500))
    else:
        st.info("No competitor profiles detected.")

    # Export CSV / HTML snapshot
    st.subheader("📤 Export (CSV / HTML)")
    if not pi.empty:
        csv = pi.to_csv(index=False).encode("utf-8")
        st.download_button("Download Player Index (CSV)", data=csv, file_name="player_index.csv", mime="text/csv")

        # Simple HTML summary export (lightweight; no extra deps)
        html = io.StringIO()
        html.write("<h2>PSV Sponsorship Snapshot</h2>")
        html.write("<p>Market Value vs Player Index scatter, sponsor sentiment, and benchmarking tables are available in the dashboard views.</p>")
        html.write("<h3>Top Players by Popularity Index</h3>")
        html.write(pi.head(15).to_html(index=False))
        st.download_button("Download Sponsor Snapshot (HTML)", data=html.getvalue().encode("utf-8"),
                           file_name="psv_sponsorship_snapshot.html", mime="text/html")


# =============== Sanne (PR) ===============
else:
    st.header("🧭 Sanne — PR & Communications")

    # Cross-source sentiment timeline (Social, News, Press)
    st.subheader("📅 Sentiment Over Time (Social vs News vs Press)")
    press_df = first_existing(frames, ["eventregistry_headlines", "google_headlines"])  # press-like sources
    # If these lack sentiment columns, fall back to empty; otherwise timeline works.
    if not press_df.empty and "pos_sentiment_(%)" not in press_df.columns:
        press_df = pd.DataFrame()  # keep logic simple for now
    sentiment_timeline_compare(social_topics_f, news_topics_f, press_df)

    # Reputation Alerts
    st.subheader("⚠️ Reputation Alerts (Negative Sentiment Spikes)")
    if not social_topics_f.empty:
        alerts = (social_topics_f.copy())
        if "neg_sentiment_(%)" in alerts.columns:
            spikes = alerts[alerts["neg_sentiment_(%)"] >= st.slider("Alert threshold (%)", 10, 60, 25)]
            if spikes.empty:
                st.success("No negative sentiment spikes in current filters.")
            else:
                st.error(f"{len(spikes)} records exceed threshold.")
                cols = ["date", "topic", "platform", "neg_sentiment_(%)"]
                st.dataframe(spikes[[c for c in cols if c in spikes.columns]].sort_values("neg_sentiment_(%)", ascending=False))
        else:
            st.info("Negative sentiment column not found.")
    else:
        st.info("No social topics available in current filters.")

    # Social vs News sentiment split per player/topic
    st.subheader("🧩 Social vs News Sentiment (per Topic/Player)")
    if not social_topics_f.empty or not news_topics_f.empty:
        soc = (social_topics_f.groupby("topic")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean()
               .rename(columns=lambda c: f"social_{c}"))
        nws = (news_topics_f.groupby("topic")[["pos_sentiment_(%)","neg_sentiment_(%)"]].mean()
               .rename(columns=lambda c: f"news_{c}"))
        comp = pd.concat([soc, nws], axis=1)
        comp = comp.replace(np.nan, 0).reset_index().rename(columns={"index":"topic"})
        st.dataframe(comp.sort_values("social_pos_sentiment_(%)", ascending=False).head(30))
        if not comp.empty:
            melted = comp.melt(id_vars="topic", value_vars=["social_pos_sentiment_(%)","news_pos_sentiment_(%)"],
                               var_name="source", value_name="pos_sentiment")
            fig = px.bar(melted.sort_values("pos_sentiment", ascending=False).head(60),
                         x="topic", y="pos_sentiment", color="source", title="Positive Sentiment — Social vs News")
            st.plotly_chart(fig, use_container_width=True)

    # Keyword Drill-Down (topics + comments + news)
    st.subheader("🔍 Keyword Drill-Down")
    query = st.text_input("Search in topics/comments/news (e.g., 'Bosz', 'Ajax', 'Sponsor'):")
    if query:
        results = []
        if not social_topics_f.empty:
            tmp = social_topics_f[social_topics_f["topic"].str.contains(query, case=False, na=False)].copy()
            tmp["source_type"] = "Social Topic"
            results.append(tmp)
        if not news_topics_f.empty:
            tmp = news_topics_f[news_topics_f["topic"].str.contains(query, case=False, na=False)].copy()
            tmp["source_type"] = "News Topic"
            results.append(tmp)
        if not comments_all_f.empty and "comment_text_unified" in comments_all_f.columns:
            tmp = comments_all_f[comments_all_f["comment_text_unified"].str.contains(query, case=False, na=False)].copy()
            tmp["source_type"] = "Comment"
            results.append(tmp)
        if results:
            out = pd.concat(results, ignore_index=True)
            keep_cols = [c for c in ["source_type","date","date_posted","platform","topic","comment_text_unified","pos_sentiment_(%)","neg_sentiment_(%)"] if c in out.columns]
            st.dataframe(out[keep_cols].head(500))
        else:
            st.info("No matches in current filters.")

# =========================================================
# Footer & Notes
# =========================================================
st.markdown("---")
st.caption("© PSV Data Intelligence • Built with Streamlit, Plotly, Seaborn • v2")

with st.expander("Data & Feature Notes"):
    st.markdown("""
- **Autoloaded datasets**: any file ending with `_cleaned.csv` in `cleaned_final/`.
- **Filters**: time & platform filters apply across modules dynamically.
- **Player Index**: rank-based weighting of Mentions, Positive Sentiment, and Engagement (see explanation in Emma's view).
- **Engagement Efficiency**: (likes + comments + shares) / followers by platform; followers sourced from competitor profiles.
- **Sponsor Analysis**: based on news topics containing the keyword *sponsor*.
- **Press**: If `eventregistry_headlines_cleaned.csv` or `google_headlines_cleaned.csv` lack sentiment columns, they’re omitted from timeline automatically.
- **Exports**: CSV and a lightweight HTML snapshot are available (no extra dependencies).
""")
