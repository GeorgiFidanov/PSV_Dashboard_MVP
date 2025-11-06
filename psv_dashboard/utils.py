# ==========================================
# utils.py — PSV Dashboard Shared Functions
# ==========================================
import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from io import BytesIO
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@st.cache_data(show_spinner=False)
def load_all_cleaned(data_path="data/cleaned_final"):
    frames = {}
    if not os.path.isdir(data_path):
        st.error("Data folder not found.")
        return frames
    for fname in os.listdir(data_path):
        if fname.endswith(".csv"):
            key = fname.replace("_cleaned.csv", "").replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(data_path, fname))
                for c in ["date", "date_posted", "create_time", "Date"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                frames[key] = df
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
                frames[key] = pd.DataFrame()
    return frames

def safe_num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0

def compute_player_index(players, social, news, posts):
    """Unified Player Index across sources."""
    if social.empty and news.empty:
        return pd.DataFrame()
    soc = social.groupby("topic")[["pos_sentiment_(%)"]].mean()
    nws = news.groupby("topic")[["pos_sentiment_(%)"]].mean()
    sent = soc.add(nws, fill_value=0) / 2
    mentions = social.groupby("topic").size().rename("mentions")
    merged = pd.concat([sent, mentions], axis=1).fillna(0)
    merged["engagement"] = 0
    if not posts.empty:
        posts["eng"] = safe_num(posts, "likes") + safe_num(posts, "num_comments")
        merged["engagement"] = merged["mentions"] / merged["mentions"].sum() * posts["eng"].sum()
    merged["player_index"] = (
        merged["mentions"].rank(pct=True) * 0.4 +
        merged["pos_sentiment_(%)"].rank(pct=True) * 0.4 +
        merged["engagement"].rank(pct=True) * 0.2
    ) * 100
    
    df_out = merged.reset_index()
    if "topic" in df_out.columns:
        df_out = df_out.rename(columns={"topic": "player"})
    return df_out.sort_values("player_index", ascending=False)

def export_excel(df):
    wb = Workbook()
    ws = wb.active
    for i, col in enumerate(df.columns, start=1):
        ws.cell(1, i, col)
    for r in df.itertuples(index=False):
        ws.append(r)
    out = BytesIO()
    wb.save(out)
    return out.getvalue()

def export_pdf(df, title="PSV Report"):
    out = BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height-40, title)
    c.setFont("Helvetica", 10)
    y = height-70
    for i, row in df.head(25).iterrows():
        line = ", ".join([f"{k}: {v}" for k, v in row.items()])
        c.drawString(40, y, line[:110])
        y -= 14
        if y < 60:
            c.showPage(); y = height-70
    c.save()
    return out.getvalue()

def data_freshness_label(path="data/cleaned_final"):
    if not os.path.exists(path): return "No data found"
    times = [os.path.getmtime(os.path.join(path,f)) for f in os.listdir(path) if f.endswith(".csv")]
    if not times: return "No CSVs found"
    latest = datetime.fromtimestamp(max(times))
    return latest.strftime("Last updated: %Y-%m-%d %H:%M")
