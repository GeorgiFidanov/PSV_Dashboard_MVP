import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re

URL = "https://www.supver-psv.nl/en/season/matches.html"
CSV_FILE = "psv_matches_season_25_26.csv"

# --- Helper functions ---

def normalize_match(match: str, result: str):
    """Ensure PSV is always first in the match name, flipping result if needed."""
    if "PSV" in match:
        teams = [t.strip() for t in match.split("-")]
        if len(teams) == 2:
            home, away = teams
            if home != "PSV" and away == "PSV":
                # Swap
                match = f"PSV - {home}"
                # Flip score if available
                if result and re.match(r"\d+\s*-\s*\d+", result):
                    a, b = [int(x) for x in re.findall(r"\d+", result)]
                    result = f"{b} - {a}"
    return match, result


def match_status(result: str):
    """Return W/L/D/TBD based on PSV's result."""
    if not result or not re.match(r"\d+\s*-\s*\d+", result):
        return "TBD"
    psv_goals, opp_goals = [int(x) for x in re.findall(r"\d+", result)]
    if psv_goals > opp_goals:
        return "W"
    elif psv_goals < opp_goals:
        return "L"
    else:
        return "D"


# --- Scrape page ---

response = requests.get(URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

rows = []
for tr in soup.select("table tr"):
    cols = [td.get_text(strip=True) for td in tr.find_all("td")]
    if cols:
        rows.append(cols)

# Drop header if detected
if rows and "Dag" in rows[0][0]:
    rows.pop(0)

# Normalize varying column lengths
normalized = []
for row in rows:
    if len(row) == 5:
        row.append("")  # Add empty result column
    elif len(row) > 6:
        row = row[:6]
    normalized.append(row)

df = pd.DataFrame(normalized, columns=["Day", "Date", "Time", "Match", "Competition_Type", "Result"])

# Clean competition names
df["Competition_Type"] = df["Competition_Type"].str.replace(
    r"(Eredivisie|Champions League|Friendly)\1", r"\1", regex=True
)

# Normalize match order and flip scores when PSV was away
df[["Match", "Result"]] = df.apply(
    lambda row: normalize_match(row["Match"], row["Result"]), axis=1, result_type="expand"
)

# Add match status column
df["Match_Status"] = df["Result"].apply(match_status)

# --- Merge with existing CSV if it exists ---
if os.path.exists(CSV_FILE):
    old_df = pd.read_csv(CSV_FILE)
    # Ensure same structure
    combined = pd.concat([old_df, df]).drop_duplicates(subset=["Date", "Match"], keep="last")
else:
    combined = df

# Save updated dataset
combined.to_csv(CSV_FILE, index=False, encoding="utf-8")

print(f"✅ Updated {CSV_FILE} with {len(combined)} total records.")
print(combined.head(10))
