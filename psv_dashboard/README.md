# PSV Dashboard - Marketing Intelligence Platform

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Pages & Features](#pages--features)
- [Visualizations & Design Choices](#visualizations--design-choices)
- [Data Sources](#data-sources)
- [Technical Implementation](#technical-implementation)
- [Installation & Setup](#installation--setup)

---

## Overview

The **PSV Unified Marketing Insights Dashboard** is a comprehensive analytics platform designed to provide real-time marketing intelligence for PSV Eindhoven. The dashboard aggregates data from multiple sources including social media platforms, news outlets, transfer market data, and fan engagement metrics to deliver actionable insights for marketing, sponsorship, and brand management decisions.

### Key Capabilities
- **Player Performance Tracking**: Monitor player popularity, market value, and fan sentiment
- **Sentiment Analysis**: Track positive, negative, and neutral sentiment across social media and news
- **Topic Trending**: Identify and analyze trending discussion topics
- **Sponsorship ROI**: Evaluate sponsor performance and campaign effectiveness
- **Reputation Management**: Monitor brand perception across regions and media sources
- **Data Export**: Export insights to Excel and PDF formats

---

## Architecture

### Application Structure
```
psv_dashboard/
├── app.py                 # Main dashboard entry point
├── utils.py               # Shared utility functions
├── pages/                 # Multi-page Streamlit application
│   ├── 1_Player_Index.py
│   ├── 2_Topics.py
│   ├── 3_Player_Comparison.py
│   ├── 4_Sponsorship.py
│   └── 5_Reputation.py
├── data/
│   └── cleaned_final/     # Processed CSV data files
└── requirements.txt       # Python dependencies
```

### Technology Stack
- **Framework**: Streamlit (v1.37.0+)
- **Visualization**: Plotly Express
- **Data Processing**: Pandas, NumPy
- **Export Formats**: Excel (openpyxl), PDF (reportlab)

---

## Pages & Features

### Main Dashboard (`app.py`)

The main landing page provides a high-level overview of club-wide metrics and key insights.

#### Features:
1. **Overall Club Sentiment**
   - Pie chart visualization showing the distribution of positive, negative, and neutral sentiment
   - Aggregates sentiment from both social media and news sources
   - Color-coded: Green (positive), Red (negative), Blue (neutral)

2. **Player Index Summary**
   - Top 10 players ranked by popularity index
   - Horizontal bar chart with Viridis color scale
   - Interactive data table showing top 15 players
   - Export functionality (Excel and PDF)

3. **Trending Topics**
   - Top 10 most mentioned topics across social platforms
   - Horizontal bar chart with orange color gradient
   - Sorted by mention frequency

4. **Sidebar Filters**
   - Period selection: Last 7 days, Last 30 days, Season
   - Platform filtering: Facebook, Instagram, TikTok, YouTube, Twitter/X
   - Data freshness indicator showing last update timestamp

#### Design Rationale:
- **Wide Layout**: Utilizes Streamlit's wide layout mode for better use of screen real estate
- **Color Coding**: Consistent color scheme for sentiment (green/red/blue) aids quick interpretation
- **Progressive Disclosure**: Summary charts on main page, detailed views in dedicated pages

---

### Player Index Insights (`pages/1_Player_Index.py`)

Deep dive into individual player performance metrics and market value correlations.

#### Features:
1. **Player Search**
   - Text input for filtering players by name
   - Case-insensitive search functionality

2. **Top Players Visualization**
   - Top 20 players displayed in horizontal bar chart
   - Viridis color scale for visual appeal and accessibility
   - Interactive hover tooltips

3. **Market Value Correlation**
   - Scatter plot comparing market value (EUR) vs. Player Popularity Index
   - Color-coded by player position
   - Bubble size represents market value magnitude
   - Designed for sponsor/marketing team insights

4. **Data Export**
   - Excel export with full player index data
   - PDF export with formatted report

#### Design Rationale:
- **Scatter Plot for Correlation**: Best visualization type for identifying relationships between two continuous variables
- **Position Color Coding**: Helps identify patterns by player role (forward, midfielder, defender, etc.)
- **Size Encoding**: Market value represented by bubble size for immediate visual impact

---

### Topics Dashboard (`pages/2_Topics.py`)

Monitor trending discussion themes, engagement metrics, and sentiment distribution.

#### Features:
1. **Top Topics by Mentions**
   - Horizontal bar chart of top 10 topics
   - Color intensity represents engagement level (orange gradient)
   - Sorted by mention frequency

2. **Topic Sentiment Overview**
   - Scatter plot with positive sentiment (%) on x-axis, negative sentiment (%) on y-axis
   - Bubble size represents engagement volume
   - Color-coded by topic for easy identification
   - Helps identify topics with high positive sentiment and low negative sentiment

3. **Full Data Table**
   - Complete topics dataset with all metrics

#### Design Rationale:
- **Sentiment Quadrant Analysis**: Scatter plot allows identification of topics in different sentiment quadrants
- **Engagement as Size**: Visual weight indicates discussion volume
- **Orange Color Scale**: Warm colors (oranges) for engagement metrics create visual interest

---

### Player Comparison (`pages/3_Player_Comparison.py`)

Side-by-side comparison tool for evaluating two players across multiple dimensions.

#### Features:
1. **Player Selection**
   - Two dropdown selectors for choosing players to compare
   - Dynamic filtering based on available data

2. **Market Value vs Performance**
   - Scatter plot showing market value against performance score
   - Color-coded by player name
   - Bubble size represents mention frequency
   - Positive sentiment percentage displayed as text annotation

3. **Goals & Assists Comparison**
   - Grouped bar chart comparing goals and assists
   - Side-by-side bars for easy comparison
   - Color-coded by player

4. **Detailed Stats Table**
   - Full comparison data in tabular format

#### Design Rationale:
- **Grouped Bar Charts**: Ideal for comparing categorical metrics (goals vs assists) between two entities
- **Multi-dimensional Scatter**: Combines market value, performance, sentiment, and mentions in one view
- **Interactive Selection**: Dropdowns provide user control while maintaining simplicity

---

### Sponsorship Performance (`pages/4_Sponsorship.py`)

Analyze sponsor ROI, campaign reach, and engagement rates across platforms.

#### Features:
1. **ROI by Sponsor**
   - Vertical bar chart showing Return on Investment percentage
   - Green color gradient (darker = higher ROI)
   - Sorted by ROI value

2. **Reach vs Engagement Rate**
   - Scatter plot analyzing campaign effectiveness
   - X-axis: Reach (audience size)
   - Y-axis: Engagement Rate (interaction percentage)
   - Color-coded by platform
   - Bubble size represents mention frequency

3. **Sponsorship Data Table**
   - Complete dataset with all sponsor metrics

#### Design Rationale:
- **Green Color Scale**: Traditional association with positive financial metrics
- **Reach vs Engagement**: Identifies high-performing campaigns (high reach + high engagement)
- **Platform Differentiation**: Color coding helps identify which platforms drive best results

---

### Club Reputation Insights (`pages/5_Reputation.py`)

Track PSV's brand perception across different regions and media sources.

#### Features:
1. **Reputation Score by Region**
   - Horizontal bar chart showing reputation scores across regions
   - Blue color gradient (darker = higher score)
   - Sorted by reputation score

2. **Sentiment per Source**
   - Scatter plot showing positive vs negative sentiment
   - Color-coded by media source
   - Bubble size represents mention volume
   - Hover shows region information

3. **Reputation Data Table**
   - Full dataset with regional and source-level metrics

#### Design Rationale:
- **Regional Analysis**: Horizontal bars allow easy comparison across multiple regions
- **Source Attribution**: Helps identify which media sources drive positive/negative sentiment
- **Blue Color Scheme**: Professional, trustworthy color for reputation metrics

---

## Visualizations & Design Choices

### Color Schemes

The dashboard employs strategic color choices for different metric types:

1. **Sentiment Colors**:
   - 🟢 Green (`#2ca02c`): Positive sentiment
   - 🔴 Red (`#d62728`): Negative sentiment
   - 🔵 Blue (`#1f77b4`): Neutral sentiment
   - **Rationale**: Universal color associations (green = good, red = bad) improve intuitive understanding

2. **Continuous Scales**:
   - **Viridis**: Used for Player Index (perceptually uniform, colorblind-friendly)
   - **Oranges**: Used for engagement/mentions (warm, attention-grabbing)
   - **Greens**: Used for ROI/financial metrics (positive associations)
   - **Blues**: Used for reputation scores (professional, trustworthy)

### Chart Types Selection

1. **Bar Charts** (Horizontal & Vertical):
   - **Use Case**: Ranking, comparison of discrete categories
   - **Examples**: Top players, top topics, ROI by sponsor
   - **Rationale**: Easy to read, compare values, and identify leaders

2. **Pie Charts**:
   - **Use Case**: Composition/proportion of whole
   - **Example**: Overall club sentiment distribution
   - **Rationale**: Intuitive for showing parts of a whole (100% sentiment split)

3. **Scatter Plots**:
   - **Use Case**: Relationship analysis, multi-dimensional data
   - **Examples**: Market value vs popularity, reach vs engagement, sentiment distribution
   - **Rationale**: Reveals correlations, clusters, and outliers

4. **Grouped Bar Charts**:
   - **Use Case**: Comparing multiple metrics across categories
   - **Example**: Goals & assists comparison
   - **Rationale**: Side-by-side comparison of related metrics

### Interactive Elements

- **Hover Tooltips**: All Plotly charts include interactive hover information
- **Responsive Design**: Charts use `use_container_width=True` for adaptive sizing
- **Filtering**: Sidebar filters and search inputs for data exploration
- **Export Options**: Download buttons for Excel and PDF formats

### Layout & Navigation

- **Multi-page Architecture**: Streamlit's native page routing (files in `pages/` directory)
- **Wide Layout**: Maximizes screen space for data visualization
- **Consistent Header**: Title and caption on each page for context
- **Sidebar Filters**: Persistent filtering options on main dashboard

---

## Data Sources

The dashboard processes cleaned CSV files from the `data/cleaned_final/` directory:

### Core Datasets:
- **transfermarket**: Player market values and transfer information
- **socials_topics**: Social media topic analysis with sentiment
- **news_topics**: News article topic analysis with sentiment
- **socials_overview**: Social media post engagement metrics
- **topics**: Aggregated topic data
- **player_comparison**: Player performance and comparison metrics
- **sponsorship**: Sponsor ROI and campaign data
- **reputation**: Regional and source-based reputation scores

### Data Processing:
- Automatic date parsing for columns: `date`, `date_posted`, `create_time`, `Date`
- Error handling for missing or malformed files
- Caching via `@st.cache_data` for performance optimization

---

## Technical Implementation

### Key Functions (`utils.py`)

1. **`load_all_cleaned()`**:
   - Loads all CSV files from the cleaned data directory
   - Automatically parses date columns
   - Returns dictionary of DataFrames keyed by filename
   - Includes error handling and caching

2. **`compute_player_index()`**:
   - Calculates unified player popularity index
   - Combines metrics from social media, news, and engagement data
   - Formula: `(mentions_rank * 0.4) + (sentiment_rank * 0.4) + (engagement_rank * 0.2) * 100`
   - Handles missing data gracefully

3. **`safe_num()`**:
   - Safely converts columns to numeric, handling errors
   - Returns 0 for missing or invalid values

4. **`export_excel()`**:
   - Generates Excel file from DataFrame
   - Uses openpyxl for compatibility

5. **`export_pdf()`**:
   - Creates PDF report from DataFrame
   - Uses reportlab with A4 page size
   - Limits to top 25 rows for readability

6. **`data_freshness_label()`**:
   - Determines most recent file modification time
   - Formats as human-readable timestamp

### Performance Optimizations

- **Streamlit Caching**: `@st.cache_data` decorator prevents redundant data loading
- **Lazy Loading**: Data loaded only when needed per page
- **Efficient Aggregations**: Groupby operations for topic/player summaries

### Error Handling

- Graceful degradation when data files are missing
- Warning messages for unloadable files
- Empty DataFrame returns instead of crashes
- User-friendly info messages for missing data

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd psv_dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are present**:
   - Place cleaned CSV files in `data/cleaned_final/` directory
   - Files should follow naming convention: `{dataset_name}_cleaned.csv`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**:
   - Open browser to `http://localhost:8501`
   - Navigate between pages using the sidebar

### Dependencies

- `streamlit>=1.37.0`: Web framework
- `pandas>=2.0.0`: Data manipulation
- `plotly>=5.20.0`: Interactive visualizations
- `numpy>=1.25.0`: Numerical operations
- `openpyxl>=3.1.3`: Excel export
- `reportlab`: PDF export
- `prophet>=1.1.5`: Time series forecasting (if needed)
- `matplotlib>=3.7.0`: Additional plotting (if needed)

---

## Design Philosophy

### User-Centric Approach
- **Progressive Disclosure**: Overview on main page, details in dedicated pages
- **Intuitive Navigation**: Clear page titles and consistent structure
- **Actionable Insights**: Visualizations designed to support decision-making

### Data Visualization Best Practices
- **Color Accessibility**: Viridis and other colorblind-friendly palettes
- **Chart Type Selection**: Matched to data type and analysis goal
- **Interactive Elements**: Hover tooltips and responsive design
- **Export Capabilities**: Multiple formats for different use cases

### Maintainability
- **Modular Code**: Shared utilities in `utils.py`
- **Consistent Patterns**: Similar structure across pages
- **Error Handling**: Graceful degradation for missing data
- **Documentation**: Clear comments and function docstrings

---

## Future Enhancements

Potential improvements for future versions:
- Real-time data updates via API integration
- Advanced filtering and date range selection
- Customizable dashboard layouts
- User authentication and role-based access
- Automated report generation and scheduling
- Additional visualization types (heatmaps, time series)
- Mobile-responsive design optimizations

---

## Version Information

**Current Version**: MVP v4.0

**Last Updated**: 2025

---

## License & Credits

© PSV Data Intelligence | MVP v4.0 — Full export & sentiment-ready architecture - Georgi F

