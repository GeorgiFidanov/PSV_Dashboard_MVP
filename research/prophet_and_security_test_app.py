import streamlit as st
import pandas as pd
from prophet import Prophet

# -----------------------------
# 🔐 SIMPLE PASSWORD PROTECTION
# -----------------------------
st.set_page_config(page_title="Forecasting App", page_icon="📈", layout="centered")

st.title("🔐 Secure Forecasting Dashboard")

PASSWORD = st.secrets["password"]

password = st.text_input("Enter password to access:", type="password")

if password != PASSWORD:
    st.warning("Please enter the correct password to continue.")
    st.stop()
else:
    st.success("Access granted! Welcome 👋")

# -----------------------------
# 📊 PROPHET FORECASTING SECTION
# -----------------------------
st.header("📈 Prophet Forecasting")

st.write("""
Upload a CSV file with **two columns**:
- `ds`: Date (in YYYY-MM-DD format)
- `y`: Numeric value to forecast
""")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.subheader("📄 Data Preview")
        st.write(df.head())

        # Check required columns
        if "ds" not in df.columns or "y" not in df.columns:
            st.error("Your CSV must have columns named 'ds' and 'y'.")
            st.stop()

        # Ensure date type
        df["ds"] = pd.to_datetime(df["ds"])

        # Model training
        with st.spinner("Training Prophet model..."):
            model = Prophet()
            model.fit(df)

        # Forecast
        periods = st.slider("Forecast Days into the Future:", 7, 365, 30)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Display results
        st.subheader("🔮 Forecast Results")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

        # Plot chart
        st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

        # Optional: download forecast
        csv = forecast.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV file to get started.")
