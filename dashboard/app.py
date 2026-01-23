import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# API base URL (change if deployed elsewhere)
API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Passaic River Forecast",layout="wide")
st.title("Passaic River Water Level Forecast - Jersey City")
st.markdown("Live data from USGS + LSTM forecast for next 3 hours")

# Fetch current level & forecast
if st.button("Refresh Forecast", type="primary"):
    with st.spinner("Fetching latest forecast..."):
        try:
            # Get auto forecast
            response = requests.get(f"{API_BASE}/forecast-auto")
            response.raise_for_status()
            data = response.json()

            current = data["current_gage_height_ft"]
            forecast = data["forecast_next_12_steps_ft"]

            # Create future timestamps
            now = datetime.now()
            future_times = [now + timedelta(minutes=15 * i) for i in range(1, 13)]

            # Display current
            st.metric("Current Gage Height", f"{current:.2f} ft", delta=None)

            # Forecast table
            df = pd.DataFrame({
                "Time": future_times,
                "Forecast (ft)": forecast
            })
            st.subheader("Forecast for Next 3 Hours")
            st.dataframe(df.style.format({"Forecast (ft)": "{:.2f}"}))

            # Plot
            fig = px.line(
                df, x="Time", y="Forecast (ft)",
                title="Predicted Water Level Trend",
                markers=True
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Gage Height (feet)")
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
        except Exception as e:
            st.error(f"Error processing forecast: {e}")

else:
    st.info("Click 'Refresh Forecast' to load latest data.")