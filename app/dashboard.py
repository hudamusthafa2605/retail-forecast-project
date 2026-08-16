import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load the saved predictions/anomalies data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'test_with_predictions.csv'))
df['Date'] = pd.to_datetime(df['Date'])
st.set_page_config(page_title="Retail Demand Forecast Dashboard", layout="wide")
st.title("📈 Retail Demand Forecasting & Anomaly Detection")
st.write("Explore predicted vs actual sales, and flagged anomalies, per store.")
store_list = sorted(df['Store'].unique())
selected_store = st.selectbox("Select a store", store_list)

store_df = df[df['Store'] == selected_store].sort_values('Date')
col1, col2, col3 = st.columns(3)

mae = (store_df['Sales'] - store_df['predicted']).abs().mean()
num_anomalies = (store_df['z_score'].abs() > 2.5).sum()

col1.metric("Mean Absolute Error", f"{mae:.1f}")
col2.metric("Anomalies Detected", int(num_anomalies))
col3.metric("Days Shown", len(store_df))
fig = go.Figure()

# Actual sales line
fig.add_trace(go.Scatter(
    x=store_df['Date'], y=store_df['Sales'],
    mode='lines', name='Actual Sales', line=dict(color='blue')
))

# Predicted sales line
fig.add_trace(go.Scatter(
    x=store_df['Date'], y=store_df['predicted'],
    mode='lines', name='Predicted Sales', line=dict(color='orange', dash='dash')
))

# Anomaly points
anomaly_points = store_df[store_df['z_score'].abs() > 2.5]
fig.add_trace(go.Scatter(
    x=anomaly_points['Date'], y=anomaly_points['Sales'],
    mode='markers', name='Anomaly',
    marker=dict(color='red', size=10, symbol='circle')
))

fig.update_layout(title=f"Store {selected_store}: Actual vs Predicted Sales",
                   xaxis_title="Date", yaxis_title="Sales")

st.plotly_chart(fig, use_container_width=True)
st.subheader("Flagged Anomalies")

if len(anomaly_points) == 0:
    st.write("No anomalies detected for this store in the selected period.")
else:
    display_cols = ['Date', 'Sales', 'predicted', 'z_score', 'Promo']
    st.dataframe(
        anomaly_points[display_cols].sort_values('Date', ascending=False),
        use_container_width=True
    )