import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Sales Forecast App", layout="wide")

st.title("📊 Sales Forecast Prediction System")

# Load model and data
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    data['date'] = pd.to_datetime(data['date'])
    return data

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.header("⚙️ Forecast Controls")
    
    months_to_predict = st.number_input(
        "Months to Predict",
        min_value=1,
        max_value=24,
        value=6
    )

    if st.button("Generate Forecast"):
        last_month = (data['date'].dt.year.max() - 2020) * 12 + data['date'].dt.month.max()
        future_months = list(range(last_month + 1, last_month + months_to_predict + 1))

        future_df = pd.DataFrame({'month': future_months})
        predictions = model.predict(future_df)

        # Store in session
        st.session_state['predictions'] = predictions
        st.session_state['future_months'] = future_months

    st.subheader("📈 Data Statistics")

    st.write(f"Total Records: {len(data)}")
    st.write(f"Date Range: {data['date'].min().date()} to {data['date'].max().date()}")

    st.write("**Latest Sales:**")
    st.write(f"Date: {data['date'].iloc[-1].date()}")
    st.write(f"Sales: {data['sales'].iloc[-1]:.2f}")

    st.write(f"Average Sales: {data['sales'].mean():.2f}")
    st.write(f"Min Sales: {data['sales'].min():.2f}")
    st.write(f"Max Sales: {data['sales'].max():.2f}")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.header("📊 Forecast Results")

    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        future_months = st.session_state['future_months']

        results = []

        for i, pred in enumerate(predictions):
            month_num = future_months[i]
            year = 2020 + (month_num - 1) // 12
            month = ((month_num - 1) % 12) + 1
            date_str = f"{year}-{month:02d}-01"
            results.append([date_str, pred])

        result_df = pd.DataFrame(results, columns=["Date", "Predicted Sales"])

        st.dataframe(result_df, use_container_width=True)

        # -------- Plot --------
        fig, ax = plt.subplots()

        ax.plot(data.index, data['sales'], label='Historical Sales')
        ax.scatter(data.index, data['sales'])

        last_index = len(data) - 1
        pred_index = list(range(last_index + 1, last_index + len(predictions) + 1))

        ax.plot(pred_index, predictions, linestyle='--', label='Forecasted Sales')
        ax.scatter(pred_index, predictions)

        ax.axvline(x=last_index + 0.5)

        ax.set_xlabel("Time Period")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Forecast Visualization")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    else:
        st.info("Click 'Generate Forecast' to see results.")
