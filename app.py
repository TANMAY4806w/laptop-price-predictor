import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
df = pickle.load(open("df.pkl", "rb"))
pipe = pickle.load(open("pipe.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    /* General container */
    .main {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        color: #ffffff;
    }

    /* Set background and text color */
    .block-container {
        background-color: #121212;
        padding: 2rem;
        border-radius: 10px;
        color: #ffffff;
    }

    /* Form sections */
    .spec-section {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Make headers white */
    h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }

    /* Improve input contrast */
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #2c2c2c !important;
        color: #ffffff !important;
        border-radius: 8px;
    }

    /* Button style */
    .predict-btn {
        width: 100%;
        margin-top: 1.5rem;
    }

    /* Responsive layout for smaller screens */
    @media screen and (max-width: 768px) {
        .stColumn {
            width: 100% !important;
            display: block;
        }

        .main {
            padding: 1rem;
        }

        .spec-section {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# App header
st.title("💻 Laptop Price Predictor")
st.markdown("Find the perfect laptop at the right price")

# Main form
with st.container():
    st.header("Basic Specifications")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Brand", df["Company"].unique())
        type = st.selectbox("Laptop Type", df["TypeName"].unique())
    with col2:
        ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
        weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

with st.container():
    st.header("Display & Performance")
    col1, col2 = st.columns(2)
    with col1:
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
        ips = st.selectbox("IPS Panel", ["No", "Yes"])
    with col2:
        screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=13.3, step=0.1)
        resolution = st.selectbox("Screen Resolution", [
            "1920x1080", "1366x768", "1600x900", "3840x2160", 
            "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"
        ])

with st.container():
    st.header("Storage & Components")
    col1, col2 = st.columns(2)
    with col1:
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
        ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024])
    with col2:
        cpu = st.selectbox("CPU", df["Cpu brand"].unique())
        gpu = st.selectbox("GPU", df["Gpu_brand"].unique())
    os = st.selectbox("Operating System", df["os"].unique())

# Prediction button
if st.button("Predict Price", key="predict", help="Click to estimate laptop price"):
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'IPS Panel': ips,
        'PPI': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu_brand': gpu,
        'os': os
    }])

    predicted_price = np.exp(pipe.predict(query)[0])
    st.success(f"### Predicted Price: ₹{int(predicted_price):,}")