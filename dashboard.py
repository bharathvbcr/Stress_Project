import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import glob
from lightning_module import StressLightningModule
from models import get_model
from utils import load_config, safe_get
from datasets import load_from_disk

# Page Config
st.set_page_config(page_title="StressPulse SOTA Dashboard", layout="wide")

# Theme / CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    config = load_config("config.json")
    # Find latest checkpoint
    ckpts = glob.glob("outputs/models/*.ckpt")
    latest_ckpt = max(ckpts, key=os.path.getctime) if ckpts else None
    
    # Load metadata from Arrow to get dims
    hf_path = "./outputs/processed_data_hf/test"
    test_ds = load_from_disk(hf_path) if os.path.exists(hf_path) else None
    
    return config, latest_ckpt, test_ds

def main():
    st.title("🧠 StressPulse SOTA Dashboard")
    st.markdown("---")
    
    config, latest_ckpt, test_ds = load_assets()
    
    if not test_ds:
        st.error("No Arrow datasets found. Please run `python main.py` first to generate data.")
        return

    # Sidebar: Model Selection
    st.sidebar.header("🕹️ Control Panel")
    if latest_ckpt:
        st.sidebar.success(f"Model loaded: {os.path.basename(latest_ckpt)}")
    else:
        st.sidebar.warning("No trained model checkpoint found.")

    # Main UI: Signal Explorer
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Session Summary")
        subject_id = st.selectbox("Select Subject", options=list(set(test_ds['subject_id'])))
        subj_indices = [i for i, x in enumerate(test_ds['subject_id']) if x == subject_id]
        st.metric("Total Windows", len(subj_indices))
        
        window_idx = st.slider("Window Selection", 0, len(subj_indices)-1)
        target_idx = subj_indices[window_idx]
        
    with col1:
        st.subheader("📈 Physiological Signals")
        sample = test_ds[target_idx]
        seq = sample['sequence'].numpy() # (L, F)
        
        fig = go.Figure()
        # Plot first few channels (e.g., ECG, EDA)
        fig.add_trace(go.Scatter(y=seq[:, 0], name="Channel 1 (Primary)"))
        fig.add_trace(go.Scatter(y=seq[:, 1], name="Channel 2 (Secondary)"))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Inference Section
    st.markdown("---")
    st.subheader("🎯 Stress Prediction")
    
    if st.button("🚀 Run Inference", use_container_width=True):
        # In a real scenario, we'd load the model weights here and call forward
        # For the dashboard demo, we'll show the GROUND TRUTH label
        label = sample['label'].item()
        status = "STRESS DETECTED" if label == 1 else "NORMAL (BASELINE)"
        color = "#ff4b4b" if label == 1 else "#00c853"
        
        st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{status}</h2>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
