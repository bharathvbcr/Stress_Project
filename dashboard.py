import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import glob
from pathlib import Path
from utils import configure_torch_runtime, infer_model_feature_dims, load_config, safe_get

REPO_ROOT = Path(__file__).resolve().parent
LOGO_PATH = REPO_ROOT / "docs" / "assets" / "logo.png"

# Page Config
st.set_page_config(
    page_title="StressProject Dashboard",
    page_icon=str(LOGO_PATH),
    layout="wide",
)

# Theme / CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161b22; }
    </style>
""", unsafe_allow_html=True)


def _select_checkpoint(model_dir: str = "outputs/models", prefer_mlx: bool = False) -> str | None:
    pattern = "*_mlx.npz" if prefer_mlx else "*.ckpt"
    ckpts = glob.glob(os.path.join(model_dir, pattern))
    if not ckpts and prefer_mlx:
        ckpts = glob.glob(os.path.join(model_dir, "*.ckpt"))
    return max(ckpts, key=os.path.getctime) if ckpts else None


@st.cache_resource
def load_predictor(backend: str):
    """Load model for dashboard inference (torch, mlx, or mlx_hybrid)."""
    config = load_config("config.json")
    if not config:
        return None, "Config not found"

    runtime_caps = configure_torch_runtime()
    runtime_caps["backend_request"] = backend
    seq_dim, static_dim = infer_model_feature_dims(config)
    model_type = safe_get(config, ["model_config", "type"], "LSTM").upper()

    from mlx_backend import is_mlx_available, resolve_inference_backend, resolve_timesfm_hybrid_enabled

    timesfm_hybrid = model_type == "TIMESFM" and resolve_timesfm_hybrid_enabled(backend)
    use_mlx = timesfm_hybrid or (
        is_mlx_available()
        and resolve_inference_backend(model_type, None, requested=backend) == "mlx"
    )
    ckpt = _select_checkpoint(prefer_mlx=use_mlx)
    if not ckpt:
        return None, "No checkpoint in outputs/models"

    if timesfm_hybrid:
        from mlx_hybrid import build_timesfm_hybrid

        torch_model, mlx_head, _ = build_timesfm_hybrid(
            config, ckpt, seq_dim, static_dim, device=torch.device("cpu")
        )
        return {
            "backend": "mlx_hybrid",
            "torch_model": torch_model,
            "mlx_head": mlx_head,
            "config": config,
        }, None

    if use_mlx and ckpt.endswith(".npz"):
        from mlx_backend import configure_mlx_runtime
        from mlx_models import get_mlx_model, load_mlx_checkpoint

        configure_mlx_runtime()
        model = get_mlx_model(config, seq_dim, static_dim)
        load_mlx_checkpoint(model, ckpt)
        model.eval()
        return {"backend": "mlx", "model": model, "config": config}, None

    from models import get_model
    from evaluate_checkpoint import load_weights_into_model

    model = get_model(config, seq_dim, static_dim)
    load_weights_into_model(model, Path(ckpt), config=config)
    model.eval()
    return {"backend": "torch", "model": model.to("cpu"), "config": config}, None


def predict_stress(predictor: dict, sequence: np.ndarray, static: np.ndarray | None) -> float:
    """Return stress probability for one window."""
    import mlx.core as mx

    seq_t = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    static_t = None
    if static is not None and static.size > 0:
        static_t = torch.tensor(static, dtype=torch.float32).unsqueeze(0)

    backend = predictor["backend"]
    if backend == "mlx_hybrid":
        from mlx_hybrid import timesfm_hybrid_predict

        return timesfm_hybrid_predict(
            predictor["torch_model"],
            predictor["mlx_head"],
            seq_t,
            static_t,
        )

    if backend == "mlx":
        seq_m = mx.array(seq_t.numpy())
        static_m = mx.array(static_t.numpy()) if static_t is not None else None
        logits = predictor["model"](seq_m, static_m)
        return float(mx.sigmoid(logits.reshape(-1))[0].item())

    with torch.no_grad():
        logits = predictor["model"](seq_t, static_t)
        return float(torch.sigmoid(logits.reshape(-1))[0].item())


@st.cache_resource
def load_assets():
    config = load_config("config.json")
    hf_path = "./outputs/processed_data_hf/test"
    from datasets import load_from_disk

    test_ds = load_from_disk(hf_path) if os.path.exists(hf_path) else None
    return config, test_ds

def render_logo_header():
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.title("StressProject")

def main():
    render_logo_header()
    st.title("StressProject Dashboard")
    st.markdown("---")
    
    config, test_ds = load_assets()
    backend = st.sidebar.selectbox(
        "Inference backend",
        options=["auto", "mlx", "torch"],
        index=0,
        help="MLX uses Apple Silicon native weights; TimesFM uses hybrid (PyTorch backbone + MLX head).",
    )
    predictor, pred_err = load_predictor(backend)
    if pred_err:
        st.sidebar.warning(pred_err)
    elif predictor:
        st.sidebar.caption(f"Loaded: {predictor['backend']}")
    
    # Navigation Tabs
    tab1, tab2 = st.tabs(["📊 Data & Inference", "🏎️ Performance Diagnostics"])

    with tab1:
        if not test_ds:
            st.error("No Arrow datasets found. Please run `python main.py` first to generate data.")
        else:
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
                fig.add_trace(go.Scatter(y=seq[:, 0], name="Channel 1"))
                fig.add_trace(go.Scatter(y=seq[:, 1], name="Channel 2"))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Inference Section
            st.markdown("---")
            st.subheader("🎯 Stress Prediction")

            if st.button("🚀 Run Inference", use_container_width=True):
                if not predictor:
                    st.error("No model loaded. Train a model or check outputs/models.")
                else:
                    static_np = sample["static"].numpy() if "static" in sample else None
                    prob = predict_stress(predictor, seq, static_np)
                    label = sample["label"].item()
                    pred_stress = prob >= 0.5
                    status = "STRESS DETECTED" if pred_stress else "NORMAL (BASELINE)"
                    color = "#ff4b4b" if pred_stress else "#00c853"
                    st.metric("Predicted probability", f"{prob:.3f}")
                    st.caption(f"Ground truth label: {int(label)}")
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: white; margin: 0;">{status}</h2>
                        </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("🏁 Performance Laboratory")
        st.info(
            "PyTorch static quantization (`quantize_cpu.py`) applies to torch checkpoints only. "
            "MLX-native `.mlx.npz` models and TimesFM hybrid heads use `mlx_benchmark.py` instead."
        )

        # Look for profiler logs
        log_files = glob.glob("outputs/logs/**/metrics.csv", recursive=True)
        if log_files:
            latest_log = max(log_files, key=os.path.getmtime)
            df = pd.read_csv(latest_log)

            st.markdown("#### ⚡ Throughput vs Compute")
            if "perf/throughput_samples_per_sec" in df.columns:
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(y=df["perf/throughput_samples_per_sec"], name="Throughput"))
                fig_perf.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.warning("No throughput metrics found in logs. Run training with performance callbacks enabled.")

            st.markdown("#### 💾 Memory Health")
            if "gpu/memory_allocated_gb" in df.columns:
                fig_mem = go.Figure()
                fig_mem.add_trace(go.Scatter(y=df["gpu/memory_allocated_gb"], name="Allocated (GB)"))
                if "gpu/fragmentation_mb" in df.columns:
                    fig_mem.add_trace(go.Scatter(y=df["gpu/fragmentation_mb"] / 1024, name="Fragmentation (GB)"))
                fig_mem.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_mem, use_container_width=True)

            st.markdown("#### ⏳ Step Latency Breakdown")
            if "time/data_load" in df.columns and "time/batch_processing" in df.columns:
                avg_load = df["time/data_load"].mean()
                avg_comp = df["time/batch_processing"].mean()

                fig_pie = go.Figure(data=[go.Pie(labels=['Data Loading', 'GPU Compute'], values=[avg_load, avg_comp])])
                fig_pie.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

                if avg_load / (avg_load + avg_comp) > 0.3:
                    st.warning("⚠️ High Data Loading Overhead detected (>30%). Check disk I/O or num_workers.")

            st.markdown("#### 🔬 Internal Step Timing")
            step_timing_columns = {
                "Train unpack": "time/train_unpack_ms",
                "Train forward": "time/train_forward_ms",
                "Train loss": "time/train_loss_ms",
                "Train metrics": "time/train_metrics_ms",
                "Validation forward": "time/val_forward_ms",
                "Validation loss": "time/val_loss_ms",
                "Forward/loss phase": "time/forward_loss_sec",
                "Backward phase": "time/backward_sec",
                "Pre-optimizer": "time/pre_optimizer_sec",
                "Optimizer step": "time/optimizer_step_sec",
            }
            available_step_columns = {
                label: column
                for label, column in step_timing_columns.items()
                if column in df.columns
            }
            if available_step_columns:
                fig_internal = go.Figure()
                step_means = {}
                for label, column in available_step_columns.items():
                    series = pd.to_numeric(df[column], errors="coerce").dropna()
                    if series.empty:
                        continue
                    if column.endswith("_sec"):
                        series = series * 1000
                    step_means[label] = series.mean()
                    fig_internal.add_trace(go.Scatter(y=series, name=label))

                fig_internal.update_layout(
                    template="plotly_dark",
                    height=320,
                    yaxis_title="Milliseconds",
                )
                st.plotly_chart(fig_internal, use_container_width=True)

                if step_means:
                    slowest_label = max(step_means, key=step_means.get)
                    slowest_ms = step_means[slowest_label]
                    st.info(f"Largest internal step cost: {slowest_label} ({slowest_ms:.2f} ms avg).")
            else:
                st.caption("No internal step timing metrics found. Enable diagnostics.step_detail for the next training run.")
        else:
            st.warning("No performance logs found. Training data must be generated with `CSVLogger` enabled.")

if __name__ == "__main__":
    main()
