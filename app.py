"""
CRISP-DM Linear Regression Explorer — single-file Streamlit app
Run: streamlit run app.py
"""

import io
import os
import pathlib
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRISP-DM · Linear Regression",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── light main bg ── */
.stApp { background: #f1f5f9; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }
[data-testid="stMain"] { background: #f1f5f9; }

/* ── sidebar: clean white ── */
section[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #1e293b !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { font-weight: 700 !important; color: #0f172a !important; }
section[data-testid="stSidebar"] hr { border-color: #cbd5e1 !important; }

/* ── hero title ── */
/* ── body text ── */
.stApp, .stApp p, .stApp li, .stApp label { color: #1e293b !important; }
.stMarkdown, .stMarkdown p { color: #1e293b !important; }

.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15; margin-bottom: .25rem;
}
.hero-sub { color: #475569; font-size: 1rem; margin-bottom: 1.8rem; }

/* ── phase badge ── */
.phase-badge {
    display: inline-block;
    background: linear-gradient(135deg,#312e81,#1e1b4b);
    border: 1px solid #4338ca; border-radius: 6px;
    font-size:.7rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#a5b4fc; padding:3px 10px;
    margin-bottom:.5rem;
}

/* ── metric cards ── */
.metric-row { display:flex; gap:1rem; margin-bottom:1rem; }
.metric-card {
    flex:1; background:#ffffff; border:1px solid #e2e8f0;
    border-radius:14px; padding:1.1rem 1.2rem; text-align:center;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.metric-card .mlabel { font-size:.72rem; font-weight:600; letter-spacing:.07em;
    text-transform:uppercase; color:#64748b; margin-bottom:.3rem; }
.metric-card .mvalue { font-size:2rem; font-weight:700; color:#0f172a; }
.metric-card .msub   { font-size:.74rem; color:#94a3b8; margin-top:.15rem; }

/* ── info box ── */
.info-box {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
    padding:.9rem 1.1rem; font-size:.88rem; color:#334155; line-height:1.7;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.info-box code { background:#f1f5f9; color:#4f46e5; border-radius:4px;
    padding:1px 6px; font-size:.85rem; }

/* ── expander headers: brighter ── */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: .5rem !important;
    border-left: 3px solid #6366f1 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}
div[data-testid="stExpander"] summary {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: .97rem !important;
}
div[data-testid="stExpander"] summary:hover {
    color: #4f46e5 !important;
    background: #f8fafc !important;
}
div[data-testid="stExpander"] summary svg {
    fill: #6366f1 !important;
}

/* ── equation highlight ── */
.eq-box {
    background:linear-gradient(135deg,#1e1b4b,#0f172a);
    border:1px solid #4338ca; border-radius:10px;
    padding:.75rem 1.2rem; text-align:center;
    font-size:1.05rem; font-weight:600; color:#c7d2fe;
    margin:.6rem 0 1rem;
}

/* ── prediction result ── */
.pred-result {
    background:linear-gradient(135deg,#ecfdf5,#d1fae5);
    border:1px solid #10b981; border-radius:12px;
    padding:1rem 1.3rem; text-align:center;
    font-size:1.4rem; font-weight:700; color:#065f46;
    margin-top:.8rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_data(n: int, a: float, b: float,
                  noise_mean: float, noise_var: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, n)
    noise = rng.normal(noise_mean, np.sqrt(max(noise_var, 1e-9)), n)
    y = a * x + b + noise
    return x, y


@st.cache_resource(show_spinner=False)
def train_model(n, a, b, noise_mean, noise_var, seed, test_size):
    x, y = generate_data(n, a, b, noise_mean, noise_var, seed)
    X = x.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ])
    pipe.fit(X_tr, y_tr)

    y_pred_tr = pipe.predict(X_tr)
    y_pred_te = pipe.predict(X_te)

    metrics = {
        "mse_train":  mean_squared_error(y_tr, y_pred_tr),
        "mse_test":   mean_squared_error(y_te, y_pred_te),
        "rmse_train": np.sqrt(mean_squared_error(y_tr, y_pred_tr)),
        "rmse_test":  np.sqrt(mean_squared_error(y_te, y_pred_te)),
        "r2_train":   r2_score(y_tr, y_pred_tr),
        "r2_test":    r2_score(y_te, y_pred_te),
    }
    return pipe, x, y, X_tr, X_te, y_tr, y_te, y_pred_te, metrics


def make_figure(x, y, pipe, X_te, y_te, y_pred_te):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0b0f1a")

    # ── left: scatter + regression line ──
    ax = axes[0]
    ax.set_facecolor("#111827")
    x_line = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    y_line = pipe.predict(x_line)
    ax.scatter(x, y, color="#60a5fa", alpha=0.35, s=12, label="Data")
    ax.plot(x_line, y_line, color="#f87171", lw=2.5, label="Fitted line")
    ax.set_xlabel("x", color="#94a3b8"); ax.set_ylabel("y", color="#94a3b8")
    ax.set_title("Data + Regression Line", color="#e2e8f0", fontweight="bold")
    ax.tick_params(colors="#64748b")
    for sp in ax.spines.values(): sp.set_edgecolor("#1f2937")
    ax.legend(facecolor="#111827", edgecolor="#1f2937",
              labelcolor="#e2e8f0", fontsize=9)
    ax.grid(True, color="#1f2937", lw=.5, ls="--", alpha=.7)

    # ── right: residuals ──
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    residuals = y_te - y_pred_te
    ax2.scatter(y_pred_te, residuals, color="#a78bfa", alpha=0.5, s=12)
    ax2.axhline(0, color="#f87171", lw=1.8, ls="--")
    ax2.set_xlabel("Predicted ŷ", color="#94a3b8")
    ax2.set_ylabel("Residual (y − ŷ)", color="#94a3b8")
    ax2.set_title("Residual Plot (Test Set)", color="#e2e8f0", fontweight="bold")
    ax2.tick_params(colors="#64748b")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1f2937")
    ax2.grid(True, color="#1f2937", lw=.5, ls="--", alpha=.7)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Model Parameters")
    st.markdown("Adjust parameters and click **Generate Data** to update everything.")
    st.markdown("---")

    st.markdown("### 📐 Data Generation")
    n          = st.slider("Sample size (n)",         100,  1000, 500,  50)
    noise_var  = st.slider("Noise variance σ²",       0,    1000, 100,  10)
    noise_mean = st.slider("Noise mean",              -10,  10,   0,    1)
    seed       = st.slider("Random seed",             0,    9999, 42,   1)

    st.markdown("---")
    st.markdown("### 🎯 True Parameters")
    a_true = st.slider("True slope a",      -10.0, 10.0, 5.0, 0.5)
    b_true = st.slider("True intercept b",  -50.0, 50.0, 15.0, 1.0)

    st.markdown("---")
    st.markdown("### 🔀 Train / Test Split")
    test_pct = st.slider("Test size (%)", 10, 40, 20, 5)

    st.markdown("---")
    generate_btn = st.button("🚀 Generate Data & Train", use_container_width=True)

# persist generation trigger across reruns
if generate_btn:
    st.session_state["generated"] = True

if "generated" not in st.session_state:
    st.session_state["generated"] = True   # auto-run on first load


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">📈 CRISP-DM Linear Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Interactive end-to-end machine learning workflow following the CRISP-DM framework</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔹 Phase 1 — Business Understanding", expanded=False):
    st.markdown('<div class="phase-badge">Phase 1</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
<b>Objective:</b> Determine whether a linear relationship exists between a single predictor <code>x</code>
and a target <code>y</code>, and build a model that accurately predicts <code>y</code> for unseen <code>x</code> values.<br><br>
<b>Success criteria:</b>
<ul style="margin:0;padding-left:1.2rem">
  <li>R² ≥ 0.90 on the test set</li>
  <li>RMSE as low as the irreducible noise allows</li>
  <li>Learned coefficients close to the true generating parameters</li>
</ul>
<br>
<b>Data source:</b> Synthetically generated using <code>y = ax + b + N(μ, σ²)</code>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE (cached)
# ══════════════════════════════════════════════════════════════════════════════

pipe, x, y, X_tr, X_te, y_tr, y_te, y_pred_te, metrics = train_model(
    n, a_true, b_true, noise_mean, noise_var, seed, test_pct / 100
)

# Extract learned raw coefficients (undo scaling for display)
lr = pipe.named_steps["lr"]
sc = pipe.named_steps["scaler"]
learned_a = (lr.coef_[0] / sc.scale_[0])
learned_b = lr.intercept_ - lr.coef_[0] * sc.mean_[0] / sc.scale_[0]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔹 Phase 2 — Data Understanding", expanded=False):
    st.markdown('<div class="phase-badge">Phase 2</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5, c6],
        ["n", "x̄", "x σ", "ȳ", "y σ", "Corr(x,y)"],
        [n, x.mean(), x.std(), y.mean(), y.std(), float(np.corrcoef(x, y)[0,1])],
    ):
        col.metric(label, f"{val:.3f}")

    st.markdown(
        f'<div class="eq-box">Generating equation: &nbsp; y = {a_true}x + {b_true} + N({noise_mean}, {noise_var})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Split:** {len(X_tr)} train / {len(X_te)} test &nbsp;|&nbsp; seed = {seed}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔹 Phase 3 — Data Preparation", expanded=False):
    st.markdown('<div class="phase-badge">Phase 3</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
<ul style="margin:0;padding-left:1.2rem">
  <li><b>Reshape:</b> <code>x</code> → <code>(n, 1)</code> for scikit-learn compatibility</li>
  <li><b>Train/Test split:</b> stratified random split via <code>train_test_split</code></li>
  <li><b>Scaling:</b> <code>StandardScaler</code> fitted on train set only, applied to both splits (no data leakage)</li>
  <li><b>No missing values</b> — synthetically generated data</li>
</ul>
</div>
""", unsafe_allow_html=True)
    st.markdown(f"**Scaler mean:** `{sc.mean_[0]:.4f}` &nbsp; | &nbsp; **Scaler std:** `{sc.scale_[0]:.4f}`")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — MODELING
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("🔹 Phase 4 — Modeling", expanded=False):
    st.markdown('<div class="phase-badge">Phase 4</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="info-box">
<b>Pipeline:</b> <code>StandardScaler</code> → <code>LinearRegression</code> (OLS)<br><br>
<b>Implementation:</b> <code>sklearn.pipeline.Pipeline</code> ensures the scaler is fitted on training data only,
preventing data leakage during cross-validation or test evaluation.
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    col_l.markdown("#### True parameters")
    col_l.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Slope **a** | `{a_true}` |
| Intercept **b** | `{b_true}` |
| Noise mean **μ** | `{noise_mean}` |
| Noise variance **σ²** | `{noise_var}` |
""")
    col_r.markdown("#### Learned coefficients")
    col_r.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Learned slope **â** | `{learned_a:.4f}` |
| Learned intercept **b̂** | `{learned_b:.4f}` |
| Slope error | `{abs(learned_a - a_true):.4f}` |
| Intercept error | `{abs(learned_b - b_true):.4f}` |
""")
    st.markdown(
        f'<div class="eq-box">Learned model: &nbsp; ŷ = {learned_a:.4f} x + {learned_b:.4f}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔹 Phase 5 — Evaluation")

m1, m2, m3, m4, m5, m6 = st.columns(6)

def mcard(col, label, value, fmt=".4f", sub=""):
    color = "#22c55e" if ("R²" in label and value >= 0.9) else \
            "#f59e0b" if ("R²" in label) else "#0f172a"
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="mlabel">{label}</div>'
        f'<div class="mvalue" style="color:{color}">{value:{fmt}}</div>'
        f'<div class="msub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

mcard(m1, "MSE — Train",  metrics["mse_train"],  ".2f",  "Mean Sq. Error")
mcard(m2, "MSE — Test",   metrics["mse_test"],   ".2f",  "Mean Sq. Error")
mcard(m3, "RMSE — Train", metrics["rmse_train"], ".4f",  "Root MSE")
mcard(m4, "RMSE — Test",  metrics["rmse_test"],  ".4f",  "Root MSE")
mcard(m5, "R² — Train",   metrics["r2_train"],   ".4f",  "≥0.90 = 🟢")
mcard(m6, "R² — Test",    metrics["r2_test"],    ".4f",  "≥0.90 = 🟢")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — DEPLOYMENT (Visualization + Prediction + Saving)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔹 Phase 6 — Deployment")

# ── charts ───────────────────────────────────────────────────────────────────
fig = make_figure(x, y, pipe, X_te, y_te, y_pred_te)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("---")

# ── prediction & saving ──────────────────────────────────────────────────────
pred_col, save_col = st.columns([1, 1], gap="large")

with pred_col:
    st.markdown("#### 🔮 Predict a New Value")
    x_input = st.number_input(
        "Enter x value", value=0.0, step=1.0, format="%.2f",
        help="The model will predict y for this x."
    )
    if st.button("Predict", use_container_width=True):
        y_hat = pipe.predict(np.array([[x_input]]))[0]
        st.markdown(
            f'<div class="pred-result">ŷ = {y_hat:.4f}</div>',
            unsafe_allow_html=True,
        )

with save_col:
    st.markdown("#### 💾 Save Trained Model")
    model_path = pathlib.Path("trained_model.joblib")

    if st.button("Save model (joblib)", use_container_width=True):
        joblib.dump(pipe, model_path)
        st.success(f"Model saved → `{model_path.resolve()}`")

    # ── download in-memory ──
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    st.download_button(
        label="⬇️ Download model file",
        data=buf,
        file_name="crisp_dm_linear_regression.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

# ── footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:.78rem;'>"
    "CRISP-DM Linear Regression Explorer &nbsp;·&nbsp; Streamlit + scikit-learn + joblib"
    "</div>",
    unsafe_allow_html=True,
)
