import sys, os, time, json, re
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

# ─── project paths ────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from inject_anomalies import AE, COLS, SEQ

MODEL_PATH = ROOT / "model" / "lstm_ae.pt"
DATA_PATH  = ROOT / "data" / "annotated.csv"
KEYS_DIR   = ROOT / "keys"
LOG_PATH   = ROOT / "listener_output.log"

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔌 Smart-Grid Anomaly Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── load+cache model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    net  = AE(len(COLS))
    net.load_state_dict(ckpt)
    net.eval()
    return net

net = load_model()

# ─── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --card: #161b22; }
body, .css-18e3th9 { background: var(--bg); color: var(--fg); font-family: 'Segoe UI', sans-serif; }
h1, h2, h3, .stMetric-label { color: var(--accent); }
.stSidebar { background: var(--card); }
div[data-baseweb] { border-radius:8px !important; }
.card {
  background: var(--card);
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.7);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.9);
}
.stDataFrame thead th {
  background-color: #21262d !important;
  color: #adbac7 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── top header + navigation ──────────────────────────────────────────────────
st.markdown("## 🔌 Smart-Grid Anomaly Dashboard", unsafe_allow_html=True)
tabs = st.tabs([
    "📊 Evaluate Model",
    "🔍 Live Monitoring",
    "📂 Offline Alerts",
    "🔑 Key Management"
])

# ─── Evaluate Model ───────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Model Evaluation")
    with st.spinner("Reconstructing & computing metrics…"):
        df = pd.read_csv(DATA_PATH).ffill()
        df["V_real"] = df["voltage_C"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
        df["V_imag"] = df["voltage_C"].str.extract(r"\+(\d*\.?\d+)j").astype(float)
        df["label"]  = df["label"].astype(str).map({"True": True, "False": False})

        x    = torch.tensor(df[COLS].values, dtype=torch.float32)
        seqs = torch.stack([x[i:i+SEQ] for i in range(len(x)-SEQ)])
        with torch.no_grad():
            recon  = net(seqs)
        errors   = ((recon - seqs)**2).mean(dim=(1,2)).cpu().numpy()
        thr      = errors.mean() + 3*errors.std()
        mean_err = errors.mean()
        std_err  = errors.std()
        pred     = (errors > thr).astype(int)
        true     = df["label"].iloc[SEQ:].astype(int).values

    # metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Threshold", f"{thr:.2e}")
    col2.metric("Mean Error", f"{mean_err:.2e}")
    col3.metric("Std Dev",    f"{std_err:.2e}")

    # classification table
    report = classification_report(true, pred, digits=4, output_dict=True, zero_division=0)
    rpt_df = pd.DataFrame(report).transpose()
    st.dataframe(rpt_df.style.background_gradient(axis=1), use_container_width=True)

    # error histogram
    fig, ax = plt.subplots(figsize=(10,3))
    ax.hist(errors, bins=40, alpha=0.85)
    ax.axvline(thr, color="#f85149", linestyle="--", lw=2)
    ax.set(title="Error Distribution", xlabel="Reconstruction Error", ylabel="Count")
    st.pyplot(fig, use_container_width=True)

# ─── Live Monitoring ──────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Real-time Alerts from Listener")
    if not LOG_PATH.exists():
        st.warning("No listener log found. Run `listener.py` first.")
    else:
        lines, alerts, buf = LOG_PATH.read_text().splitlines(), [], {}
        for ln in lines:
            if ln.startswith("🚨 Decrypted alert:"):
                buf = {}
            if "Timestamp" in ln:
                buf["timestamp"] = ln.split(":",2)[2].strip()
            if "Error" in ln and "error" not in buf:
                buf["error"] = float(ln.split(":",1)[1])
            if "Key File" in ln:
                buf["key"] = ln.split(":",1)[1].strip()
                alerts.append(buf.copy())

        if not alerts:
            st.info("No decrypted alerts yet.")
        else:
            for a in alerts[-5:]:
                st.markdown(f"""
                <div class="card">
                  <h4>🚨 {a['timestamp']}</h4>
                  <p><strong>Error:</strong> {a['error']:.2f}</p>
                  <p><small>Key:</small> {a['key']}</p>
                </div>
                """, unsafe_allow_html=True)

# ─── Offline Alerts ──────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Decrypt Offline Alerts")
    st.info("Upload any text or log file with lines beginning `enc_alert=`")
    up = st.file_uploader("Select file", type=["txt","log"])
    if up:
        raw   = up.getvalue().decode("utf-8", errors="ignore")
        lines = [l for l in raw.splitlines() if l.strip().startswith("enc_alert=")]
        if not lines:
            st.error("No valid `enc_alert=` lines found.")
        else:
            parsed = []
            for l in lines:
                hexstr = l.split("=",1)[1]
                try:
                    pkt = bytes.fromhex(hexstr)
                except ValueError:
                    continue
                nonce, ct = pkt[:12], pkt[12:]
                for kf in sorted(KEYS_DIR.glob("*.bin"), reverse=True):
                    try:
                        pt = AESGCM(kf.read_bytes()).decrypt(nonce, ct, None)
                        d  = json.loads(pt)
                        d["key"] = kf.name
                        parsed.append(d)
                        break
                    except:
                        continue
            if parsed:
                st.success(f"Decrypted {len(parsed)} alerts")
                st.table(pd.DataFrame(parsed))
            else:
                st.error("Could not decrypt any with existing keys.")

# ─── Key Management ─────────────────────────────────────────────────────────
with tabs[3]:
    st.header("QKD Key Management")
    st.write(f"**Key directory:** `{KEYS_DIR}`")
    keys = sorted(KEYS_DIR.glob("*.bin"))
    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Generate New Key"):
            new  = secrets.token_bytes(32)
            name = f"{int(time.time())}.bin"
            (KEYS_DIR/name).write_bytes(new)
            st.success(f"Created {name}")
    with c2:
        st.metric("Total Keys", len(keys))
    exp = st.expander("Show all keys")
    for k in keys:
        exp.write(k.name)
