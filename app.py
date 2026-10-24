import streamlit as st
import joblib
import os
from feature_extractor import FeatureExtractor
from heuristics import analyse as heuristic_analyse
from db import init_db, log_scan, get_recent_scans, get_scan_stats

# Initialize database
init_db()

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI - URL Safety Scanner",
    page_icon="shield",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -- Load CSS ------------------------------------------------------------------
def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass

# -- Load Model ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists("model.pkl"):
        payload = joblib.load("model.pkl")
        if isinstance(payload, dict):
            return payload["model"], payload["meta"]["feature_names"]
        return payload, None
    return None, None

model, feature_names = load_model()

# -- Hybrid Prediction ---------------------------------------------------------
def predict(url: str):
    """
    Returns (is_phishing: bool, confidence: float, flags: list, source: str)
    Combines ML model output with rule-based heuristics.
    """
    # --- Heuristic layer ---
    h = heuristic_analyse(url)
    h_score   = h["score"]    # 0-1, how suspicious
    h_flags   = h["flags"]
    h_override = h["override"]

    # --- ML layer ---
    ml_phishing_prob = 0.5
    if model is not None:
        ext = FeatureExtractor()
        feat_df = ext.extract_features(url)
        if feature_names:
            feat_df = feat_df[[c for c in feature_names if c in feat_df.columns]]
        proba = model.predict_proba(feat_df)[0]
        ml_phishing_prob = proba[1]   # probability of phishing class

    # --- Combine ---
    # If heuristics fire a hard override (IP domain, @ trick): force phishing
    if h_override:
        final_phishing_prob = max(0.90, h_score)
        source = "heuristic"
    elif h_score >= 0.5:
        # Strong heuristic signal: blend 70% heuristic, 30% ML
        final_phishing_prob = 0.7 * h_score + 0.3 * ml_phishing_prob
        source = "hybrid"
    elif h_score >= 0.2:
        # Mild heuristic signal: blend 40% heuristic, 60% ML
        final_phishing_prob = 0.4 * h_score + 0.6 * ml_phishing_prob
        source = "hybrid"
    else:
        # No meaningful heuristic signal: trust the ML model
        final_phishing_prob = ml_phishing_prob
        source = "model"

    is_phishing = final_phishing_prob >= 0.5
    confidence  = final_phishing_prob if is_phishing else (1.0 - final_phishing_prob)

    return is_phishing, round(confidence * 100, 1), h_flags, source


# -- UI Helpers ----------------------------------------------------------------
def conf_bar(pct: float, kind: str) -> str:
    fill_class = "conf-bar-fill-safe" if kind == "safe" else "conf-bar-fill-phishing"
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-label">
            <span>Confidence</span><span>{pct:.1f}%</span>
        </div>
        <div class="conf-bar-track">
            <div class="{fill_class}" style="width:{pct}%;"></div>
        </div>
    </div>
    """

def metric_card(icon, label, value, color_class=""):
    return f"""
    <div class="metric-card">
        <div class="m-icon">{icon}</div>
        <div class="m-label">{label}</div>
        <div class="m-value {color_class}">{value}</div>
    </div>
    """


# -- HERO ----------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI-Powered Security</div>
    <h1 class="hero-title">PhishGuard AI</h1>
    <p class="hero-sub">
        Instantly detect phishing &amp; malicious URLs using a hybrid
        Machine Learning + heuristic engine trained on 235,000+ real-world samples.
        No sign-up. No tracking. 100% private.
    </p>
</div>
""", unsafe_allow_html=True)

# -- SCAN CARD -----------------------------------------------------------------
st.markdown('<div class="scan-card">', unsafe_allow_html=True)

with st.form(key="scan_form", clear_on_submit=False):
    url_input = st.text_input(
        label="URL to Scan",
        placeholder="e.g.  https://paypal-secure-login.fakesite.com/verify",
        label_visibility="visible",
        key="url_field"
    )
    scan_btn = st.form_submit_button("Scan URL for Threats")

if scan_btn:
    raw_url = url_input.strip()
    if not raw_url:
        st.warning("Please enter a URL before scanning.")
    else:
        with st.spinner("Analysing URL..."):
            is_phishing, confidence, flags, source = predict(raw_url)
            log_scan(raw_url, is_phishing, confidence, source, flags)

            ext = FeatureExtractor()
            feat_df = ext.extract_features(raw_url)
            feats = feat_df.iloc[0]

            # Display scanned URL
            display_url = raw_url if len(raw_url) <= 70 else raw_url[:67] + "..."
            st.markdown(f"""
            <div class="scanned-url">
                <span>Scanned URL</span>&nbsp; {display_url}
            </div>
            """, unsafe_allow_html=True)

            # ---- Result card -----
            if not is_phishing:
                st.markdown(f"""
                <div class="result-safe">
                    <div class="result-icon-safe">shield</div>
                    <div class="result-heading-safe">Legitimate URL</div>
                    <p class="result-desc">
                        This URL does not exhibit known phishing patterns.
                        It appears safe to visit — but always stay cautious online.
                    </p>
                    {conf_bar(confidence, "safe")}
                </div>
                """, unsafe_allow_html=True)
            else:
                flags_html = ""
                if flags:
                    items = "".join(f"<li>{f}</li>" for f in flags)
                    flags_html = f'<ul class="flags-list">{items}</ul>'

                st.markdown(f"""
                <div class="result-phishing">
                    <div class="result-icon-phishing">warning</div>
                    <div class="result-heading-phishing">Phishing Detected!</div>
                    <p class="result-desc">
                        Warning — this URL exhibits strong phishing indicators.
                        Do <strong>not</strong> enter credentials or personal info here.
                    </p>
                    {flags_html}
                    {conf_bar(confidence, "phishing")}
                </div>
                """, unsafe_allow_html=True)

            # ---- URL metrics breakdown ----
            st.markdown('<div class="metrics-section">', unsafe_allow_html=True)
            st.markdown('<div class="metrics-heading">URL Analysis Breakdown</div>', unsafe_allow_html=True)

            https_val = "HTTPS" if feats["IsHTTPS"] else "HTTP"
            https_cls = "good" if feats["IsHTTPS"] else "warn"
            is_ip_val = "Yes" if feats["IsDomainIP"] else "No"
            is_ip_cls = "bad" if feats["IsDomainIP"] else "good"
            sub_cls   = "bad" if feats["NoOfSubDomain"] >= 4 else ("warn" if feats["NoOfSubDomain"] >= 2 else "good")
            obf_cls   = "bad" if feats["NoOfObfuscatedChar"] > 0 else "good"
            len_cls   = "warn" if feats["URLLength"] > 75 else ""

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(metric_card("URL", "URL Length", int(feats["URLLength"]), len_cls), unsafe_allow_html=True)
            with col2:
                st.markdown(metric_card("DOM", "Domain Length", int(feats["DomainLength"]), "warn" if feats["DomainLength"] > 30 else ""), unsafe_allow_html=True)
            with col3:
                st.markdown(metric_card("SEC", "Protocol", https_val, https_cls), unsafe_allow_html=True)

            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(metric_card("SUB", "Subdomains", int(feats["NoOfSubDomain"]), sub_cls), unsafe_allow_html=True)
            with col5:
                st.markdown(metric_card("IP", "IP Domain", is_ip_val, is_ip_cls), unsafe_allow_html=True)
            with col6:
                st.markdown(metric_card("OBF", "Obfuscation", int(feats["NoOfObfuscatedChar"]), obf_cls), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -- Example chips -------------------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-top:1.6rem;">
    <p style="font-size:0.78rem;color:#334155;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.6rem;">
        Try an example
    </p>
    <span class="pill">google.com</span>
    <span class="pill">github.com</span>
    <span class="pill">paypal-verify.fakesite.xyz</span>
    <span class="pill">http://192.168.1.1/login</span>
</div>
""", unsafe_allow_html=True)

# -- Database Audit Logs & Analytics -------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Database Audit Logs & Scan History (Cloud Persistence Layer)", expanded=True):
    total_scans, phishing_count, safe_count = get_scan_stats()
    st.write(f"**Total Database Logs:** `{total_scans}` &nbsp;|&nbsp; **Phishing Detected:** `{phishing_count}` &nbsp;|&nbsp; **Legitimate URLs:** `{safe_count}`")
    
    logs = get_recent_scans(20)
    if logs:
        import pandas as pd
        df_logs = pd.DataFrame(logs, columns=["ID", "URL", "Is Phishing", "Confidence (%)", "Engine", "Flags", "Timestamp"])
        df_logs["Verdict"] = df_logs["Is Phishing"].apply(lambda x: "🚨 Phishing" if x == 1 else "✅ Safe")
        st.dataframe(
            df_logs[["Timestamp", "URL", "Verdict", "Confidence (%)", "Engine", "Flags"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No scan records in database yet. Scan a URL above to save entries to the database.")

# -- Footer --------------------------------------------------------------------
st.markdown("""
<div class="footer-strip">
    Built with Streamlit &nbsp;|&nbsp;
    Hybrid ML + Heuristic Engine &nbsp;|&nbsp;
    Trained on the PhiUSIIL Phishing URL Dataset (235,000+ samples)
</div>
""", unsafe_allow_html=True)
