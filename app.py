import streamlit as st
import pandas as pd
import joblib
import random
from datetime import datetime

# ==========================
# Load model and pipeline
# ==========================
pipeline = joblib.load("unsw_pipeline_xgb.pkl")
label_encoder = joblib.load("unsw_label_encoder.pkl")
label_classes = label_encoder.classes_

# ==========================
# Feature mapping
# ==========================
numeric_features = [
    'dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload',
    'sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin',
    'tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len',
    'ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','ct_src_ltm','ct_srv_dst'
]

categorical_features = [
    'proto','service','state','is_ftp_login','ct_ftp_cmd',
    'ct_flw_http_mthd','is_sm_ips_ports'
]

feature_fullnames = {
    'dur': "Duration (sec)",
    'proto': "Protocol",
    'service': "Service Type",
    'state': "Connection State",
    'spkts': "Source Packets",
    'dpkts': "Destination Packets",
    'sbytes': "Source Bytes",
    'dbytes': "Destination Bytes",
    'rate': "Flow Rate",
    'sttl': "Source TTL",
    'dttl': "Destination TTL",
    'sload': "Source Load",
    'dload': "Destination Load",
    'sloss': "Source Packet Loss",
    'dloss': "Destination Packet Loss",
    'sinpkt': "Source Input Packets",
    'dinpkt': "Destination Input Packets",
    'sjit': "Source Jitter",
    'djit': "Destination Jitter",
    'swin': "Source TCP Window Size",
    'stcpb': "Source TCP Bytes",
    'dtcpb': "Destination TCP Bytes",
    'dwin': "Destination TCP Window Size",
    'tcprtt': "TCP Round Trip Time",
    'synack': "SYN-ACK Time",
    'ackdat': "ACK Data Time",
    'smean': "Source Mean Packet Size",
    'dmean': "Destination Mean Packet Size",
    'trans_depth': "Transaction Depth",
    'response_body_len': "Response Body Length",
    'ct_srv_src': "Connections to Same Service by Source",
    'ct_state_ttl': "States with Same TTL",
    'ct_dst_ltm': "Destination in Last Time Minute",
    'ct_src_dport_ltm': "Source Destination Ports in Last Time Minute",
    'ct_dst_sport_ltm': "Destination Source Ports in Last Time Minute",
    'ct_dst_src_ltm': "Destination Source in Last Time Minute",
    'is_ftp_login': "FTP Login Flag",
    'ct_ftp_cmd': "FTP Commands Count",
    'ct_flw_http_mthd': "HTTP Methods Count",
    'ct_src_ltm': "Source in Last Time Minute",
    'ct_srv_dst': "Service to Destination Count",
    'is_sm_ips_ports': "Small IPs & Ports Flag"
}

categories_dict = {
    'proto': ['tcp','udp','icmp'],
    'service': ['http','ftp','ssh','dns','smtp','other'],
    'state': ['CON','REQ','FIN','INT'],
    'is_ftp_login': ['0','1'],
    'ct_ftp_cmd': ['0','1','2'],
    'ct_flw_http_mthd': ['0','1','2','3'],
    'is_sm_ips_ports': ['0','1']
}

# ==========================
# Random sample generator
# ==========================
def generate_sample():
    sample = {}
    for feat in numeric_features:
        sample[feat] = random.randint(0, 1000)
    for feat in categorical_features:
        sample[feat] = random.choice(categories_dict[feat])
    return sample

def apply_sample_to_state(sample):
    st.session_state.features = sample
    for feat in numeric_features:
        st.session_state[f"num_{feat}"] = int(sample[feat])
    for feat in categorical_features:
        st.session_state[f"cat_{feat}"] = str(sample[feat])

def generate_sample_with_limit(max_value):
    sample = {}
    for feat in numeric_features:
        sample[feat] = random.randint(0, int(max_value))
    for feat in categorical_features:
        sample[feat] = random.choice(categories_dict[feat])
    return sample

def generate_numeric_only_sample(max_value, current_features):
    sample = dict(current_features)
    for feat in numeric_features:
        sample[feat] = random.randint(0, int(max_value))
    return sample

def build_preset_sample(preset_name, random_cap):
    sample = generate_sample_with_limit(random_cap)

    if preset_name == "Normal Web Traffic":
        sample.update({
            "proto": "tcp", "service": "http", "state": "CON",
            "dur": 12, "spkts": 20, "dpkts": 18, "rate": 80,
            "sbytes": 4200, "dbytes": 3900, "sloss": 0, "dloss": 0,
            "ct_srv_src": 3, "ct_state_ttl": 2
        })
    elif preset_name == "DNS Spike":
        sample.update({
            "proto": "udp", "service": "dns", "state": "CON",
            "dur": 2, "spkts": 90, "dpkts": 84, "rate": 520,
            "sbytes": 2500, "dbytes": 2400, "ct_srv_src": 10,
            "ct_dst_ltm": 14, "ct_srv_dst": 12
        })
    elif preset_name == "Recon Suspicion":
        sample.update({
            "proto": "tcp", "service": "other", "state": "REQ",
            "dur": 1, "spkts": 150, "dpkts": 8, "rate": 900,
            "sbytes": 1600, "dbytes": 220, "ct_src_dport_ltm": 30,
            "ct_dst_sport_ltm": 28, "ct_dst_src_ltm": 26, "ct_srv_src": 22
        })

    return sample

def zeroed_sample():
    sample = {}
    for feat in numeric_features:
        sample[feat] = 0
    for feat in categorical_features:
        sample[feat] = categories_dict[feat][0]
    return sample

def clear_prediction_state():
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_confidence", None)
    st.session_state["prediction_history"] = []

performance_reference = {
    "test_accuracy": 0.8359,
    "roc_auc": 0.9640,
    "evaluation_split": "Train/Test split from ai-based-ids.ipynb",
}

attack_knowledge_base = {
    "Normal": {"severity": "Low", "description": "Legitimate network activity without known attack signatures.", "action": "Allow traffic and continue passive monitoring."},
    "Analysis": {"severity": "Medium", "description": "Suspicious probing behavior often linked to reconnaissance or environment testing.", "action": "Inspect source host behavior and tighten monitoring rules."},
    "Backdoor": {"severity": "High", "description": "Potential unauthorized remote-access channel established by attacker tools.", "action": "Isolate affected endpoints and rotate credentials immediately."},
    "DoS": {"severity": "High", "description": "Flooding or resource exhaustion attempt to reduce service availability.", "action": "Trigger rate limiting and upstream mitigation controls."},
    "Exploits": {"severity": "High", "description": "Traffic pattern indicating exploitation of software vulnerabilities.", "action": "Patch vulnerable services and block offending indicators."},
    "Fuzzers": {"severity": "Medium", "description": "Repeated malformed input attempts to discover crashable attack surfaces.", "action": "Monitor for crash logs and restrict suspicious clients."},
    "Generic": {"severity": "High", "description": "Broad attack behavior with high malicious confidence but less specific family mapping.", "action": "Quarantine source and perform deeper packet inspection."},
    "Reconnaissance": {"severity": "Medium", "description": "Scanning and discovery activity for ports, hosts, or service fingerprints.", "action": "Throttle scan-like behavior and alert security analysts."},
    "Shellcode": {"severity": "Critical", "description": "Payload-like content that may indicate active code execution attempts.", "action": "Treat as incident: isolate systems and initiate response plan."},
    "Worms": {"severity": "Critical", "description": "Self-propagating behavior that can spread rapidly across network segments.", "action": "Segment network immediately and contain lateral movement."},
}

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="AI-Based IDS", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at top right, #1f2a44 0%, #111827 45%, #0b1220 100%);
        color: #e5e7eb;
        font-family: "Manrope", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif !important;
        color: #f9fafb !important;
    }

    p, label, span, div {
        color: #d1d5db;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #273449;
    }

    [data-testid="stSidebar"] * {
        color: #d1d5db !important;
    }

    [data-testid="stNumberInput"] input {
        background: #111827 !important;
        color: #f9fafb !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1rem;
    }

    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.45rem;
        flex-wrap: wrap;
    }

    [data-testid="stRadio"] label {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 999px !important;
        padding: 0.2rem 0.7rem !important;
        min-height: 2rem !important;
    }

    [data-testid="stRadio"] label p {
        color: #d1d5db !important;
        font-weight: 600;
    }

    [data-testid="stRadio"] label:hover {
        border-color: #60a5fa !important;
        background: #172554 !important;
    }

    [data-testid="stRadio"] label[aria-checked="true"],
    [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(120deg, #0ea5e9 0%, #2563eb 100%) !important;
        border-color: #93c5fd !important;
        box-shadow: 0 0 0 1px rgba(147, 197, 253, 0.45) inset;
    }

    [data-testid="stRadio"] label[aria-checked="true"] p,
    [data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
    }

    .stButton > button, [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(120deg, #0ea5e9 0%, #2563eb 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .hero-card {
        background: linear-gradient(120deg, rgba(14, 165, 233, 0.22), rgba(37, 99, 235, 0.22));
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }

    .hero-card p {
        margin: 0.3rem 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h2 style="margin:0;">AI-Based Intrusion Detection System</h2>
        <p>Analyze traffic signals and classify intrusion risk in one streamlined workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.caption("This IDS uses an XGBoost model trained on UNSW-NB15 dataset features.")
    with st.expander("Model Labels"):
        st.write(", ".join([str(label) for label in label_classes]))

    st.markdown("---")
    st.header("⚙️ Controls")

    random_cap = st.slider(
        "Random Sample Max Value",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )

    if st.button("Generate Random Sample", use_container_width=True):
        # Randomize only numeric signals, keep current categorical choices.
        apply_sample_to_state(generate_numeric_only_sample(random_cap, st.session_state.features))
    if st.button("Reset to New Sample", use_container_width=True):
        # Full new sample: randomize numeric + categorical.
        apply_sample_to_state(generate_sample_with_limit(random_cap))
        clear_prediction_state()
    if st.button("Clear Prediction", use_container_width=True):
        apply_sample_to_state(zeroed_sample())
        clear_prediction_state()

    st.markdown("---")
    st.subheader("Traffic Presets")
    selected_preset = st.selectbox(
        "Choose preset profile",
        ["Normal Web Traffic", "DNS Spike", "Recon Suspicion"],
        index=0
    )
    if st.button("Apply Preset", use_container_width=True):
        apply_sample_to_state(build_preset_sample(selected_preset, random_cap))
        clear_prediction_state()

    st.markdown("---")
    st.subheader("View Options")
    st.session_state["show_all_numeric"] = st.toggle(
        "Show all numeric features",
        value=st.session_state.get("show_all_numeric", False)
    )

if 'features' not in st.session_state:
    apply_sample_to_state(generate_sample())
else:
    for feat in numeric_features:
        key = f"num_{feat}"
        if key not in st.session_state:
            st.session_state[key] = int(st.session_state.features[feat])
    for feat in categorical_features:
        key = f"cat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = str(st.session_state.features[feat])
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_confidence' not in st.session_state:
    st.session_state.prediction_confidence = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
tab_predict, tab_dashboard, tab_performance, tab_knowledge = st.tabs(
    ["Predict", "Dashboard", "Model Performance", "Attack Knowledge Base"]
)

with tab_predict:
    f1, f2, f3 = st.columns(3)
    f1.metric("Numeric Features", len(numeric_features))
    f2.metric("Categorical Features", len(categorical_features))
    f3.metric("Model", "XGBoost")

    st.subheader("🔧 Input Network Traffic Features")
    with st.form("ids_form"):
        numeric_inputs, categorical_inputs = st.columns([1.4, 1])

        with numeric_inputs:
            st.markdown("**Numeric Features**")
            numeric_values = {}
            if st.session_state.get("show_all_numeric", False):
                display_numeric_features = numeric_features
            else:
                display_numeric_features = [
                    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
                    "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_srv_dst"
                ]

            for feat in display_numeric_features:
                numeric_values[feat] = st.number_input(
                    feature_fullnames[feat],
                    step=1,
                    format="%d",
                    key=f"num_{feat}"
                )

        with categorical_inputs:
            st.markdown("**Categorical Features (Button Style)**")
            categorical_values = {}
            for feat in categorical_features:
                options = categories_dict[feat]
                categorical_values[feat] = st.radio(
                    feature_fullnames[feat],
                    options=options,
                    horizontal=True,
                    key=f"cat_{feat}"
                )
            st.caption(
                "Selected: " + " | ".join(
                    [f"{feat}={categorical_values[feat]}" for feat in categorical_features]
                )
            )

        submitted = st.form_submit_button("Predict Attack Category", use_container_width=True)

    complete_numeric_values = {feat: int(st.session_state[f"num_{feat}"]) for feat in numeric_features}
    complete_numeric_values.update(numeric_values)
    input_dict = {**complete_numeric_values, **categorical_values}
    st.session_state.features = input_dict

    st.markdown("---")
    if submitted:
        try:
            df_input = pd.DataFrame([input_dict])
            pred_encoded = pipeline.predict(df_input)
            pred_label = label_encoder.inverse_transform(pred_encoded)[0]

            confidence = None
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(df_input)
                confidence = float(proba.max())

            st.session_state.prediction = pred_label
            st.session_state.prediction_confidence = confidence
            st.session_state.prediction_history.insert(
                0,
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": pred_label,
                    "confidence": round(confidence * 100, 2) if confidence is not None else None,
                    "proto": input_dict["proto"],
                    "service": input_dict["service"],
                    "state": input_dict["state"],
                    "rate": input_dict["rate"],
                    "spkts": input_dict["spkts"],
                    "dpkts": input_dict["dpkts"],
                },
            )
            st.session_state.prediction_history = st.session_state.prediction_history[:25]
        except Exception as e:
            st.session_state.prediction = None
            st.session_state.prediction_confidence = None
            st.error(f"Error during prediction: {e}")

    if st.session_state.prediction:
        conf = st.session_state.prediction_confidence
        if conf is not None:
            st.success(f"**Predicted Attack Category:** {st.session_state.prediction}  |  Model confidence: {conf*100:.2f}%")
            st.caption("Model confidence is a probability estimate, not a guarantee of correctness.")
        else:
            st.success(f"**Predicted Attack Category:** {st.session_state.prediction}")

    st.markdown("---")
    with st.expander("Preview Current Payload"):
        st.dataframe(pd.DataFrame([st.session_state.features]), use_container_width=True)

with tab_dashboard:
    st.subheader("📊 Dashboard")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Model", "XGBoost")
    d2.metric("Classes", len(label_classes))
    d3.metric("Predictions (Session)", len(st.session_state.prediction_history))
    d4.metric("Current Protocol", str(st.session_state.features["proto"]).upper())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current Traffic Snapshot**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rate": st.session_state.features["rate"],
                        "spkts": st.session_state.features["spkts"],
                        "dpkts": st.session_state.features["dpkts"],
                        "sbytes": st.session_state.features["sbytes"],
                        "dbytes": st.session_state.features["dbytes"],
                    }
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("**Current Selection**")
        st.write(f"Protocol: `{st.session_state.features['proto']}`")
        st.write(f"Service: `{st.session_state.features['service']}`")
        st.write(f"State: `{st.session_state.features['state']}`")
        if st.session_state.prediction:
            st.write(f"Last Prediction: `{st.session_state.prediction}`")
        else:
            st.write("Last Prediction: `-`")

    hist_df = pd.DataFrame(st.session_state.prediction_history)
    if not hist_df.empty:
        st.markdown("**Recent Predictions**")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        class_counts = hist_df["prediction"].value_counts()
        st.markdown("**Prediction Distribution (Session)**")
        st.bar_chart(class_counts)

        trend_df = hist_df.loc[:, ["time", "rate", "spkts", "dpkts"]].copy()
        trend_df = trend_df.iloc[::-1].reset_index(drop=True)
        trend_df.index = trend_df["time"]
        st.markdown("**Traffic Trend (Recent Runs)**")
        st.line_chart(trend_df[["rate", "spkts", "dpkts"]])
    else:
        st.info("No predictions yet. Run a prediction in the Predict tab to populate dashboard analytics.")

with tab_performance:
    st.subheader("📈 Model Performance")
    st.caption("Reference metrics from notebook evaluation.")

    p1, p2, p3 = st.columns(3)
    p1.metric("Test Accuracy", f"{performance_reference['test_accuracy'] * 100:.2f}%")
    p2.metric("ROC-AUC", f"{performance_reference['roc_auc']:.4f}")
    p3.metric("Classes", len(label_classes))
    st.caption(performance_reference["evaluation_split"])

    if st.session_state.prediction_history:
        perf_hist = pd.DataFrame(st.session_state.prediction_history)
        st.markdown("**Session Prediction Class Distribution**")
        st.bar_chart(perf_hist["prediction"].value_counts())

        if "confidence" in perf_hist.columns and perf_hist["confidence"].notna().any():
            st.markdown("**Session Model Confidence (%)**")
            conf_series = perf_hist["confidence"].dropna().iloc[::-1].reset_index(drop=True)
            st.line_chart(conf_series)
    else:
        st.info("No session predictions available yet. Run predictions to populate live performance panels.")

with tab_knowledge:
    st.subheader("🧠 Attack Knowledge Base")
    st.caption("Quick reference for predicted classes, severity, and response actions.")

    kb_rows = []
    for label in label_classes:
        label_str = str(label)
        info = attack_knowledge_base.get(
            label_str,
            {"severity": "Unknown", "description": "No class note available.", "action": "Review manually."},
        )
        kb_rows.append(
            {
                "Class": label_str,
                "Severity": info["severity"],
                "Description": info["description"],
                "Recommended Action": info["action"],
            }
        )

    kb_df = pd.DataFrame(kb_rows)
    st.dataframe(kb_df, use_container_width=True, hide_index=True)

    selected_class = st.selectbox("Class details", [str(c) for c in label_classes], index=0)
    selected_info = attack_knowledge_base.get(
        selected_class,
        {"severity": "Unknown", "description": "No class note available.", "action": "Review manually."},
    )
    st.markdown(f"**Severity:** `{selected_info['severity']}`")
    st.markdown(f"**Description:** {selected_info['description']}")
    st.markdown(f"**Recommended Action:** {selected_info['action']}")

st.markdown("Built with ❤️ using Streamlit | UNSW-NB15 Dataset")
