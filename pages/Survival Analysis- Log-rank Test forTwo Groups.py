import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Survival Analysis | Log-rank Test Sample Size (Two Groups)", page_icon="⚡")

st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Sample Size Calculation for Survival Analysis | Log-rank Test (Two Sample)")

# Sample size calculation function for log-rank test
def nSampleSurvival(HR=0.7, Pw=0.8, Conf=0.95, p=0.5, eventRate=0.6, designEf=1.0, dropOut=0.0):
    z_alpha = norm.ppf(1 - (1 - Conf) / 2)
    z_beta = norm.ppf(Pw)
    n_events = ((z_alpha + z_beta) ** 2) / (np.log(HR) ** 2 * p * (1 - p))
    n_total = n_events / eventRate
    return int(np.ceil((n_total / (1 - dropOut)) * designEf))

if "survival_history" not in st.session_state:
    st.session_state.survival_history = []

# Sidebar inputs
HR = st.sidebar.number_input("Hazard Ratio (HR)", value=0.7, min_value=0.01, max_value=10.0)
power = st.sidebar.number_input("Power (%)", value=80.0, min_value=0.0, max_value=100.0)
#conf = st.sidebar.number_input("Confidence Level (%)", value=95.0, min_value=50.0, max_value=99.999)
p = st.sidebar.number_input("Allocation Ratio (Group 1) [Exposed Group]", value=0.5, min_value=0.01, max_value=0.99)
eventRate = st.sidebar.number_input("Expected Event Rate", value=0.6, min_value=0.01, max_value=1.0)
drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=100.0)

method = st.sidebar.radio("Choose Method for Design Effect:", options=["Given", "Calculate"])

if method == "Given":
    designEffect = st.sidebar.number_input("Design Effect", value=1.0, min_value=1.0)
    m = None
    ICC = None
else:
    m = st.sidebar.number_input("Number of Clusters (m)", min_value=2)
    ICC = st.sidebar.number_input("Intra-class Correlation (ICC)", min_value=0.0)
    designEffect = 1 + (m - 1) * ICC
    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster Size (m)", value=m)
    col2.metric("ICC", value=ICC)
    col3.metric("Design Effect", value=round(designEffect, 2))

# Calculate button
go = st.button("Calculate Sample Size")

# History label
def make_survival_label(HR, power, p, eventRate, drp, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"HR={HR}, Power={power}%, EventRate={eventRate}, DE={round(designEffect, 2)}"
    else:
        return f"HR={HR}, Power={power}%, EventRate={eventRate}, DE={round(designEffect, 2)}, m={m}, ICC={ICC}"

# Select from history
selected_history = None
if st.session_state.survival_history:
    st.subheader("📜 Select from Past Inputs")
    survival_labels = [make_survival_label(**item) for item in st.session_state.survival_history]
    selected = st.selectbox("Choose a past input set:", survival_labels, key="survival_history_selector")
    if selected:
        selected_history = next(item for item in st.session_state.survival_history if make_survival_label(**item) == selected)
        recalc = st.button("🔁 Recalculate")
    else:
        recalc = False
else:
    recalc = False

if go or recalc:
    if recalc and selected_history:
        HR = selected_history["HR"]
        power = selected_history["power"]
        #conf = selected_history["conf"]
        p = selected_history["p"]
        eventRate = selected_history["eventRate"]
        drp = selected_history["drp"]
        designEffect = selected_history["designEffect"]
    else:
        st.session_state.survival_history.append({
            "HR": HR,
            "power": power,
            #"conf": conf,
            "p": p,
            "eventRate": eventRate,
            "drp": drp,
            "designEffect": designEffect,
            "m": m,
            "ICC": ICC,
            "method": method
        })

    conf_levels = [0.8,0.9,0.97,0.99,0.999,0.9999]
    results = []
    for conf in conf_levels:
        n = nSampleSurvival(HR=HR, Conf= conf, Pw=power / 100,p=p, eventRate=eventRate, designEf=designEffect, dropOut=drp / 100)
        results.append(n)

    df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})

    sample_size = nSampleSurvival(HR=HR, Pw=power / 100, Conf=(0.95), p=p, eventRate=eventRate, designEf=designEffect, dropOut=drp / 100)

    st.write("The required total sample size is:")
    st.markdown(f"""
    <div style='display: flex; justify-content: center;'>
        <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
            {sample_size}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"""To detect a hazard ratio of {HR} with {power}% power and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, assuming event rate of {eventRate}, allocation ratio {p}, design effect of {round(designEffect, 2)} and drop-out of {drp}%.""", unsafe_allow_html=True)

    st.subheader("Sample Sizes at Other Confidence Levels")
    st.dataframe(df)

st.markdown("---")
st.subheader("📌 Formula Used")
st.latex(r"""
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2}{[\log(HR)]^2 \cdot p(1-p)} \times \frac{1}{\text{event rate}} \times DE
""")

st.markdown("### Design Effect (if clusters are used):")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")
st.markdown("""
- **\( HR \)**: Hazard Ratio to detect
- **\( p \)**: Allocation proportion (Group 1) [Exposed Group]
- **\( Z_{1-\alpha/2} \)**: Z-score for confidence level
- **\( Z_{1-\beta} \)**: Z-score for power
- **\( event\ rate \)**: Proportion of individuals expected to experience the event
- **\( DE \)**: Design effect
- **\( ICC \)**: Intra-cluster correlation
""")

st.markdown("---")
st.markdown("""
    <div style="
        background-color: #48D1CC;
        padding: 10px;
        border-left: 5px solid orange;
        border-radius: 5px;
        font-size: 18px;">
        <b>Note:</b> This tool calculates the expected number of events by default. Multiply by 𝑑(event rate[0-1)) if you are calculating the expected number of events from a given sample size (default option) and Divide by 𝑑 if you are calculating the required sample size from a given number of events. Suppose your calculation shows that you need 100 events to detect a hazard ratio difference with the desired statistical power. If your estimated event rate is 40% (i.e., only 40% of participants are expected to experience the event), then you will need to enroll 100 divided by 0.4, which equals 250 participants in total. Conversely, if you are planning to enroll 250 participants in the study and you expect an event rate of 40%, then the expected number of events will be 250 multiplied by 0.4, which equals 100 events.
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.subheader("References")
st.markdown("""
1. Freedman, L. S. (1982). Tables of the number of patients required in clinical trials using the log-rank test. *Stat Med.*
2. Chow, S. C., Shao, J., & Wang, H. (2008). Sample Size Calculations in Clinical Research.
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
