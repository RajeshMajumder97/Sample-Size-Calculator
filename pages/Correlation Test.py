import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title=" StydySizer | Correlation Hypothesis Test", page_icon="🧮")

st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sample Size Calculation for Correlation Test | H₀: ρ = ρ₀ vs H₁: ρ ≠ ρ₀")

# Fisher's Z-transformation sample size function
def nSampleCorrelation(rho0=0.0, rho1=0.3, Pw=0.8, Conf=0.95, designEf=1, dropOut=0):
    z0 = 0.5 * np.log((1 + rho0) / (1 - rho0))
    z1 = 0.5 * np.log((1 + rho1) / (1 - rho1))
    delta_z = abs(z1 - z0)
    n = ((norm.ppf(1 - (1 - Conf) / 2) + norm.ppf(Pw)) / delta_z) ** 2 + 3
    return int(np.ceil((n / (1 - dropOut)) * designEf))

if "corr_history" not in st.session_state:
    st.session_state.corr_history = []

# Sidebar inputs
rho0 = st.sidebar.number_input("Null hypothesis correlation (ρ₀)", value=0.0, min_value=-0.99, max_value=0.99)
rho1 = st.sidebar.number_input("Expected correlation (ρ₁)", value=0.3, min_value=-0.99, max_value=0.99)

if rho0 == rho1:
    st.sidebar.warning("ρ₀ and ρ₁ cannot be the same.")
    st.stop()

power = st.sidebar.number_input("Power (%)", value=80.0, min_value=50.0, max_value=99.0)
drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0)

method = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

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

# Button to calculate
go = st.button("Calculate Sample Size")

# Helper for history label
def make_corr_label(rho0, rho1, power, drp, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"ρ₀={rho0}, ρ₁={rho1}, Power={power}%, DropOut={drp}%, DE={round(designEffect, 2)}"
    else:
        return f"ρ₀={rho0}, ρ₁={rho1}, Power={power}%, DropOut={drp}%, DE={round(designEffect, 2)}, m={m}, ICC={ICC}"

# History selector
selected_history = None
if st.session_state.corr_history:
    st.subheader("📜 Select from Past Inputs")
    corr_labels = [make_corr_label(**item) for item in st.session_state.corr_history]
    selected = st.selectbox("Choose a past input set:", corr_labels, key="corr_history_selector")
    if selected:
        selected_history = next(item for item in st.session_state.corr_history if make_corr_label(**item) == selected)
        recalc = st.button("🔁 Recalculate")
    else:
        recalc = False
else:
    recalc = False

if go or recalc:
    if recalc and selected_history:
        rho0 = selected_history["rho0"]
        rho1 = selected_history["rho1"]
        power = selected_history["power"]
        drp = selected_history["drp"]
        designEffect = selected_history["designEffect"]
    else:
        st.session_state.corr_history.append({
            "rho0": rho0,
            "rho1": rho1,
            "power": power,
            "drp": drp,
            "designEffect": designEffect,
            "m": m,
            "ICC": ICC,
            "method": method
        })

    conf_levels = [0.8,0.9,0.97,0.99,0.999,0.9999]
    results = []
    for conf in conf_levels:
        n = nSampleCorrelation(rho0=rho0, rho1=rho1, Pw=power / 100, Conf=conf, designEf=designEffect, dropOut=drp / 100)
        results.append(n)

    df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})
    n95 = nSampleCorrelation(rho0, rho1, Pw=power / 100, Conf=0.95, designEf=designEffect, dropOut=drp / 100)

    st.write("The required total sample size is:")
    st.markdown(f"""
    <div style='display: flex; justify-content: center;'>
        <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
            {n95}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"""To detect a difference between ρ₀ = {rho0} and ρ₁ = {rho1} with {power}% power and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, considering a design effect of {round(designEffect, 2)} and drop-out rate of {drp}%.""", unsafe_allow_html=True)
    st.subheader("Sample Sizes at Other Confidence Levels")
    st.dataframe(df)

st.markdown("---")
st.subheader("📌 Formula Used")
st.latex(r"""
n = \left(\left( \frac{Z_{1-\alpha/2} + Z_{1-\beta}}{\frac{1}{2} \ln\left(\frac{1 + \rho_1}{1 - \rho_1}\right) - \frac{1}{2} \ln\left(\frac{1 + \rho_0}{1 - \rho_0}\right)} \right)^2 + 3\right) \times DE
""")

st.markdown("### Design Effect (if clusters are used):")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{alpha} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{beta} \)**: Standard normal quantile for power (1 - beta).
- **\( n \)**: Number of observations.
- **\( rho_0 \)**: Correlation under null hypothesis.
- **\( rho_1 \)**: Correlation under alternative hypothesis.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")


st.markdown("---")
st.subheader("References")
st.markdown("""
1. **Hulley et al. (2013).** Designing Clinical Research. 4th Ed. Lippincott Williams & Wilkins.
2. **Cohen, J. (1988).** Statistical Power Analysis for the Behavioral Sciences. Routledge.
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")