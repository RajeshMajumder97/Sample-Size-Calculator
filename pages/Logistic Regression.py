import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Logistic Regression", page_icon="🧮")

# Hide Streamlit styles
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Sample Size Calculation for Multiple Logistic Regression")

# Formula function
def nSampleLogisticRegression(P=0.1, OR=1.5, R2=0.2, alpha=0.05, power=0.8, k=1, designEf=1, dropOut=0.0):
    Z_alpha = norm.ppf(1 - alpha / 2)
    Z_beta = norm.ppf(power)

    log_OR = np.log(OR)
    q = 1 - P
    numerator = (Z_alpha + Z_beta) ** 2 * (1 + (1 - R2) * (k - 1))
    denominator = P * q * (log_OR ** 2) * (1 - R2)
    n = numerator / denominator
    n_adjusted = n / (1 - dropOut)
    return int(np.ceil(n_adjusted * designEf))

# Initialize session history
if "logr_history" not in st.session_state:
    st.session_state.logr_history = []

# Sidebar inputs
P = st.sidebar.number_input("Overall Proportion of Disease (P)", value=10.0, min_value=1.0, max_value=99.99)
OR = st.sidebar.number_input("Anticipated Odds Ratio (OR)", value=1.5, min_value=0.01)
R2 = st.sidebar.number_input("R-squared with other predictors (R²)", value=0.2, min_value=0.0, max_value=0.99)
power = st.sidebar.number_input("Power (%)", value=80.0, min_value=50.0, max_value=99.9)
#alpha = st.sidebar.number_input("Significance Level (α)", value=0.05, min_value=0.001, max_value=0.2)
drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0)
k = st.sidebar.number_input("Number of Predictors", value=1, min_value=1)

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

go = st.button("Calculate Sample Size")

def make_logr_history_label(P, OR, R2, power, k, drp, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"P={P}, OR={OR}, R²={R2}, Power={power}%, k={k}, DropOut={drp}%, DE={round(designEffect,2)}"
    else:
        return f"P={P}, OR={OR}, R²={R2}, Power={power}%, k={k}, DropOut={drp}%, DE(Calc)={round(designEffect,2)}, m={m}, ICC={ICC}"

selected_history = None
if st.session_state.logr_history:
    st.subheader("📜 Select from Past Inputs")
    hist_labels = [make_logr_history_label(**entry) for entry in st.session_state.logr_history]
    selected = st.selectbox("Choose a past input set:", hist_labels, key="logr_history_selector")
    if selected:
        selected_history = next((item for item in st.session_state.logr_history if make_logr_history_label(**item) == selected), None)
        recalc = st.button("🔁 Recalculate")
    else:
        recalc = False
else:
    recalc = False

if go or recalc:
    if recalc and selected_history:
        P = selected_history["P"]
        OR = selected_history["OR"]
        R2 = selected_history["R2"]
        power = selected_history["power"]
        #alpha = selected_history["alpha"]
        drp = selected_history["drp"]
        k = selected_history["k"]
        designEffect = selected_history["designEffect"]
    else:
        st.session_state.logr_history.append({
            "P": P, "OR": OR, "R2": R2, "power": power,
            "drp": drp, "k": k,
            "designEffect": designEffect, "m": m, "ICC": ICC, "method": method
        })

    conf_levels = [0.8,0.9,0.97,0.99,0.999,0.9999]
    results = [nSampleLogisticRegression(P/100  , OR, R2, 1 - cl, power / 100, k, designEffect, drp / 100) for cl in conf_levels]

    df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})
    n95 = nSampleLogisticRegression(P/100, OR, R2, 0.95, power / 100, k, designEffect, drp / 100)

    st.write("The required sample size is:")
    st.markdown(f"""
    <div style='display: flex; justify-content: center;'>
        <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
            {n95}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"""
    To detect an Odds Ratio of {OR} for a predictor in a multiple logistic regression model with overall disease proportion {P}, R²={R2}, {power}% power, and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level. Design effect: {round(designEffect,2)}, Drop-out: {drp}%.
    """, unsafe_allow_html=True)

    st.subheader("Sample Sizes at Other Confidence Levels")
    st.dataframe(df)

# Math and parameters
st.markdown("---")
st.subheader("📌 Formula Used")

st.latex(r"""
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (1 + (1 - R^2)(k - 1))}{P(1 - P)(\ln(OR))^2(1 - R^2)} \times DE
""")

st.markdown("### Design Effect (if clusters are used):")
st.latex(r"""
DE = 1 + (m - 1) \cdot ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( P \)**: Overall disease prevalence or proportion of success.
- **\( OR \)**: Anticipated Odds Ratio.
- **\( R^2 \)**: Multiple correlation coefficient of exposure with other covariates.
- **\( \alpha \)**: Significance level (typically 0.05).
- **\( \beta \)**: Type II error = 1 - Power.
- **\( k \)**: Number of predictors in the model.
- **\( DE \)**: Design effect (for cluster sampling).
""")

st.subheader("📌 References")
st.markdown("""
1. **Hsieh FY, Bloch DA, Larsen MD. (1998)** A simple method of sample size calculation for linear and logistic regression. Statistics in Medicine.
2. **Dupont WD, Plummer WD. (1998)** PS Power and Sample Size Calculations. Controlled Clinical Trials.
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app](https://rajeshmajumderblog.netlify.app)")
