import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# Streamlit App Setup
st.set_page_config(page_title="Two Sample Proportion Hypothesis Testing", page_icon="🧊")

# Title
st.title("Sample Size Calculation for Proportion Test | H0: P1=P2")

# Function to calculate sample size
def nSampleProp(p1=0.5, p2=0.5, delta=0.0, Pw=0.8, Conf=0.95, designEf=1, dropOut=0):
    n = (((norm.ppf(1 - (1 - Conf) / 2) + norm.ppf(Pw)) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2))) / (delta ** 2)
    return abs(round((n / (1 - dropOut)) * designEf))

# Initialize history store
if "prop_test_history" not in st.session_state:
    st.session_state.prop_test_history = []

# Input form
p1 = st.sidebar.number_input("Proportion in 1st (Reference) Group (%)", value=50.0, min_value=0.0, max_value=100.0)
p2 = st.sidebar.number_input("Proportion in 2nd (Test) Group (%)", value=40.0, min_value=0.0, max_value=100.0)
delta = abs(p2 - p1)
power = st.sidebar.number_input("Power (%)", value=80.0, min_value=0.0, max_value=100.0)
drpt = st.sidebar.number_input("Drop-Out (%)", min_value=0.0, value=0.0, max_value=100.0)

x = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

if x == "Given":
    designEffect = st.sidebar.number_input("Design Effect", value=1.0, min_value=1.0)
    m = None
    ICC = None
else:
    m = st.sidebar.number_input("Number of clusters", min_value=2)
    ICC = st.sidebar.number_input("ICC", min_value=0.0)
    designEffect = 1 + (m - 1) * ICC
    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster Size (m)", value=m)
    col2.metric("Intra Class Correlation (ICC)", value=ICC)
    col3.metric("Design Effect", value=round(designEffect, 2))

# Calculate button
go = st.button("Calculate Sample Size")

# Helper to generate label for dropdown
def make_prop_test_history_label(p1, p2, power, drpt, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"Ref={p1}%, Test={p2}%, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
    else:
        return (f"Ref={p1}%, Test={p2}%, Power={power}%, DropOut={drpt}%, "
                f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

# Select from history
selected_history = None
selected_label = None

if st.session_state.prop_test_history:
    st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
    prop_test_options = [make_prop_test_history_label(**entry) for entry in st.session_state.prop_test_history]
    selected_label = st.selectbox("Choose a past input set:", prop_test_options, key="prop_test_history_selector")

    if selected_label:
        selected_history = next((item for item in st.session_state.prop_test_history
                                 if make_prop_test_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False
else:
    hist_submit = False

# Calculation condition
if go or hist_submit:
    if hist_submit and selected_history:
        # Use selected history
        p1 = selected_history["p1"]
        p2 = selected_history["p2"]
        power = selected_history["power"]
        drpt = selected_history["drpt"]
        designEffect = selected_history["designEffect"]
        delta = abs(p2 - p1)
    else:
        # Add current input to history
        new_entry = {
            "p1": p1,
            "p2": p2,
            "power": power,
            "drpt": drpt,
            "designEffect": designEffect,
            "m": m,
            "ICC": ICC,
            "method": x
        }
        st.session_state.prop_test_history.append(new_entry)

    confidenceIntervals = [0.8, 0.9, 0.97, 0.99, 0.999, 0.9999]
    out = []

    for conf in confidenceIntervals:
        sample_size = nSampleProp(p1=(p1 / 100), p2=(p2 / 100), delta=(delta / 100),
                                  Pw=(power / 100), Conf=conf,
                                  designEf=designEffect, dropOut=(drpt / 100))
        out.append(sample_size)

    df = pd.DataFrame({
        "Confidence Levels (%)": [cl * 100 for cl in confidenceIntervals],
        "Sample Size": out
    })

    dds = nSampleProp(p1=(p1 / 100), p2=(p2 / 100), delta=(delta / 100),
                      Pw=(power / 100), Conf=0.95,
                      designEf=designEffect, dropOut=(drpt / 100))

    st.write("The study would need a total sample size of:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: #48D1CC;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {2 * dds}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write(f"""Participants (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{dds}</span> individuals in each group) 
                to achieve a power of {power}% and 
                <span style="background-color: yellow; font-weight: bold; font-size: 26px;">95%</span> confidence level, 
                for detecting a difference in proportions of {delta}% 
                between the test and reference group, assuming that {p1}% of the subjects in the reference population have the factor of interest. 
                Design effect is **{round(designEffect, 1)}** with **{drpt}%** drop-out.""",
             unsafe_allow_html=True)

    st.subheader("📊 Sample Sizes at Other Confidence Levels")
    st.dataframe(df)

# Footer and references
st.markdown("---")
st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Two-Sample Proportion Hypothesis Test Sample Size Formula**")
st.latex(r"""
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (p_1(1 - p_1) + p_2(1 - p_2))}{\delta^2} \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")
st.markdown("""
- **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{1-beta} \)**: Power.
- **\( p_1 \)**, **\( p_2 \)**: Proportions in the first and second groups.
- **\( \delta \)**: Expected difference between the two proportions.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of cluster.
- **\( ICC \)**: Intra-cluster correlation coefficient.
""")

st.subheader("📌 References")
st.markdown("""
1. **Lemeshow, S., Hosmer Jr., D.W., Klar, J., Lwanga, S.K. (1990).** Adequacy of sample size in health studies. England: John Wiley & Sons.
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
