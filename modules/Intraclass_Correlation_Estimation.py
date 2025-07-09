import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm


def main():
    st.title("Sample Size Calculation for ICC Estimation (Precision-based)")

    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-header"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def nSampleICC_k(Z_alpha, rho, d, k, designEf=1.0, dropOut=0.0):
        numerator = 8 * (Z_alpha ** 2) * ((1 - rho) ** 2) * ((1 + (k - 1) * rho) ** 2)
        denominator = k * (k - 1) * (d ** 2)
        N = 1 + numerator / denominator
        return round((N / (1 - dropOut)) * designEf)

    # Initialize history
    if "icc_estimation_history" not in st.session_state:
        st.session_state.icc_estimation_history = []

    st.sidebar.header("🔧 Input Parameters")

    rho = st.sidebar.number_input("Expected ICC (ρ)", value=0.8, min_value=0.01, max_value=0.99,help= "Enter a decimal value (e.g., 0.05)")

    prec_type = st.sidebar.radio("Choose Precision Option", options=['Absolute Precision', 'Relative Precision'])
    if prec_type == 'Absolute Precision':
        d = st.sidebar.number_input("Absolute Precision (d)", value=0.05, min_value=0.001, max_value=0.50, help="Enter a decimal value (e.g., 0.05)")
    else:
        d_percent = st.sidebar.number_input("Relative Precision (%)", value=5.0, min_value=0.01, max_value=50.0, help="Enter a percentage value (e.g., 5%)")
        d = (d_percent / 100) * rho
        col1, col2 = st.columns(2)
        col1.metric("Relative Precision (%)", value=d_percent)
        col2.metric("Precision (d)", value=round(d, 4))

    k = st.sidebar.number_input("Number of Raters / Repeated Measures (k)", min_value=2, value=3, help="Enter an integer value (e.g., 3)")
    drpt = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0,help="Enter a percentage value (e.g., 1%)") / 100

    x = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

    if x == "Given":
        designEffect = st.sidebar.number_input("Design Effect", value=1.0, min_value=1.0,help="Enter an decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of clusters", min_value=2, value=4,help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", value=0.05, min_value=0.0, max_value=1.0,help= "Enter a decimal value (e.g., 0.05)")
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("Intra-class Correlation", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    go = st.button("Calculate Sample Size")

    def make_history_label(rho, d, k, designEffect, drpt, m=None, ICC=None, method="Given", prec_type="Absolute Precision", d_percent=None):
        if prec_type == 'Absolute Precision':
            d_label = f"Precision={round(d, 3)}"
        else:
            d_label = f"Precision={round(d_percent, 1)}% of ICC = {round(d, 3)}"

        if method == "Given":
            return f"ICC={rho}, {d_label}, Raters={k}, DE={round(designEffect,2)}, Dropout={int(drpt*100)}%"
        else:
            return (f"ICC={rho}, {d_label}, Raters={k}, Dropout={int(drpt*100)}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    selected_history = None
    selected_label = None

    if st.session_state.icc_estimation_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        options = [make_history_label(**entry) for entry in st.session_state.icc_estimation_history]
        selected_label = st.selectbox("Choose a past input set:", options, key="icc_estimation_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.icc_estimation_history
                                     if make_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False

    if go or hist_submit:
        if hist_submit and selected_history:
            rho = selected_history["rho"]
            d = selected_history["d"]
            k = selected_history["k"]
            designEffect = selected_history["designEffect"]
            drpt = selected_history["drpt"]
            m = selected_history.get("m")
            ICC = selected_history.get("ICC")
            x = selected_history.get("method", "Given")
        else:
            new_entry = {
                "rho": rho, "d": d, "k": k,
                "designEffect": designEffect, "drpt": drpt,
                "m": m, "ICC": ICC, "method": x,
                "prec_type": prec_type
            }
            if prec_type == 'Relative Precision':
                new_entry['d_percent'] = d_percent
            st.session_state.icc_estimation_history.append(new_entry)

        confidenceIntervals = [0.8,0.9,0.97,0.99,0.999,0.9999]
        out = []

        for conf in confidenceIntervals:
            Z_alpha = norm.ppf(1 - (1 - conf) / 2)
            sample_size = nSampleICC_k(Z_alpha, rho, d, k, designEf=designEffect, dropOut=drpt)
            out.append(sample_size)

        df = pd.DataFrame({
            "Confidence Levels (%)": [cl * 100 for cl in confidenceIntervals],
            "Sample Size": out
        })

        Z_alpha_95 = norm.ppf(1 - (1 - 0.95) / 2)
        sample_size_95 = nSampleICC_k(Z_alpha_95, rho, d, k, designEf=designEffect, dropOut=drpt)

        st.write("The reliability study design would require a sample size of:")
        st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <div style="
                font-size: 36px;
                font-weight: bold;
                background-color: #48D1CC;
                padding: 10px;
                border-radius: 10px;
                text-align: center;">
                {sample_size_95}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prec_type == 'Absolute Precision':
            prec_note = f"absolute precision **{round(d*100,2)}**"
        else:
            prec_note = f"relative precision **{d_percent}%** of ICC = **{round(d,3)}**"

        st.write(f"""for the estimation of Intraclass Correlation with <span style="font-weight: bold; font-size: 26px;">95%</span>  confidence level, 
        assuming **{k}** raters, expected ICC **{rho}**, {prec_note}, 
        design effect **{round(designEffect,2)}**, and **{int(drpt*100)}%** dropout.""",unsafe_allow_html=True)

        st.subheader("List of Sample Sizes at other Confidence Levels")
        st.dataframe(df)

    st.markdown("---")
    st.subheader("📌 Formula for ICC Estimation Sample Size")
    st.latex(r"""
    N = 1 + \frac{8Z_{1-\alpha/2}^2 (1 - \rho)^2 (1 + (k - 1)\rho)^2}{k(k - 1)d^2} \times \frac{DE}{1 - \text{Dropout\%}}
    """)

    st.markdown("### **Design Effect Calculation (if clusters are used):**")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)


    st.subheader("📌 Description of Parameters")
    st.markdown("""
    - **\( rho \)**: Expected ICC
    - **\( d \)**: Absolute precision (or derived from relative % of ICC)
    - **\( Z_{1-alpha/2} \)**: Z-score for confidence level
    - **\( k \)**: Number of raters (replicates per subject)
    - **Design Effect**: Accounts for clustering
    - **Dropout%**: Expected loss of data
    """)

    st.markdown("---")
    st.subheader("📌 Reference")
    st.markdown("""
    Bonett, D. G. (2002). Sample size requirements for estimating intraclass correlations with desired precision. *Statistics in Medicine*, 21(9), 1331–1335.
    """)

    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")

