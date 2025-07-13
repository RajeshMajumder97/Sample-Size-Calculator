# Revised Python function for sample size calculation for multiple linear regression
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

def main():
    #    st.set_page_config(page_title="StydySizer | Linear Regression", page_icon="🧮")

    st.title("Sample Size Calculator for Multiple Linear Regression")
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
    # Sample size calculation function
    def sample_size_linear_regression(R_squared, k, alpha=0.05, power=0.8, designEf=1.0, dropOut=0.0):
        Z_alpha = norm.ppf(1 - alpha / 2)
        Z_beta = norm.ppf(power)
        numerator = (Z_alpha + Z_beta) ** 2 * (1 - R_squared)
        denominator = R_squared / k
        n = numerator / denominator + k + 1
        adjusted_n = np.ceil(n / (1 - dropOut) * designEf)
        return int(adjusted_n)

    # Store history
    if "reg_history" not in st.session_state:
        st.session_state.reg_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")
    # Sidebar inputs
    R_squared = st.sidebar.number_input("Expected R²", min_value=0.0001, max_value=0.99, value=0.13,help="Enter a decimal value (e.g., 0.05)")
    k = st.sidebar.number_input("Number of Predictors (k)", min_value=1, value=5,help="Enter an integer value (e.g., 5)")
    power = st.sidebar.number_input("Power (%)", min_value=50.0, max_value=99.0, value=80.0,help="Enter a percentage value (e.g., 80%)")
    drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0,help="Enter a percentage value (e.g., 1%)")
    method = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

    if method == "Given":
        designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0,help="Enter a decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of Clusters (m)", min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("ICC", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    go = st.button("Calculate Sample Size")

    # History label helper
    def make_reg_label(R_squared, k, power, drp, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"R²={R_squared}, Predictors={k}, Power={power}%, DropOut={drp}%, DE={round(designEffect, 2)}"
        else:
            return f"R²={R_squared}, Predictors={k}, Power={power}%, DropOut={drp}%, DE={round(designEffect, 2)}, m={m}, ICC={ICC}"

    # History selector
    selected_history = None
    if st.session_state.reg_history:
        st.subheader("📜 Select from Past Inputs")
        reg_labels = [make_reg_label(**item) for item in st.session_state.reg_history]
        selected = st.selectbox("Choose a past input set:", reg_labels, key="reg_history_selector")
        if selected:
            selected_history = next(item for item in st.session_state.reg_history if make_reg_label(**item) == selected)
            recalc = st.button("🔁 Recalculate")
        else:
            recalc = False
    else:
        recalc = False

    if go or recalc:
        if recalc and selected_history:
            R_squared = selected_history["R_squared"]
            k = selected_history["k"]
            power = selected_history["power"]
            drp = selected_history["drp"]
            designEffect = selected_history["designEffect"]
        else:
            st.session_state.reg_history.append({
                "R_squared": R_squared,
                "k": k,
                "power": power,
                "drp": drp,
                "designEffect": designEffect,
                "m": m,
                "ICC": ICC,
                "method": method
            })

        conf_levels = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
        results = []
        for conf in conf_levels:
            n = sample_size_linear_regression(R_squared=R_squared, k=k, alpha=1 - conf, power=power / 100,
                                            designEf=designEffect, dropOut=drp / 100)
            results.append(n)

        df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})
        n95 = sample_size_linear_regression(R_squared, k, alpha=0.05, power=power / 100,
                                            designEf=designEffect, dropOut=drp / 100)

        st.write("The required total sample size is:")
        st.markdown(f"""
        <div style='display: flex; justify-content: center;'>
            <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
                {n95}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""To detect an overall R² = {R_squared} with {power}% power and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, using {k} predictors, with design effect of {round(designEffect, 2)} and drop-out rate of {drp}%.""", unsafe_allow_html=True)
        st.subheader("Sample Sizes at Other Confidence Levels")
        st.dataframe(df)

    st.markdown("---")
    st.subheader("📌 Formula Used")
    st.latex(r"""
    n = \left(\left(\frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (1 - R^2)}{R^2 / k} \right) + k + 1 \right)\times \frac{DE}{1 - \text{Dropout\%}}
    """)
    st.markdown("### Design Effect (if clusters are used):")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)

    st.subheader("📌 Description of Parameters")
    st.markdown("""
    - **\( Z_{1-alpha/2} \)**: Z-value for the confidence level (e.g., 1.96 for 95% confidence).  
    - **\( Z_{1-beta} \)**: Z-value for power (e.g., 0.84 for 80% power).  
    - **\( R^2 \)**: Anticipated coefficient of determination.  
    - **\( k \)**: Number of predictor variables.  
    - **\( DE \)**: Design Effect.  
    - **\( m \)**: Number of clusters.  
    - **\( ICC \)**: Intra-cluster correlation coefficient.
    - **Dropout%**: Anticipated percentage of dropout in the study.
    """)

    st.markdown("---")
    st.subheader("📌 References")
    st.markdown("""
    1. **Dupont WD, Plummer WD.** Power and sample size calculations for studies involving linear regression. Controlled Clinical Trials 1998; 19:589-601.  
    2. **Green, S. B. (1991).** How many subjects does it take to do a regression analysis? *Multivariate Behavioral Research*, 26(3), 499–510.
    """)

    st.markdown("---")
    st.subheader("Citation")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app](https://rajeshmajumderblog.netlify.app)")
