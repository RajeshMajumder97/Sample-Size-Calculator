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

    rho = st.sidebar.number_input("Expected ICC (ρ)", value=0.8, min_value=0.01, max_value=0.99,format="%.6g",help= "Enter a decimal value (e.g., 0.05)")

    prec_type = st.sidebar.radio("Choose Precision Option", options=['Absolute Precision', 'Relative Precision'])
    if prec_type == 'Absolute Precision':
        d = st.sidebar.number_input("Absolute Precision (d)", value=0.05, min_value=0.001, max_value=0.50,format="%.6g", help="Enter a decimal value (e.g., 0.05)")
    else:
        d_percent = st.sidebar.number_input("Relative Precision (%)", value=5.0, min_value=0.01, max_value=50.0,format="%.6g", help="Enter a percentage value (e.g., 5%)")
        d = (d_percent / 100) * rho
        col1, col2 = st.columns(2)
        col1.metric("Relative Precision (%)", value=d_percent)
        col2.metric("Precision (d)", value=round(d, 4))

    k = st.sidebar.number_input("Number of Raters / Repeated Measures (k)", min_value=2, value=3,format="%.6g", help="Enter an integer value (e.g., 3)")
    drpt = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0,format="%.6g",help="Enter a percentage value (e.g., 1%)") / 100

    x = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

    if x == "Given":
        designEffect = st.sidebar.number_input("Design Effect", value=1.0, min_value=1.0,format="%.6g",help="Enter a decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of clusters", min_value=2, value=4,format="%.6g",help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", value=0.05, min_value=0.0, max_value=1.0,format="%.6g",help= "Enter a decimal value (e.g., 0.05)")
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
        tabs = st.tabs(["Tabulate", "Precision V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
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

            confidenceIntervals = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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
        with tabs[1]:
            if(prec_type=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no cross table will be generated.")
                cross_table=None
            else:
                # D efine power and confidence levels
                precision = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
                conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

                st.subheader("📈 Sample Size Cross Table for Different Precisions and Confidence Levels")

                power_labels = [f"{int(p * 100)}%" for p in precision]
                conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
                cross_table = pd.DataFrame(index=conf_labels, columns=power_labels)
                # Fill the cross table
                for i, conf in enumerate(conf_levels):
                    for j, d_val in enumerate(precision):
                        d = d_val * rho
                        Z_alpha = norm.ppf(1 - (1 - conf) / 2)
                        ss = nSampleICC_k(Z_alpha, rho, d, k, designEf=designEffect, dropOut=drpt)
                        cross_table.iloc[i, j] = ss
                # Label table
                cross_table.index.name = "Confidence levels (%)"
                cross_table.columns.name = "Precision(%)"
                st.dataframe(cross_table)
                st.write("**Rows are Confidence levels; Columns are Precision**")
                #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            if(prec_type=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no visualization will be available.")
            else:
                import matplotlib.pyplot as plt

                # Prepare precision and confidence values from cross_table
                precision = [int(col.strip('%')) for col in cross_table.columns]
                conf_levels = [int(row.strip('%')) for row in cross_table.index]

                precision_sorted = sorted(precision)
                conf_levels_sorted = sorted(conf_levels)

                power_labels = [f"{p}%" for p in precision_sorted]
                conf_labels = [f"{cl}%" for cl in conf_levels_sorted]

                col1, col2 = st.columns(2)

                # === Plot 1: Sample Size vs Precision at fixed Confidence Levels ===
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    conf_levels_to_plot = [80, 95, 97, 99]
                    for cl in conf_levels_to_plot:
                        cl_label = f"{cl}%"
                        if cl_label in cross_table.index:
                            sample_sizes = cross_table.loc[cl_label, power_labels].astype(float).tolist()
                            ax1.plot(precision_sorted, sample_sizes, marker='o', linestyle='-', label=f'CL {cl_label}')
                    ax1.set_title("Sample Size vs Precision")
                    ax1.set_xlabel("Precision (%)")
                    ax1.set_ylabel("Sample Size")
                    ax1.grid(True)
                    ax1.legend(title="Confidence Level")
                    st.pyplot(fig1)

                # === Plot 2: Sample Size vs Confidence Level at fixed Precision Levels ===
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    precision_levels_to_plot = [2, 3, 5, 10]
                    for pr in precision_levels_to_plot:
                        pr_label = f"{pr}%"
                        if pr_label in cross_table.columns:
                            sample_sizes = cross_table[pr_label].astype(float).tolist()
                            ax2.plot(conf_levels_sorted, sample_sizes, marker='s', linestyle='--', label=f'Precision {pr_label}')
                    ax2.set_title("Sample Size vs Confidence Level")
                    ax2.set_xlabel("Confidence Level (%)")
                    ax2.set_ylabel("Sample Size")
                    ax2.grid(True)
                    ax2.legend(title="Precision")
                    st.pyplot(fig2)

                st.markdown("---")
                with st.expander("💡Show the Interpretation of the plots"):
                    st.markdown("### Plot 1: Sample Size vs Precision")
                    st.markdown("- As **precision becomes tighter (i.e., smaller %) the required sample size increases** exponentially.")
                    st.markdown("- Higher confidence levels (e.g., 99%) require larger sample sizes than lower ones (e.g., 80%) for the same precision.")
                    st.markdown("### Plot 2: Sample Size vs Confidence Level")
                    st.markdown("- As **confidence level increases**, so does the **required sample size** to ensure the estimate remains within the desired precision.")
                    st.markdown("- At lower confidence (e.g., 70–80%), sample size requirements are modest, but they grow rapidly beyond 95%, especially at tighter precision levels.")
        

    st.markdown("---")
    with st.expander("Show the formula and the references"):
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
    st.subheader("Citation")
    from datetime import datetime
    # Get current date and time
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    # Citation with access timestamp
    st.markdown(f"""
    *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.streamlit.app/](https://studysizer.streamlit.app/). Accessed on {now}.*
    """)


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")

