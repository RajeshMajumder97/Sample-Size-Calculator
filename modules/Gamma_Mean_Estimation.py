import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

def main():
    # st.set_page_config(page_title="StudySizer | Gamma Mean Estimation", page_icon="🧮")
    st.title("Sample Size Calculation for Gamma Mean: Mean Estimation (Gamma Distribution)")

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

    # Inject CSS override for number_input
    st.markdown(
        """
        <style>
        /* Hide the up/down spin buttons on number_input */
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield; /* Firefox */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Function for Gamma sample size (Absolute / Relative precision)
    def nSampleMean_Gamma(mu=None, sigma=None, d=None, r=None, Conf=0.95, designEf=1, dropOut=0, mode="Absolute"):
        Z = norm.ppf(1 - ((1 - Conf) / 2))
        CV = sigma / mu if (mu and sigma) else None  # compute CV if available
        
        if mode == "Absolute":
            # Absolute precision: n = (Z^2 * (CV*mu)^2) / d^2
            n = (Z**2 * (CV * mu)**2) / (d**2)
        else:
            # Relative precision: n = (Z^2 * CV^2) / r^2
            n = (Z**2 * (CV**2)) / (r**2)
        
        # Adjust for overdispersion, dropout, and design effect
        n = (n / (1 - dropOut)) * designEf
        return abs(round(n))


    # Initialize history store
    if "Ggamma_mean_history" not in st.session_state:
        st.session_state.Ggamma_mean_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    mu = st.sidebar.number_input("Expected Mean", value=35.0, min_value=0.01,format="%.6g", max_value=1e6, help="Enter expected mean of Gamma distribution (μ > 0).")
    sigma = st.sidebar.number_input("Standard Deviation (SD)", value=15.0, min_value=0.01,format="%.6g", help="Enter a value >0")
    ads = st.sidebar.radio("Choose Precision Option",options=['Absolute Precision', 'Relative Precision'],help="Absolute precision defines a fixed margin (e.g., ±2 units). Relative precision defines a percentage margin (e.g., ±5% of mean).")

    if ads == 'Absolute Precision':
        d = st.sidebar.number_input("Absolute Precision (d)", value=1.5, min_value=0.00,format="%.6g", max_value=100.0, help="Enter a value (e.g., 1.5)")
        d1 = d
        r = None
        mode = "Absolute"
    else:
        d = st.sidebar.number_input("Relative Precision(%)", value=5.0, min_value=0.00,format="%.6g", max_value=99.99, help="Enter a percentage value (e.g., 5%)")
        r = d / 100
        d1 = r * mu
        mode = "Relative"
        col1, col2, col3 = st.columns(3)
        col1.metric("Relative Precision(%)", value=d)
        col2.metric("Anticipated Mean", value=mu)
        col3.metric("Precision", value=round(d1, 2))

    if d1 == 0:
        st.error("Precision cannot be zero.")
        st.stop()

    drpt = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0,format="%.6g", max_value=50.0, help="Enter dropout percentage.")
    
    x = st.sidebar.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])
    if x == "Given":
        designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0,format="%.6g", min_value=1.0, help="Enter design effect directly.")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of Clusters (m)", min_value=2, format="%.6g", value=2)
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC)", min_value=0.0,format="%.6g", max_value=1.0, value=0.05)
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("Intra Class Correlation (ICC)", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # Helper for history label
    def make_Ggamma_mean_history_label(sigma, d, d1, drpt, designEffect, mu=None, m=None, ICC=None,method="Given", absolute='Absolute Precision', r=None, mode="Absolute"):
        if method == "Given":
            if absolute == 'Absolute Precision':
                return f"Sigma={sigma}, Precision={absolute}, d={round(d1, 2)}, μ={mu}, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return f"Sigma={sigma}, Precision={absolute}({round(r*100,2)}%), d={round(d1, 2)}, μ={mu}, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            if absolute == 'Absolute Precision':
                return (f"Sigma={sigma}, Precision={absolute}, d={round(d1, 2)}, μ={mu}, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")
            else:
                return (f"Sigma={sigma}, Precision={absolute}({round(r*100,2)}%), d={round(d1, 2)}, μ={mu}, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None
    if st.session_state.Ggamma_mean_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        mean_options = [make_Ggamma_mean_history_label(**entry) for entry in st.session_state.Ggamma_mean_history]
        selected_label = st.selectbox("Choose a past input set:", mean_options, key="Ggamma_mean_history_selector")
        if selected_label:
            selected_history = next((item for item in st.session_state.Ggamma_mean_history
                                    if make_Ggamma_mean_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False

    if go or hist_submit:
        tabs = st.tabs(["Tabulate", "Precision V/s Confidence Table", "Visualisation"])

        with tabs[0]:
            if hist_submit and selected_history:
                sigma = selected_history["sigma"]
                d1 = selected_history["d1"]
                mu = selected_history["mu"]
                drpt = selected_history["drpt"]
                ads = selected_history["absolute"]
                designEffect = selected_history["designEffect"]
                r = selected_history.get("r", None)
                mode = selected_history.get("mode", "Absolute")
            else:
                new_entry = {
                    "sigma": sigma,
                    "d1": d1,
                    "drpt": drpt,
                    "designEffect": designEffect,
                    "m": m,
                    "ICC": ICC,
                    "method": x,
                    "absolute": ads,
                    "mu": mu,
                    "d": d,
                    "r": r,
                    "mode": mode
                }
                st.session_state.Ggamma_mean_history.append(new_entry)

            confidenceIntervals = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
            out = []
            for conf in confidenceIntervals:
                if mode == "Absolute":
                    sample_size = nSampleMean_Gamma(mu=mu, sigma=sigma, d=d1, Conf=conf,designEf=designEffect, dropOut=(drpt/100),mode="Absolute")
                else:
                    sample_size = nSampleMean_Gamma(mu=mu, sigma=sigma, r=r, Conf=conf,designEf=designEffect, dropOut=(drpt/100),mode="Relative")
                out.append(sample_size)

            df = pd.DataFrame({
                "Confidence Levels (%)": [cl * 100 for cl in confidenceIntervals],
                "Sample Size": out
            })

            dds = out[0]  # sample size at 95%

            if ads == 'Absolute Precision':
                st.write(f"Assuming a **Gamma distribution** with expected mean= {mu} and SD= {sigma}, "
                         f"the study would require a sample size of:")
                st.markdown(f"""
                <div style="display: flex; justify-content: center;">
                <div style=" font-size: 36px; font-weight: bold; background-color: #48D1CC;
                padding: 10px; border-radius: 10px; text-align: center;">
                {dds}
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"""for estimating the mean with absolute precision **{d}** at 
                95% confidence, considering DE={round(designEffect,1)} and {drpt}% dropout.""")

            else:
                st.write(f"Assuming a **Gamma distribution** with expected mean= {mu} and SD= {sigma}, "
                         f"the study would require a sample size of:")
                st.markdown(f"""
                <div style="display: flex; justify-content: center;">
                <div style=" font-size: 36px; font-weight: bold; background-color: #48D1CC;
                padding: 10px; border-radius: 10px; text-align: center;">
                {dds}
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"""for estimating the mean with relative precision **({mu}×{d}%=){round(d1,1)}** at 
                95% confidence, considering DE={round(designEffect,1)} and {drpt}% dropout.""")

            st.subheader("List of Sample Sizes at other Confidence Levels")
            st.dataframe(df)
        with tabs[1]:
            if(ads=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no cross table is available.")
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
                        d1= (d_val*mu)
                        ss = nSampleMean_Gamma(mu=mu, sigma=sigma, d=d1, Conf=conf,designEf=designEffect, dropOut=(drpt/100),mode="Absolute")
                        #nSampleMean_Gamma(sigma=sigma, d=d1, Conf=conf,designEf=designEffect, dropOut=(drpt / 100), phi=phi)
                        cross_table.iloc[i, j] = ss
                # Label table
                cross_table.index.name = "Confidence levels (%)"
                cross_table.columns.name = "Precision(%)"
                st.dataframe(cross_table)
                st.write("**Rows are Confidence levels; Columns are Precision**")
                #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            if(ads=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no visualization is available.")
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

        # Keep tabs[1] and tabs[2] same as before (cross table + plots), just replace calls with nSampleMean_Gamma
        # (due to message length, I’ve truncated here — but everything else remains intact with Gamma wording + phi)

    # References section
    st.markdown("---")
    with st.expander("Show the formula and the derivation"):
        st.subheader("📌 Derivation — Gamma mean sample size (in terms of CV)")

        st.markdown("**1. Notation / facts for Gamma:**")
        st.markdown(r"""
        - Let X∼Gamma(k,θ) with mean μ=E[X]=kθ and variance σ^2=Var(X)=kθ^2.  
        - Coefficient of variation:
        """)
        st.latex(r"""CV=\frac{\sigma}{\mu}=\frac{1}{\sqrt{k}}""")

        st.markdown("**2. Confidence interval for the sample mean (normal approximation):**")
        st.latex(r"""\bar{X} \pm Z_{1-\alpha/2}\,\sqrt{\frac{\sigma^2}{n}}""")
        st.markdown("To achieve a margin of error \(d\) (absolute precision) we require:")
        st.latex(r"""Z_{1-\alpha/2}\,\sqrt{\frac{\sigma^2}{n}}\le d""")

        st.markdown("Square and solve for \(n\):")
        st.latex(r"""\displaystyle n \ge \frac{Z_{1-\alpha/2}^2\,\sigma^2}{d^2}""")

        st.markdown("Substitute variance in terms of CV:")
        st.latex(r"""\displaystyle n \;=\; \frac{Z_{1-\alpha/2}^2\,(CV\cdot\mu)^2}{d^2} \qquad\text{(Absolute precision)}""")

        st.markdown("**3. Relative precision form (margin = \(r\mu\)):**")
        st.markdown("If \(d = r\mu\) (so \(r\) is the fraction, e.g. 0.05 for 5%), substitute into the absolute formula:")
        st.latex(r"""\displaystyle n \;=\; \frac{Z_{1-\alpha/2}^2\,(CV\cdot\mu)^2}{(r\mu)^2} \;=\; \frac{Z_{1-\alpha/2}^2\,CV^2}{r^2} \qquad\text{(Relative precision)}""")
        st.markdown("Note: for the relative formula the mean \(\mu\) cancels out — you only need CV and \(r\).")

        st.markdown("**4. Adjustments commonly applied in practice:**")
        st.markdown(r"""
        - Design effect (DE): multiply by DE (if cluster design).  
        - Dropout (expressed as proportion): divide by \((1 - \text{dropout})\).
        """)

        st.markdown("Applying those adjustments gives the final working formulas:")

        st.latex(r"""\boxed{\;
        n_{\text{abs}} \;=\; \frac{Z_{1-\alpha/2}^2\,(CV\cdot\mu)^2}{d^2} \times DE \div (1-\text{Dropout})
        \; }""")

        st.latex(r"""\boxed{\;
        n_{\text{rel}} \;=\; \frac{Z_{1-\alpha/2}^2\,CV^2}{r^2} \times DE \div (1-\text{Dropout})
        \; }""")

        st.markdown("**Practical notes:**")
        st.markdown(r"""
        - Always round \(n\) **up** to the next integer.  
        - If CV is unknown, estimate from pilot data or literature.  
        - The normal approximation for \(\bar{X}\) is reasonable for moderate-to-large \(n\); for very small \(n\) or extreme skewness consider simulation.  
        - For Absolute form you must supply \(\mu\) (or equivalently \(\sigma\) and \(\mu\)); for Relative form you only need CV and \(r\).
        """)

    # Citation
    st.markdown("---")
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    st.markdown(f""" *Majumder, R. (2025). StudySizer: A sample size calculator (Gamma Mean Estimation, Version 0.1.0). 
    Available online: [https://studysizer.streamlit.app/](https://studysizer.streamlit.app/). Accessed on {now}.* """)
    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**") 
    st.markdown("**Email:** rajeshnbp9051@gmail.com") 
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
