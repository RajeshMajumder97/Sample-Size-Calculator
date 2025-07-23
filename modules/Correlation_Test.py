import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm


def main():
    #    st.set_page_config(page_title=" StydySizer | Correlation Hypothesis Test", page_icon="🧮")
    #
    #    st.markdown("""
    #        <style>
    #        button[data-testid="stBaseButton-header"] {
    #            display: none !important;
    #        }
    #        </style>
    #    """, unsafe_allow_html=True)

    st.title("Sample Size Calculation for Correlation Test | H₀: ρ = ρ₀ vs H₁: ρ ≠ ρ₀")
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
    # Fisher's Z-transformation sample size function
    def nSampleCorrelation(rho0=0.0, rho1=0.3, Pw=0.8, Conf=0.95, designEf=1, dropOut=0):
        z0 = 0.5 * np.log((1 + rho0) / (1 - rho0))
        z1 = 0.5 * np.log((1 + rho1) / (1 - rho1))
        delta_z = abs(z1 - z0)
        n = ((norm.ppf(1 - (1 - Conf) / 2) + norm.ppf(Pw)) / delta_z) ** 2 + 3
        return int(np.ceil((n / (1 - dropOut)) * designEf))

    if "corr_history" not in st.session_state:
        st.session_state.corr_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")
    # Sidebar inputs
    rho0 = st.sidebar.number_input("Null hypothesis correlation (ρ₀)", value=0.0, min_value=-0.99, max_value=0.99,help="Enter a decimal value (e.g., 0.0)")
    rho1 = st.sidebar.number_input("Expected correlation (ρ₁)", value=0.3, min_value=-0.99, max_value=0.99,help="Enter a decimal value (e.g., 0.3)")

    if rho0 == rho1:
        st.sidebar.warning("ρ₀ and ρ₁ cannot be the same.")
        st.stop()

    power = st.sidebar.number_input("Power (%)", value=80.0, min_value=50.0, max_value=99.0,help="Enter a percentage value (e.g., 80%)")
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
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        corr_labels = [make_corr_label(**item) for item in st.session_state.corr_history]
        selected = st.selectbox("Choose a past input set:", corr_labels, key="corr_history_selector")
        if selected:
            selected_history = next(item for item in st.session_state.corr_history if make_corr_label(**item) == selected)
            recalc = st.button("🔁 Recalculate from Selected History")
        else:
            recalc = False
    else:
        recalc = False

    if go or recalc:
        tabs = st.tabs(["Tabulate", "Power V/s Confidelce Table" ,"Visualisation"])
        with tabs[0]:
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

            conf_levels = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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

        with tabs[1]:
            # D efine power and confidence levels
            powers = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
            conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

            st.subheader("📈 Sample Size Cross Table for Different Powers and Confidence Levels")

            power_labels = [f"{int(p * 100)}%" for p in powers]
            conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
            cross_table = pd.DataFrame(index=conf_labels, columns=power_labels)
            # Fill the cross table
            for i, conf in enumerate(conf_levels):
                for j, power_val in enumerate(powers):
                    ss = nSampleCorrelation(rho0=rho0, rho1=rho1, Pw= power_val, Conf=conf, designEf=designEffect, dropOut=(drp / 100)) 
                    cross_table.iloc[i, j] = ss
            # Label table
            cross_table.index.name = "Confidence Level (%)"
            cross_table.columns.name = "Power (%)"
            st.dataframe(cross_table)
            st.write("**Rows are Confidence Levels; Columns are Powers**")
            #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            ##
            import matplotlib.pyplot as plt

            powers = [int(col.strip('%')) for col in cross_table.columns]
            conf_levels = [int(row.strip('%')) for row in cross_table.index]

            # Sort both for consistent plotting
            powers_sorted = sorted(powers)
            conf_levels_sorted = sorted(conf_levels)

            # Convert back to string labels
            power_labels = [f"{p}%" for p in powers_sorted]
            conf_labels = [f"{cl}%" for cl in conf_levels_sorted]

            # Plotting
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Power curves at selected Confidence Levels (primary y-axis)
            conf_levels_to_plot = [90, 95, 97, 99]
            for cl in conf_levels_to_plot:
                cl_label = f"{cl}%"
                if cl_label in cross_table.index:
                    sample_sizes = cross_table.loc[cl_label, power_labels].astype(float).tolist()
                    ax1.plot(sample_sizes, powers_sorted, marker='o', linestyle='-', label=f'Power at {cl_label} CL')

            ax1.set_xlabel("Sample Size")
            ax1.set_ylabel("Power (%)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim([60, 100])
            ax1.grid(True)

            # Alpha curves at selected Power Levels (secondary y-axis)
            power_levels_to_plot = [80, 85, 90, 95]
            ax2 = ax1.twinx()
            for pwr in power_levels_to_plot:
                pwr_label = f"{pwr}%"
                if pwr_label in cross_table.columns:
                    sample_sizes = cross_table[pwr_label].astype(float).tolist()
                    alpha_vals = [100 - int(idx.strip('%')) for idx in cross_table.index]
                    ax2.plot(sample_sizes, alpha_vals, marker='s', linestyle='--', label=f'Alpha at {pwr_label} Power')

            ax2.set_ylabel("Alpha Level (%)", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim([0, 30])

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            #ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1.05, 0.5), ncol=2)

            # Title and layout
            plt.title("Sample Size vs Power and Alpha Level (Multiple Lines)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
            #plt.tight_layout(rect=[0, 0.1, 1, 1])
            # Show in Streamlit
            st.pyplot(fig)
            st.markdown("---")
            with st.expander("💡Show the Interpretation of the plot"):
                st.markdown("- This plot demonstrates **how sample size influences both statistical power and the risk of Type I error (alpha)**—two critical factors in designing reliable health research.")
                st.markdown("- The **Left Y-Axis (Blue)**, solid lines represent the probability of correctly detecting a true effect (power), which increases with larger sample sizes, improving the study's ability to identify meaningful (clinical) differences.")
                st.markdown("- On the other hand, the **Right Y-Axis (Orange/Yellow)**, dashed lines indicate the likelihood of a false positive result (alpha), which typically decreases with larger samples, reflecting a more conservative test. Conversely, increasing alpha reduces the required sample size to achieve a given power, but increases the risk of Type I error. For an example, if you want 80% power, increasing alpha (e.g., from 0.01 to 0.05) means you need fewer subjects.")
                st.markdown("- **Points where the power and alpha curves intersect** represent sample sizes where the chance of detecting a real effect (power) equals the chance of making a false claim (alpha)—an undesirable scenario. In health research, we strive for power to be much higher than alpha to ensure that findings are both valid and clinically trustworthy, in line with the principles of the most powerful statistical tests. ")


    st.markdown("---")
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation")
        st.latex(r"""
        n = \left(\left( \frac{Z_{1-\alpha/2} + Z_{1-\beta}}{\frac{1}{2} \ln\left(\frac{1 + \rho_1}{1 - \rho_1}\right) - \frac{1}{2} \ln\left(\frac{1 + \rho_0}{1 - \rho_0}\right)} \right)^2 + 3\right) \times \frac{DE}{1 - \text{Dropout\%}}
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
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)


        st.markdown("---")
        st.subheader("References")
        st.markdown("""
        1. **Hulley et al. (2013).** Designing Clinical Research. 4th Ed. Lippincott Williams & Wilkins.
        2. **Cohen, J. (1988).** Statistical Power Analysis for the Behavioral Sciences. Routledge.
        """)

    st.markdown("---")
    st.subheader("Citation")
    from datetime import datetime
    # Get current date and time
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    # Citation with access timestamp
    st.markdown(f"""
    *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.netlify.app/](https://studysizer.netlify.app/). Accessed on {now}. [https://doi.org/10.5281/zenodo.16375937](https://doi.org/10.5281/zenodo.16375937).*
    """)


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")