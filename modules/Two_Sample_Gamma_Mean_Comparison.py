import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
#from Functions.SSound import *

def main():

    st.title("Sample Size Calculation for Comparing Two Means Assuming Gamma Distribution | (H₀: μ₁ = μ₂)")
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
    
    # Function: Sample size for Poisson rate comparison (GLM approach with person-time, design effect, dropout)
    def nSampleGammaGLM(mu0, mu1, sd0, sd1, Q0=0.5, Q1=0.5, alpha=0.05, power=0.8, designEffect=1.0, dropout=0.0):
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        # Convert SD to shape (κ = μ² / σ²)
        kappa0 = (mu0 ** 2) / (sd0 ** 2)
        kappa1 = (mu1 ** 2) / (sd1 ** 2)

        # Log difference in means
        log_diff = np.log(mu0) - np.log(mu1)
        if log_diff == 0:
            return np.nan 
        
        # Variance term (from GLM gamma log-link derivation)
        variance_term = (1 / (Q1 * kappa1)) + (1 / (Q0 * kappa0))
        
        sqrt_N = (z_alpha + z_beta) * np.sqrt(variance_term) / abs(log_diff)
        N = sqrt_N ** 2
        N_adj = N * designEffect / (1 - dropout)
        return np.ceil(N_adj)

    # Initialize history store
    if "gamma_mean_history" not in st.session_state:
        st.session_state.gamma_mean_history = []

    # Sidebar inputs
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    mu0 = st.sidebar.number_input("Mean in Control Group (μ₀)", value=10.0, min_value=0.01, format="%.6g", help="Enter mean value for the control group (μ₀ > 0)")
    sd0 = st.sidebar.number_input("Standard Deviation in Control Group (σ₀)", value=3.0, min_value=0.01, format="%.6g", help="Enter standard deviation for the control group (σ₀ > 0)")

    mu1 = st.sidebar.number_input("Mean in Treatment Group (μ₁)", value=8.0, min_value=0.01, format="%.6g", help="Enter mean value for the treatment group (μ₁ > 0)")
    sd1 = st.sidebar.number_input("Standard Deviation in Treatment Group (σ₁)", value=2.5, min_value=0.01, format="%.6g", help="Enter standard deviation for the treatment group (σ₁ > 0)")

    design = st.sidebar.radio("Choose Group proportion", ["Equal proportion", "Unequal proportion"])
    st.sidebar.text("Choose Unequal proportion when the study groups are not equally sampled.")
    if design == "Equal proportion":
        Q0 = 0.5
        Q1 = 0.5
        col1, col2 = st.columns(2)
        col1.metric("Control Group Q0", value=Q0)
        col2.metric("Treatement Group Q1", value=Q1)
    else:
        Q0 = st.sidebar._number_input("Enter the Control group proportion (Q0) (%)", min_value=0.0, max_value=99.99, value=55.0, format="%.6g", help="Enter a percentage value (e.g., 55%)") / 100
        Q1 = 1 - Q0
        col1, col2 = st.columns(2)
        col1.metric("Control Group Q0", value=round(Q0, 2))
        col2.metric("Treatement Group Q1", value=round(Q1, 2))

    power = st.sidebar.number_input("Power (%)", min_value=50.0, max_value=99.9, value=80.0, format="%.6g", help="Enter a percentage value (e.g., 80%)") / 100
    drpt = st.sidebar.number_input("Drop-Out (%)", value=0.0, max_value=50.0, min_value=0.0, format="%.6g", help="Enter a percentage value (e.g., 1%)") / 100

    # Design Effect
    design_method = st.sidebar.radio("Design Effect Option:", ["Given", "Calculate"])
    if design_method == "Given":
        designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0, format="%.6g", help="Enter an decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of Clusters (m)", min_value=2, value=4, format="%.6g", help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-cluster Correlation (ICC) for clustering", min_value=0.0, max_value=1.0, value=0.05, format="%.6g", help="Enter a decimal value (e.g., 0.05)")
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("Intra Class Correlation (ICC)", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # Helper to generate label for dropdown
    def make_gamma_mean_history_label(mu0, mu1, sd0, sd1, Q0, Q1, power, drpt, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"mu0={mu0}, mu1={mu1},sd0={sd0}, sd1={sd1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            return (f"mu0={mu0}, mu1={mu1},sd0={sd0}, sd1={sd1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.gamma_mean_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        poisson_rate_options = [make_gamma_mean_history_label(**entry) for entry in st.session_state.gamma_mean_history]
        selected_label = st.selectbox("Choose a past input set:", poisson_rate_options, key="gamma_mean_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.gamma_mean_history
                                    if make_gamma_mean_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False

    if go or hist_submit:
        tabs = st.tabs(["Tabulate", "Power V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
            if hist_submit and selected_history:
                # Use selected history
                mu0= selected_history["mu0"]
                mu1= selected_history["mu1"]
                sd0= selected_history["sd0"]
                sd1= selected_history["sd1"]
                power = selected_history["power"]
                drpt = selected_history["drpt"]
                designEffect = selected_history["designEffect"]
                Q1= selected_history["Q1"]
                Q0= selected_history["Q0"]
            else:
                # Add current input to history
                new_entry = {
                    "mu0":mu0,
                    "mu1":mu1,
                    "sd0":sd0,
                    "sd1":sd1,
                    "power":power,
                    "drpt":drpt,
                    "designEffect":designEffect,
                    "m":m,
                    "ICC":ICC,
                    "method":design_method,
                    "Q1":Q1,
                    "Q0":Q0
                }
                st.session_state.gamma_mean_history.append(new_entry)

            confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
            out=[]

            for conf in confidenceIntervals:
                sample_size= nSampleGammaGLM(mu0=mu0, mu1=mu1, sd0=sd0, sd1=sd1,Q0=Q0, Q1=Q1, alpha=(1-(conf)), power=power, designEffect=designEffect, dropout=drpt)
                out.append(sample_size)

            df= pd.DataFrame({
                "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                "Sample Size": out
            })

            dds= nSampleGammaGLM(mu0=mu0, mu1=mu1, sd0=sd0, sd1=sd1,Q0=Q0, Q1=Q1, alpha=0.05,designEffect=designEffect,dropout=drpt)

            st.write(f"The study would require a total sample size of:")
            st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <div style="
                    font-size: 36px;
                    font-weight: bold;
                    background-color: #48D1CC;
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;">
                    {int(dds)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write(f""" number of individuals (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q0)}</span> and <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q1)}</span> individuals respectively in control and intervention group with unequal Sample size ratio= {round(Q0,2)}:{round(Q1,2)} % respectively) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for detecting a true difference in means {mu0} and {mu1} respectively for control and intervention groups. The calculation is base on the assumeption that the population variances are {sd0} and {sd1} respectively for control and Intervention groups, with the design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out rate.""",unsafe_allow_html=True)
            st.subheader("List of Sample Sizes at other Confidence Levels")
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
                    ss = nSampleGammaGLM(mu0=mu0, mu1=mu1, sd0=sd0, sd1=sd1,Q0=Q0, Q1=Q1, alpha=(1-(conf)), power=power_val, designEffect=designEffect, dropout=drpt) 
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
        st.markdown("### **Sample Size Formula for Comparing Two Means under Gamma Distribution (GLM with Log-link)**")

        st.latex(r"""
            n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot 
            \left( \frac{1}{Q_1 \kappa_1} + \frac{1}{Q_0 \kappa_0} \right)}
            {\left[ \log \left( \frac{\mu_1}{\mu_0} \right) \right]^2} 
            \times \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("### **Relation between Shape (κ), Mean (μ), and Standard Deviation (σ):**")
        st.latex(r"""
            \kappa = \frac{\mu^2}{\sigma^2}
        """)

        st.markdown("### **Design Effect Calculation (if clusters are used):**")
        st.latex(r"""
            DE = 1 + (m - 1) \times ICC
        """)

        st.subheader("📌 Description of Parameters")

        st.markdown("""
        - **Z_{1-alpha/2}**: Z-value for two-sided significance level.
        - **Z_{1-beta}**: Z-value corresponding to desired power.
        - **mu_0**: Mean in the control group.
        - **mu_1**: Mean in the treatment group.
        - **sigma_0**: Standard deviation in the control group.
        - **sigma_1**: Standard deviation in the treatment group.
        - **kappa_0 (= mu_0^2 / sigma_0^2)**: Gamma shape parameter for control.
        - **kappa_1 (= mu_1^2 / sigma_1^2)**: Gamma shape parameter for treatment.
        - **Q_0**: Proportion of participants in the control group.
        - **Q_1**: Proportion of participants in the treatment group.
        - **DE**: Design Effect (to adjust for cluster sampling, if applicable).
        - **m**: Average number of subjects per cluster.
        - **ICC**: Intra-class correlation coefficient.
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)
        st.subheader("📌 References")

        st.markdown("""
        1. Cundill, Bonnie, and Neal D E Alexander. “Sample size calculations for skewed distributions.” BMC medical research methodology vol. 15 28. 2 Apr. 2015, doi:10.1186/s12874-015-0023-0
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

