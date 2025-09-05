import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.special import erf

def main():
#    st.set_page_config(page_title="StudySizer | One-way ANOVA",
#                    page_icon="🧮")
#
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Options")

    chooseButton= st.sidebar.radio("Choose Method", options=["Help", "Direct Method", "Non-central F-Distribution Method"],index=0)
    if chooseButton == "Help":
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
        st.title("Approaches to Sample Size Calculation for Mean Comparison Across more than Two Groups (One-way ANOVA)")
        st.markdown("---")

        st.header("Summary Table")

        st.markdown("""
        | Feature                        | Approach 1 (Direct Method)                   | Approach 2 (Non-central F)                                      |
        |--------------------------------|----------------------------------------------|-----------------------------------------------------------------|
        | **What input is needed?**      | Group means, SDs                             | Just effect size (f)                                            |
        | **Pilot data needed?**         | Yes                                          | No                                                              |
        | **When to use?**               | If you already have pilot or published data  | When no prior data is available                                 |
        | **Ease of use**                | Moderate (some calculations needed)          | Simple (choose f)                                               |
        | **Simplicity**                 | Simple (once inputs are known)               | Complex (needs F inverse calculations)                          |
        | **Accuracy**                   | Data-driven, very precise                    | Accurate for ANOVA but depends on assumed effect size.          |
        | **Example**                    | Comparing HbA1c means from 3 diet groups     | Comparing blood pressure in 4 drug groups without prior data.   |
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        st.header("Approach 1: Direct Formula Using Group Means and Standard Deviations")

        st.markdown("""
        This method calculates the sample size directly using the expected or observed **group means and standard deviations**.
        """)

        st.subheader("Formula")
        st.latex(r"n = \frac{(k - 1) + k(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \bar{\sigma}^2}{\sum_{i=1}^{k} (\mu_i - \bar{\mu})^2}")

        st.subheader("Use When")
        st.markdown("""
        - You have **pilot study data** or previous study estimates of means and SDs.  
        - You want a **data-driven** sample size calculation.  
        - You are evaluating real effect magnitude, not generalized sizes.
        """)

        st.subheader("Example")
        st.markdown("""
        Suppose you want to compare **3 diet interventions** on weight loss after 3 months. From a pilot study:

        - Group means (kg): 2.0, 3.5, 5.0  
        - Pooled SD: 1.2  
        - Desired power: 80% (Z = 0.84), α = 0.05 (Z = 1.96)

        By plugging values into the formula, you can calculate the per-group sample size.
        """)

        st.markdown("---")

        st.header("Approach 2: Non-central F-distribution Method")

        st.markdown("""
        This approach is **statistically rigorous** and matches the actual test used in ANOVA. It is especially useful if you **do not have pilot data**, but want to rely on a standardized measure of effect size.
        """)

        st.subheader("Power Formula")
        st.latex(r"\text{Power} = 1 - F^{-1}_{\text{crit}, df_1, df_2}(\alpha, \lambda)")
        st.markdown("""Where: """)
        st.latex(r"""F_{\text{crit}, df_1, df_2}^{-1}=\; \text{is the inverse cumulative distribution function (CDF) of the central F-distribution}""")
        st.latex(r"""df_1 = k - 1=\; \text{degrees of freedom between groups}""")
        st.latex(r"""df_2 = k(n - 1)=\; \text{degrees of freedom within groups}""")
        st.latex(r"""\lambda = f \cdot k \cdot n=\; \text{is the noncentrality parameter}""")
        st.latex(r"""n=\; \text{is the per-group sample size.}""")
        st.latex(r"""k=\; \text{is the number of groups.}""")
        st.latex(r"""f=\sqrt{\frac{\eta^2}{1-\eta^2}}=\; \text{is the Cohen's f : Effect size}""")
        st.latex(r"""\eta=\frac{SS_{\text{Treatment}}}{SS_{\text{Total}}}""")

        st.subheader("Choosing Effect Size (Cohen’s f)")

        st.markdown("Cohen’s f tells us how large the group differences are relative to the overall variability. It is related to a statistic called η² (eta squared), which measures the proportion of total variation explained by group differences:")
        st.latex(r"""f=\sqrt{\frac{\eta^2}{1-\eta^2}}""")
        st.markdown("""
        - **Small effect (f = 0.10, η² ≈ 0.01):** Use when expecting subtle group differences.  
        - **Medium effect (f = 0.25, η² ≈ 0.06):** Use when prior research or pilot data suggest moderate differences.  
        - **Large effect (f = 0.40, η² ≈ 0.14):** Use when expecting strong differences between groups.  
        
        When pilot or prior data are available, compute η² directly:
        - Calculate **SS(Treatment)** and **SS(Total)** from ANOVA.  
        - Compute η² = SS(Treatment) / SS(Total).  
        - Convert to f using f = sqrt(η² / (1 - η²)).
        """)

        st.subheader("Use When")
        st.markdown("""
        - You want **high statistical accuracy**, particularly in small samples.  
        - You want to base the calculation on **η² or Cohen’s f**.  
        - You are conducting **advanced planning or simulations**.
        """)

        st.subheader("Example")
        st.markdown("""
        Suppose you are comparing **average systolic blood pressure in 4 drug groups.**

        - You expect only small differences between drugs → use 𝑓 = 0.10. This will require a large sample size per group.
        - If you believe differences will be clinically meaningful but moderate → use 𝑓 = 0.25. Sample size per group will be smaller.
        - If you expect one drug to be much better than the others → use 𝑓 = 0.40. Only a modest sample per group will be needed.
    
        Using this effect size in the non-central F formula, the required per-group sample size is calculated.
        """)

        st.subheader("Practical Interpretation")
        st.markdown("""
        - Choosing the effect size depends on **clinical judgment** and **what difference is meaningful in practice.**
        - For example, in diabetes research, a **0.5% HbA1c difference** may already be clinically important, so a **medium effect size** is often chosen.
        - In blood pressure studies, a **5 mmHg reduction** may be considered meaningful.
                    
        This method is ideal when you want to plan a study **without exact pilot means**, but still want to account for the likely size of group differences.
        """)

        st.markdown("---")
        st.markdown("**Developed by [Rajesh Majumder]**")
        st.markdown("**Email:** rajeshnbp9051@gmail.com")
        st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")

    elif chooseButton=="Direct Method":

        st.title("Sample Size Calculation for One-way ANOVA (Direct Formula Method)")
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
        # Function to compute sample size using Direct Method (Approach 2)
        def direct_anova_sample_size(means, sds, z_alpha, z_beta, method="equal", dropout=0, design_effect=1):
            k = len(means)
            grand_mean = np.mean(means)

            # Between-group variation
            ss_between = sum((m - grand_mean)**2 for m in means)

            # Within-group variance
            if method == "equal":
                pooled_var = np.mean([sd**2 for sd in sds])
            else:  # unequal SDs
                pooled_var = sum(sd**2 for sd in sds) / k

            numerator = (k - 1) + k * ((z_alpha + z_beta) ** 2) * pooled_var
            n_per_group = numerator / ss_between
            n_per_group_design = n_per_group * design_effect
            n_per_group_adjusted = n_per_group_design / (1 - dropout)

            return round(n_per_group_adjusted), round(n_per_group_adjusted) * k

        # Initialize history store
        if "anova1_history" not in st.session_state:
            st.session_state.anova1_history = []

        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Input Parameters")

        # Sidebar Inputs
        approach = st.sidebar.selectbox("Select Method", ["Equal SDs", "Unequal SDs"])
        k = st.sidebar.number_input("Number of Groups", min_value=2, value=3,format="%.6g",help="Enter an integer value (e.g., 3)")
        means_input = st.sidebar.text_input(f"Enter {k} Group Means (comma-separated)", value="10,12,15")
        sds_input = st.sidebar.text_input(f"Enter {k} SDs (comma-separated)", value="3,3,3")
        power = st.sidebar.number_input("Power(%)", min_value=1.0, max_value=99.9, value=80.0,format="%.6g",help="Enter a percentage value (e.g., 80%)")
        dropout = st.sidebar.number_input("Drop-out Rate (%)", min_value=0.0, max_value=100.0, value=0.0,format="%.6g",help="Enter a percentage value (e.g., 1%)") / 100

        # Design effect options
        st.sidebar.markdown("### Design Effect")
        de_mode = st.sidebar.radio("Choose Method for Design Effect:", options=['Given','Calculate'])
        if de_mode == "Given":
            design_effect = st.sidebar.number_input("Design Effect (Given)", min_value=1.0, value=1.0,format="%.6g",help="Enter a decimal value (e.g., 1.5)")
            m, rho = None, None
        else:
            m = st.sidebar.number_input("Number of Clusters (m)", min_value=2, value=4,format="%.6g",help="Enter an integer value (e.g., 4)")
            rho = st.sidebar.number_input("Intraclass Correlation (ICC) for clustering", min_value=0.0, max_value=1.0, value=0.05,format="%.6g",help="Enter a decimal value (e.g., 0.05)")
            design_effect = 1 + (m - 1) * rho
            col1,col2,col3=st.columns(3)
            col1.metric("Cluster Size (m)",value=m)
            col2.metric("Intra Class Correlation (ICC)",value=rho)
            col3.metric("Design Effect",value= round(design_effect,2))

        calculate = st.button("Calculate Sample Size")

        # Helper to generate label for dropdown
        def make_anova1_history_label(modes,k, means, sds, design_effect, power, m=None, rho=None, method="Given", dropout=0, **kwargs):
            # includes method in label
            means_str = ','.join(map(str, means))
            sds_str = ','.join(map(str, sds))
            if method == "Given":
                return f"Method={modes},Groups={k}, Means={means_str}, SDs={sds_str}, Power={power}%, Dropout={round(dropout * 100)}%, DE(Given)={round(design_effect, 2)}"
            else:
                return f"Method={modes},Groups={k}, Means={means_str}, SDs={sds_str}, Power={power}%, Dropout={round(dropout * 100)}%, m={m}, ICC={rho}, DE(Calculated)={round(design_effect, 2)}"

        # Select from history
        selected_history = None
        selected_label = None

        if st.session_state.anova1_history:
            st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
            history_options = [make_anova1_history_label(**{**entry, 'power': entry.get('power', power)}) for entry in st.session_state.anova1_history]
            selected_label = st.selectbox("Choose a past input set:", history_options, key="anova1_selector")

            if selected_label:
                selected_history = next((item for item in st.session_state.anova1_history
                                    if make_anova1_history_label(**{**item, 'power': item.get('power', power)}) == selected_label), None)
                hist_submit = st.button("🔁 Recalculate from Selected History")
            else:
                hist_submit = False
        else:
            hist_submit = False

        if calculate or hist_submit:
            try:
                tabs = st.tabs(["Tabulate", "Power V/s Confidence Table" ,"Visualisation"])
                with tabs[0]:
                    if selected_history and hist_submit:
                        approach= selected_history["modes"]
                        means = selected_history["means"]
                        sds = selected_history["sds"]
                        k = selected_history["k"]
                        design_effect = selected_history["design_effect"]
                        dropout = selected_history["dropout"]
                        de_mode = selected_history["method"]
                    else:
                        means = [float(x.strip()) for x in means_input.split(",")]
                        sds = [float(x.strip()) for x in sds_input.split(",")]
                        assert len(means) == k and len(sds) == k
                        method = "Given" if de_mode == "Given" else "Calculated"
                        new_entry = {
                            "modes": approach,
                            "means": means,
                            "sds": sds,
                            "k": k,
                            "design_effect": design_effect,
                            "m": m,
                            "rho": rho,
                            "method": de_mode,
                            "power": power,
                            "dropout": dropout
                        }
                        if new_entry not in st.session_state.anova1_history:
                            st.session_state.anova1_history.append(new_entry)

                    method_type = "equal" if approach == "Equal SDs" else "unequal"
                    power_z = stats.norm.ppf(power / 100)

                    confidence_levels = [0.8, 0.9, 0.95, 0.97, 0.99, 0.999, 0.9999]
                    st.subheader("🧮 Sample Size at Different Confidence Levels")
                    results = []
                    for conf in confidence_levels:
                        z_alpha = stats.norm.ppf(1-((1-conf)/2))
                        n_pg, total_n = direct_anova_sample_size(means, sds, z_alpha, power_z, method_type, dropout, design_effect)
                        results.append({"Confidence Leves(%)": round((conf*100),2), "Sample/Group": int(n_pg), "Total Sample": int(total_n)})  #"Z(alpha)": z_alpha, 
                    results_df = pd.DataFrame(results)
                    results_df["Confidence Leves(%)"] = results_df["Confidence Leves(%)"].map(lambda x: f"{x:.2f}")

                    def highlight_95(row):
                        return ['background-color: lightgreen' if row['Confidence Leves(%)'] == "95.00" else '' for _ in row]
                    st.dataframe(results_df.style.apply(highlight_95, axis=1))
                    #st.dataframe(pd.DataFrame(results))

                    st.markdown("---")
                    st.subheader("🧾 Calculation Details")
                    st.write(f"**Means:** {means}")
                    st.write(f"**Standard Deviations:** {sds}")
                    st.write(f"**Grand Mean:** {round(np.mean(means), 2)}")
                    st.write(f"**Drop-out adjusted:** {round(dropout * 100, 1)}%")
                    st.write(f"**Z(beta):** {power_z}")
                    st.write(f"**Power (1 - β):** {round(power, 1)}%")
                    st.write(f"**Design Effect used:** {round(design_effect, 3)}")
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
                            z_alpha = stats.norm.ppf(1-((1-conf)/2))
                            power_z = stats.norm.ppf(power_val)
                            nn,ss = direct_anova_sample_size(means, sds, z_alpha, power_z, method_type, dropout, design_effect)
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

            except Exception as e:
                st.error(f"Input error: {e}")

        st.markdown("---")
        with st.expander("Show the formula and the references"):
            st.subheader("📌 Formula (Direct Method)")
            st.latex(r"n = \frac{(k - 1) + k(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \bar{\sigma}^2}{\sum_{i=1}^{k} (\mu_i - \bar{\mu})^2} \times \frac{DE}{1 - \text{Dropout\%}}")

            st.markdown("### **Design Effect Calculation (if clusters are used):**")
            st.latex(r"""
            DE = 1 + (m - 1) \times ICC
            """)

            st.subheader("📌 Description of Parameters")
            st.markdown("- k: number of groups")
            st.markdown("- mu_i: group means")
            st.markdown("- mu_bar: grand mean")
            st.markdown("- sigma^2: average of group variances (equal/unequal)")
            st.markdown("- (Z_{1-alpha/2}, Z_{1-beta}\): critical values for significance level and power")
            st.markdown("- DE = Design Effect")
            st.markdown("- Dropout%: Anticipated percentage of dropout in the study.")

            st.markdown("---")
            st.subheader("📌 Notes")
            st.markdown("- This calculator assumes a **balanced design** (equal n per group).")
            st.markdown(""""
            - <span style="font-weight: bold; font-size: 26px;">Note that</span>, when your terget is the multiple comparisons use  **Bonferroni correction**:           
            """,unsafe_allow_html=True)
            st.latex(r"""
            \alpha_{\text{adjusted}}=\frac{\alpha}{\text{Number of Comparisons}}
            """)
            st.markdown("""  to adjust the significance level. This adjustment helps to control family-wise errow rate (FWER). Others are **Sidak Correction**,**Holm-Bonferroni**,**Benjamini-Hochberg**.""")


            st.markdown("---")
            st.subheader("📌 References")
            st.markdown("""
                        1. **Chow, S.C., Shao, J., & Wang, H. (2008).** Sample Size Calculations in Clinical Research (2nd Ed.) [Chapter: One-way ANOVA]
                        2. **Machin, D., Campbell, M. J., Tan, S. B., & Tan, S. H. (2018).** Sample Size Tables for Clinical Studies (3rd Ed.)
                        """)

        st.markdown("---")
        st.subheader("Citation")
        from datetime import datetime
        # Get current date and time
        now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        # Citation with access timestamp
        st.markdown(f"""
        *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.netlify.app/](https://studysizer.netlify.app/). Accessed on {now}.*
        """)
        
    else:
        # Streamlit App
        st.title("Sample Size Calculation for One way ANOVA (Non central F-didtribution Method)")
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
        ## Functuion
        def calculate_anova_sample_size(effect_size, alpha, power, k,dpt):

            # Get Z-scores for alpha and power
            #z_alpha = stats.norm.ppf(1 - (1-alpha) / 2)  # Two-tailed test
            #z_beta = stats.norm.ppf(power)
        
            # Start with an initial guess for sample size
            n = 2
        
            while True:
                # Degrees of freedom
                df1 = k - 1  # Between-group df
                df2 = k * (n - 1)  # Within-group df
            
                # Compute the critical F-value
                f_crit = stats.f.ppf(1 - (1-alpha), df1, df2)
            
                # Compute the noncentrality parameter
                lambda_ncp = effect_size * np.sqrt(k * n)
            
                # Compute power using the noncentral F-distribution
                power_calc = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_ncp**2)
            
                # Check if computed power meets or exceeds desired power
                if power_calc >= power:
                    break
            
                # Increase sample size and try again
                n += 1
                n= n/(1-dpt)

            # Return per-group and total sample size
            return lambda_ncp**2,f_crit,df1,df2,power_calc,n, k * n,k

        # Initialize history store
        if "anova_history" not in st.session_state:
            st.session_state.anova_history = []

        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Input Parameters")

        esize = st.sidebar.number_input("Effect size (Cohen's f)",value=0.25,min_value=0.0,max_value=1.0,format="%.6g",help="Enter a decimal value (e.g., 0.25)")
        st.sidebar.text("0.10 = Small effect size\n 0.25= Medium effect size\n 0.40= Large effect size")
        power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=99.99,format="%.6g",help="Enter a percentage value (e.g., 5%)")
        KK=st.sidebar.number_input("Number of groups (k)",value=5,min_value=3,format="%.6g",help="Enter an integer value (e.g., 3)")
        drpt= st.sidebar.number_input("Drop-Out (%)",min_value=0.0,value=0.0,max_value=50.0,format="%.6g",help="Enter an integer value (e.g., 1%)")
        go= st.button("Calculate Sample Size")


        # Helper to generate label for dropdown
        def make_anova_history_label(esize, KK, power, drpt):
                return f"Cohen's f={esize}, k={KK}, Power={power}%, DropOut={drpt}%"

        # Select from history
        selected_history = None
        selected_label = None

        if st.session_state.anova_history:
            st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
            anova_options = [make_anova_history_label(**entry) for entry in st.session_state.anova_history]
            selected_label = st.selectbox("Choose a past input set:", anova_options, key="anova_history_selector")

            if selected_label:
                selected_history = next((item for item in st.session_state.anova_history
                                        if make_anova_history_label(**item) == selected_label), None)
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
                    esize= selected_history["esize"]
                    KK= selected_history["KK"]
                    power = selected_history["power"]
                    drpt = selected_history["drpt"]
                else:
                    # Add current input to history
                    new_entry = {
                        "esize":esize,
                        "KK":KK,
                        "power":power,
                        "drpt":drpt
                    }
                    st.session_state.anova_history.append(new_entry)

                confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
                out=[]

                for conf in confidenceIntervals:
                    sample_size= calculate_anova_sample_size(effect_size=esize, alpha=conf, power=(power/100), k=KK,dpt=(drpt/100))
                    out.append(sample_size)

                df= pd.DataFrame({
                    "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                    "Sample sise par group wise": [row[5] for row in out],
                    "Total Sample sise": [row[6] for row in out],
                    "No. of groups": [row[7] for row in out],
                    "Noncentrality parameter": [row[0] for row in out],
                    "F-critical value": [row[1] for row in out],
                    "Upper df.": [row[2] for row in out],
                    "Lower df.": [row[3] for row in out],
                    "Calculated Power": [row[4] for row in out]
                })

                dds= calculate_anova_sample_size(effect_size=esize, alpha=0.95, power=(power/100), k=KK,dpt=(drpt/100))
            
                st.write(f"The study would need a total sample size of:")
                st.markdown(f"""
                <div style="display: flex; justify-content: center;">
                    <div style="
                        font-size: 36px;
                        font-weight: bold;
                        background-color: #48D1CC;
                        padding: 10px;
                        border-radius: 10px;
                        text-align: center;">
                        {int(dds[6])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"""participants(i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds[5])}</span> participants at each group) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for comparing {KK} different group means, where the non-centrality parameter is {round(dds[0],4)}, F-test critical value is {round(dds[1],4)} with numerator and denominator dfs are {int(dds[2])} and {int(dds[3])} respectively  where drop-out sample percentage is **{(drpt)}%**.""",unsafe_allow_html=True)
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
                        ss = calculate_anova_sample_size(effect_size=esize, alpha=conf, power=power_val, k=KK,dpt=(drpt/100))  
                        cross_table.iloc[i, j] = ss[6]
                # Label table
                cross_table.index.name = "Confidence Level (%)"
                cross_table.columns.name = "Power (%)"

                st.dataframe(cross_table)
                st.write("**Cell values are Total SSample size; Rows are Confidence Levels; Columns are Powers**")
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
                plt.title("Total Sample Size vs Power and Alpha Level (Multiple Lines)")
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

        st.markdown("---")  # Adds a horizontal line for separation
        with st.expander("Show the formula and the references"):
            st.subheader("📌 Formula for Sample Size Calculation")

            st.markdown("### **One-way ANOVA Test Sample Size Formula**")

            st.markdown("The sample size for One-way ANOVA is calculated using noncentral F-distribution:")
            st.latex(r"""
            \text{Power} = 1 - F_{\text{crit}, df_1, df_2}^{-1} (\alpha, \lambda)
            """)

            st.subheader("📌 Description of Parameters")

            st.latex(r"""
            F_{\text{crit}, df_1, df_2}^{-1}=\; \text{is the inverse cumulative distribution function (CDF) of the central F-distribution}
            """)
            st.latex(r"""
            df_1 = k - 1=\; \text{degrees of freedom between groups}
            """)
            st.latex(r"""
            df_2 = k(n - 1)=\; \text{degrees of freedom within groups}
            """)
            st.latex(r"""
            \lambda = f \cdot k \cdot n=\; \text{is the noncentrality parameter}
            """)
            st.latex(r"""
            n=\; \text{is the per-group sample size.}
            """)
            st.latex(r"""
            k=\; \text{is the number of groups.}
            """)
            st.latex(r"""
            f=\sqrt{\frac{\eta^2}{1-\eta^2}}=\; \text{is the Cohen's f : Effect size}
            """)
            st.latex(r"""
            \eta=\frac{SS_{\text{Treatment}}}{SS_{\text{Total}}}
            """)

            st.subheader("📌 References")

            st.markdown("""
            1. **Cohen, J.** A power primer. Psychological bulletin vol. 112,1 (1992): 155-9. doi:10.1037//0033-2909.112.1.155
            2. **Jan, Show-Li, and Gwowen Shieh.** Sample size determinations for Welch's test in one-way heteroscedastic ANOVA. The British journal of mathematical and statistical psychology vol. 67,1 (2014): 72-93. doi:10.1111/bmsp.12006
            3. **Bujang, Mohamad Adam.** A Step-by-Step Process on Sample Size Determination for Medical Research. The Malaysian journal of medical sciences : MJMS vol. 28,2 (2021): 15-27. doi:10.21315/mjms2021.28.2.2
            """)

            st.markdown(""""
            <span style="font-weight: bold; font-size: 26px;">Note that</span>, when your terget is the multiple comparisons use  **Bonferroni correction**:           
            """,unsafe_allow_html=True)
            st.latex(r"""
            \alpha_{\text{adjusted}}=\frac{\alpha}{\text{Number of Comparisons}}
            """)
            st.markdown("""to adjust the significance level. This adjustment helps to control family-wise errow rate (FWER). Others are **Sidak Correction**,**Holm-Bonferroni**,**Benjamini-Hochberg**.""")


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